"""Étape 4b - Évaluation du détecteur à fenêtre glissante.

Pré-requis : le CNN « avion / fond » entraîné avec 03_classification.py :
    python 03_classification.py --data data/fenetres --arch resnet50 --size 96 --name detecteur_resnet50
    python 03_classification.py --data data/fenetres --arch resnet18 --size 96 --name detecteur_resnet18

1. Le CNN note toutes les fenêtres de chaque image (partie lente). Ces scores
   « bruts » sont gardés dans brut_val.pt / brut_test.pt : relancer le script
   ne refait pas ce calcul.
2. Réglages choisis sur la VALIDATION (meilleur F1 = compromis précision / rappel) :
     - le seuil de score :
         seuil haut -> peu de fausses alarmes mais des avions ratés (rappel bas)
         seuil bas  -> presque tous les avions trouvés mais des fausses alarmes
     - avec ou sans suppression des boîtes incluses dans une meilleure boîte
       (petites fenêtres posées sur une aile ou un moteur d'un avion déjà détecté).
3. Évaluation finale sur le TEST : une détection est correcte si elle recouvre
   un vrai avion avec IoU >= 0.5.
     précision = détections correctes / détections
     rappel    = avions trouvés / avions réels
4. Images de test avec les détections dessinées :
     vert = bonne détection, rouge = fausse alarme, jaune = avion manqué

Sortie dans outputs/detection/<name>/ :
  rapport_detection.txt, metrics.json, courbe_seuil.png, exemples/*.jpg

Usage : python 06_detection.py --name detecteur_resnet50 [--max-val 120]
"""
import argparse
import glob
import importlib
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_tensor

from src.detection import apparier, post_traitement, scorer_fenetres
from src.voc import parse_xml

clf = importlib.import_module("03_classification")

SCORE_MIN = 0.3
SEUILS = np.array([0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.93, 0.95,
                   0.97, 0.98, 0.99, 0.995, 0.998, 0.999])
# Réglages du post-traitement testés sur la validation (en plus du seuil)
IOUS_NMS = (0.2, 0.3, 0.4)
INCLUSIONS = (None, 0.5, 0.7)          # suppression des boîtes incluses (None = non)
FUSIONS = (False, True)                # moyenne des fenêtres qui votent pour le même avion
REFERENCE = (0.3, None, False)         # post-traitement de départ (NMS seule), pour comparer


def decrire(cfg):
    iou, incl, fus = cfg
    return (f"NMS IoU {iou}, suppression des boîtes incluses : {'non' if incl is None else f'oui (> {incl:.0%})'}, "
            f"fusion des fenêtres : {'oui' if fus else 'non'}")


def charger_modele(path, device):
    ckpt = torch.load(path, map_location=device)
    m = clf.build_model(ckpt["arch"], len(ckpt["classes"]))
    m.load_state_dict(ckpt["state_dict"])
    m = m.to(device).eval().to(memory_format=torch.channels_last)
    return m, ckpt


def scores_bruts(model, idx, taille, dossier, max_images, cache):
    """Partie lente : score de toutes les fenêtres, image par image (gardé en cache)."""
    if os.path.exists(cache):
        print(f"  scores déjà calculés -> {cache}")
        return torch.load(cache)
    xmls = sorted(glob.glob(os.path.join(dossier, "*.xml")))[:max_images]
    res = []
    t0 = time.time()
    for n, x in enumerate(xmls, 1):
        a = parse_xml(x, dossier)
        img = Image.open(a.image_path).convert("RGB")
        boites, scores = scorer_fenetres(model, to_tensor(img), idx, taille_entree=taille, score_min=SCORE_MIN)
        gt = torch.tensor([[b.xmin, b.ymin, b.xmax, b.ymax] for b in a.boxes], dtype=torch.float32).reshape(-1, 4)
        res.append({"xml": x, "boites": boites, "scores": scores, "gt": gt})
        if n % 20 == 0 or n == len(xmls):
            print(f"  {n}/{len(xmls)} images ({(time.time() - t0) / n:.1f} s/image)")
    s_img = (time.time() - t0) / max(len(xmls), 1)
    data = {"images": res, "s_par_image": s_img}
    torch.save(data, cache)
    return data


def appariements(data, cfg, seuil_iou):
    """Post-traitement au seuil minimal puis appariement avec les vrais avions.
    Le seuil, la NMS et l'appariement traitent les boîtes par score décroissant
    (et la fusion utilise toujours toutes les fenêtres >= SCORE_MIN) : filtrer
    ensuite à un seuil plus haut donne le même résultat que tout refaire à ce
    seuil -> on calcule une seule fois par réglage."""
    iou_nms, inclusion, fusion = cfg
    tous_s, tous_tp, n_gt, par_image = [], [], 0, []
    for r in data["images"]:
        b, s = post_traitement(r["boites"], r["scores"], SCORE_MIN, iou_nms, inclusion, fusion)
        _, s2, tp, _ = apparier(b, s, r["gt"], seuil_iou)
        tous_s.append(s2); tous_tp.append(tp); n_gt += len(r["gt"])
        par_image.append((r["xml"], b, s, r["gt"]))
    return torch.cat(tous_s).numpy(), torch.cat(tous_tp).numpy(), n_gt, par_image


def prf(scores, tp, n_gt, seuil):
    garde = scores >= seuil
    n_tp = int(tp[garde].sum()); n_pred = int(garde.sum())
    p = n_tp / n_pred if n_pred else 0.0
    r = n_tp / n_gt if n_gt else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1, n_tp, n_pred - n_tp, n_gt - n_tp


def dessiner(xml, boites, scores, gt, seuil, seuil_iou, path):
    a = parse_xml(xml, os.path.dirname(xml))
    img = Image.open(a.image_path).convert("RGB")
    d = ImageDraw.Draw(img)
    garde = scores >= seuil
    b, s, tp, trouves = apparier(boites[garde], scores[garde], gt, seuil_iou)
    for g, ok in zip(gt.tolist(), trouves.tolist()):
        if not ok:
            d.rectangle(g, outline=(255, 215, 0), width=3)                      # avion manqué
    for box, sc, ok in zip(b.tolist(), s.tolist(), tp.tolist()):
        d.rectangle(box, outline=(40, 200, 70) if ok else (230, 40, 40), width=3)
        d.text((box[0] + 3, box[1] + 2), f"{sc:.2f}", fill=(255, 255, 255))
    img.save(path, quality=92)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="detecteur_resnet50",
                   help="dossier du CNN avion/fond dans outputs/classification/ (detecteur_resnet18 ou detecteur_resnet50)")
    p.add_argument("--images", default="data/images_detection")
    p.add_argument("--max-val", type=int, default=120, help="images de val pour choisir les réglages")
    p.add_argument("--iou-eval", type=float, default=0.5)
    p.add_argument("--n-exemples", type=int, default=12)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = charger_modele(os.path.join("outputs", "classification", args.name, "best_model.pt"), device)
    idx = ckpt["classes"].index("avion")
    taille = ckpt["size"]
    out = os.path.join("outputs", "detection", args.name)
    os.makedirs(os.path.join(out, "exemples"), exist_ok=True)

    print("Validation : scores des fenêtres")
    val = scores_bruts(model, idx, taille, os.path.join(args.images, "val"), args.max_val,
                       os.path.join(out, "brut_val.pt"))
    print("Test : scores des fenêtres")
    test = scores_bruts(model, idx, taille, os.path.join(args.images, "test"), 10**9,
                        os.path.join(out, "brut_test.pt"))

    # --- choix des réglages sur la validation ---
    courbes, meilleur = {}, None
    print("\nValidation : F1 selon le post-traitement (meilleur seuil pour chacun)")
    for iou in IOUS_NMS:
        for incl in INCLUSIONS:
            for fus in FUSIONS:
                cfg = (iou, incl, fus)
                s, tp, n, _ = appariements(val, cfg, args.iou_eval)
                courbes[cfg] = [prf(s, tp, n, t) for t in SEUILS]
                k = int(np.argmax([c[2] for c in courbes[cfg]]))
                print(f"  F1 {courbes[cfg][k][2]:.3f} (seuil {SEUILS[k]:g})  <- {decrire(cfg)}")
                if meilleur is None or courbes[cfg][k][2] > meilleur[2]:
                    meilleur = (cfg, float(SEUILS[k]), courbes[cfg][k][2], courbes[cfg][k])
    cfg, seuil, _, (pv, rv, fv, *_) = meilleur
    k_ref = int(np.argmax([c[2] for c in courbes[REFERENCE]]))
    f1_ref = courbes[REFERENCE][k_ref][2]
    print(f"\nRéglages choisis : seuil {seuil}, {decrire(cfg)}  (F1 val {fv:.3f} contre {f1_ref:.3f} avec la NMS seule)")

    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(SEUILS))   # seuils en abscisse à intervalles réguliers (ils sont serrés près de 1)
    c = courbes[cfg]
    for k, nom, col in ((0, "précision", "#2a6fdb"), (1, "rappel", "#e8792b"), (2, "F1", "#333333")):
        ax.plot(x, [v[k] for v in c], label=nom, color=col, linewidth=2, marker="o", markersize=3)
    if cfg != REFERENCE:
        ax.plot(x, [v[2] for v in courbes[REFERENCE]], color="#333333", linewidth=1, linestyle=":",
                label="F1 avec la NMS seule (départ)")
    i = int(np.where(SEUILS == seuil)[0][0])
    ax.axvline(i, color="#888888", linestyle="--", linewidth=1)
    ax.text(i, 0.02, f" seuil choisi {seuil}", color="#888888")
    ax.set_xticks(x); ax.set_xticklabels([f"{t:g}" for t in SEUILS], rotation=45, fontsize=8)
    ax.set_xlabel("Seuil de score"); ax.set_ylim(0, 1.02)
    ax.set_title("Validation : précision / rappel selon le seuil", loc="left")
    ax.spines[["top", "right"]].set_visible(False); ax.grid(color="#eeeeee"); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "courbe_seuil.png"), dpi=150); plt.close(fig)

    # --- évaluation finale sur le test ---
    s_te, tp_te, n_te, par_image = appariements(test, cfg, args.iou_eval)
    pr, rc, f1, n_tp, n_fp, n_fn = prf(s_te, tp_te, n_te, seuil)
    s_img = test["s_par_image"]
    txt = (f"Détecteur : fenêtre glissante + {ckpt['arch']} avion/fond (entrée {taille}px)\n"
           f"Post-traitement : {decrire(cfg)}\n"
           f"Seuil de score choisi sur la validation : {seuil}  (val : P {pv:.3f} R {rv:.3f} F1 {fv:.3f})\n"
           f"TEST ({len(par_image)} images, {n_te} avions, détection correcte si IoU >= {args.iou_eval})\n"
           f"  précision {pr:.3f}   rappel {rc:.3f}   F1 {f1:.3f}\n"
           f"  bonnes détections {n_tp}   fausses alarmes {n_fp}   avions manqués {n_fn}\n"
           f"  temps de détection : {s_img:.2f} s/image\n")
    print("\n" + txt)
    with open(os.path.join(out, "rapport_detection.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump({"arch": ckpt["arch"], "seuil": seuil, "iou_nms": cfg[0], "inclusion": cfg[1], "fusion": cfg[2],
                   "precision": pr, "rappel": rc, "f1": f1,
                   "bonnes_detections": n_tp, "fausses_alarmes": n_fp, "avions_manques": n_fn,
                   "secondes_par_image": round(s_img, 2)}, f, indent=1)

    for xml, boites, scores, gt in par_image[:args.n_exemples]:
        nom = os.path.splitext(os.path.basename(xml))[0]
        dessiner(xml, boites, scores, gt, seuil, args.iou_eval, os.path.join(out, "exemples", f"{nom}.jpg"))
    print(f"Résultats dans {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
