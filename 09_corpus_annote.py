"""Corpus de test annoté par le modèle (demandé dans le rapport).

Pour chaque image de test : détection + classification, puis on enregistre
  - <image>.jpg : l'image avec les avions dessinés et leur classe prédite
  - <image>.xml : les prédictions au format Pascal VOC (même format que les
                  annotations fournies, avec en plus le score de confiance)
et une évaluation de la chaîne complète : un avion est « correct » si une
détection le recouvre (IoU >= 0.5) ET a la bonne classe.

et detections.csv (une ligne par avion détecté, toutes images confondues).
Si le dossier ne contient pas d'annotations (corpus d'évaluation), seules les
prédictions sont produites. Si le programme est interrompu, le relancer avec
les mêmes options reprend là où il s'était arrêté.

Sortie : outputs/corpus_annote/ (+ corpus_annote.zip à mettre en lien dans le rapport)

Usage : python 09_corpus_annote.py [--images data/images_detection/test] [--out outputs/corpus_annote]
"""
import argparse
import glob
import json
import os
import shutil
import time
import xml.etree.ElementTree as ET
from collections import Counter

import torch
from PIL import Image
from torchvision.ops import box_iou

from src.pipeline import Pipeline, dessiner
from src.voc import parse_xml


def ecrire_xml(path, nom_image, img, dets):
    a = ET.Element("annotation")
    ET.SubElement(a, "filename").text = nom_image
    t = ET.SubElement(a, "size")
    for k, v in (("width", img.width), ("height", img.height), ("depth", 3)):
        ET.SubElement(t, k).text = str(v)
    for d in dets:
        o = ET.SubElement(a, "object")
        ET.SubElement(o, "name").text = d["classe"]
        ET.SubElement(o, "confidence").text = f"{d['proba']:.4f}"
        ET.SubElement(o, "detection_score").text = f"{d['score']:.4f}"
        b = ET.SubElement(o, "bndbox")
        for k, v in zip(("xmin", "ymin", "xmax", "ymax"), d["boite"]):
            ET.SubElement(b, k).text = str(v)
    ET.indent(a)
    ET.ElementTree(a).write(path, encoding="utf-8", xml_declaration=True)


def ecrire_json(args, images, pipe, classes):
    """predictions.json : toutes les prédictions du corpus dans un seul fichier,
    reconstruit à partir des .xml (donc complet même après une reprise)."""
    res = []
    for chemin in images:
        xml = os.path.join(args.out, os.path.splitext(os.path.basename(chemin))[0] + ".xml")
        racine = ET.parse(xml).getroot()
        taille = racine.find("size")
        avions = []
        for o in racine.findall("object"):
            b = o.find("bndbox")
            avions.append({
                "classe": o.findtext("name"),
                "confiance_classe": float(o.findtext("confidence")),
                "score_avion": float(o.findtext("detection_score")),
                "boite": {k: int(float(b.findtext(k))) for k in ("xmin", "ymin", "xmax", "ymax")},
            })
        res.append({"image": os.path.basename(chemin),
                    "largeur": int(taille.findtext("width")), "hauteur": int(taille.findtext("height")),
                    "nb_avions": len(avions), "avions": avions})
    donnees = {
        "modele": {"detecteur": args.detecteur, "classifieur": args.classifieur,
                   "seuil_detection": pipe.seuil, "iou_nms": pipe.iou_nms,
                   "suppression_incluses": pipe.inclusion, "fusion_fenetres": pipe.fusion},
        "resume": {"nb_images": len(res), "nb_avions": sum(r["nb_avions"] for r in res),
                   "avions_par_classe": dict(sorted(classes.items(), key=lambda kv: int(kv[0][1:])))},
        "images": res,
    }
    chemin_json = os.path.join(args.out, "predictions.json")
    with open(chemin_json, "w", encoding="utf-8") as f:
        json.dump(donnees, f, ensure_ascii=False, indent=1)
    print(f"Fichier JSON : {os.path.abspath(chemin_json)}")
    ecrire_soumission(donnees, os.path.join(args.out, "soumission.json"))


def ecrire_soumission(donnees, chemin):
    """Format demandé par le site de validation du prof :
    {"1.jpg": [{"class": "A2", "coordinates": {"xmin":..,"ymin":..,"xmax":..,"ymax":..}}, ...], ...}
    Toutes les images sont présentes, même celles sans avion (liste vide)."""
    soumission = {im["image"]: [{"class": a["classe"], "coordinates": a["boite"]} for a in im["avions"]]
                  for im in donnees["images"]}
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump(soumission, f, indent=2)
    print(f"Fichier à déposer sur le site de validation : {os.path.abspath(chemin)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", default=os.path.join("data", "images_detection", "test"))
    p.add_argument("--detecteur", default="detecteur_resnet18",
                   help="ResNet18 : meilleur F1 en détection et 3x plus rapide que ResNet50 (voir 07_comparaison.py)")
    p.add_argument("--classifieur", default="resnet50")
    p.add_argument("--out", default=os.path.join("outputs", "corpus_annote"))
    args = p.parse_args()

    pipe = Pipeline(detecteur=args.detecteur, classifieur=args.classifieur)
    os.makedirs(args.out, exist_ok=True)
    # toutes les images du dossier (on ignore les fichiers cachés, ex. ceux créés par macOS)
    images = sorted(f for f in glob.glob(os.path.join(args.images, "*"))
                    if f.lower().endswith((".jpg", ".jpeg", ".png")) and not os.path.basename(f).startswith("."))
    # reprise : les images déjà traitées (leur .xml existe) sont sautées
    deja = [c for c in images if os.path.exists(os.path.join(args.out, os.path.splitext(os.path.basename(c))[0] + ".xml"))]
    deja_set = set(deja)
    a_faire = [c for c in images if c not in deja_set]
    if deja:
        print(f"Reprise : {len(deja)} images déjà traitées, il en reste {len(a_faire)}")
    csv_path = os.path.join(args.out, "detections.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("image,classe,confiance_classe,score_avion,xmin,ymin,xmax,ymax\n")
    n_gt = n_det = n_bonne_boite = n_correct = 0
    erreurs_classe = Counter()
    t0 = time.time()
    for n, chemin in enumerate(a_faire, 1):
        nom = os.path.splitext(os.path.basename(chemin))[0]
        img = Image.open(chemin).convert("RGB")
        dets = pipe.analyser(img)
        dessiner(img, dets).save(os.path.join(args.out, nom + ".jpg"), quality=92)
        with open(csv_path, "a", encoding="utf-8") as f:
            for d in dets:
                f.write(f"{os.path.basename(chemin)},{d['classe']},{d['proba']:.4f},{d['score']:.4f},"
                        + ",".join(str(v) for v in d["boite"]) + "\n")
        # le .xml est écrit en dernier : s'il existe, l'image est entièrement traitée
        ecrire_xml(os.path.join(args.out, nom + ".xml"), os.path.basename(chemin), img, dets)

        xml_vrai = os.path.join(args.images, nom + ".xml")   # vraies annotations, si elles existent
        if os.path.exists(xml_vrai):
            vrais = parse_xml(xml_vrai, args.images).boxes
            gt = torch.tensor([[b.xmin, b.ymin, b.xmax, b.ymax] for b in vrais], dtype=torch.float32).reshape(-1, 4)
            pred = torch.tensor([d["boite"] for d in dets], dtype=torch.float32).reshape(-1, 4)
            sc = torch.tensor([d["score"] for d in dets])
            ordre = sc.argsort(descending=True)
            # appariement glouton (même logique que apparier() de src/detection.py) :
            # par score décroissant, chaque détection prend le vrai avion libre le plus recouvert
            if len(gt) and len(pred):
                ious = box_iou(pred[ordre], gt)
                pris = torch.zeros(len(gt), dtype=torch.bool)
                for i, k in enumerate(ordre.tolist()):
                    v = ious[i].clone(); v[pris] = -1
                    j = int(v.argmax())
                    if v[j] >= 0.5:
                        pris[j] = True
                        n_bonne_boite += 1
                        if dets[k]["classe"] == vrais[j].label:
                            n_correct += 1
                        else:
                            erreurs_classe[f"{vrais[j].label} -> {dets[k]['classe']}"] += 1
            n_gt += len(gt)
        n_det += len(dets)
        if n % 20 == 0 or n == len(a_faire):
            print(f"  {len(deja) + n}/{len(images)} images ({(time.time() - t0) / n:.1f} s/image)")

    # bilan sur tout le corpus (y compris les images d'une session précédente), à partir du CSV
    with open(csv_path, encoding="utf-8") as f:
        classes = Counter(l.split(",")[1] for l in f.read().splitlines()[1:] if l)
    par_classe = ", ".join(f"{c} : {n}" for c, n in sorted(classes.items(), key=lambda kv: int(kv[0][1:])))
    lignes = [f"Corpus annoté : {len(images)} images, {sum(classes.values())} avions détectés",
              f"Détecteur {args.detecteur} (seuil {pipe.seuil}) + classifieur {args.classifieur}",
              f"Avions par classe : {par_classe}"]
    if n_gt and deja:
        lignes += ["(scores ci-dessous calculés seulement sur les images traitées pendant cette session)"]
    if n_gt:
        p_ = n_correct / n_det if n_det else 0.0
        r_ = n_correct / n_gt
        f_ = 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0
        lignes += ["",
                   f"Chaîne complète (bonne boîte IoU >= 0.5 ET bonne classe), sur {n_gt} vrais avions :",
                   f"  précision {p_:.3f}   rappel {r_:.3f}   F1 {f_:.3f}",
                   f"  avions bien localisés : {n_bonne_boite}, dont bien classés : {n_correct} "
                   f"({n_correct / max(n_bonne_boite, 1):.1%})"]
        if erreurs_classe:
            lignes += ["  confusions de classe les plus fréquentes : " +
                       ", ".join(f"{k} ({v})" for k, v in erreurs_classe.most_common(5))]
    txt = "\n".join(lignes) + "\n"
    print("\n" + txt)
    with open(os.path.join(args.out, "resume.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    ecrire_json(args, images, pipe, classes)
    archive = shutil.make_archive(args.out, "zip", args.out)
    print(f"Corpus dans {os.path.abspath(args.out)}\nArchive : {os.path.abspath(archive)}")


if __name__ == "__main__":
    main()
