"""Corpus de test annoté par le modèle (demandé dans le rapport).

Pour chaque image de test : détection + classification, puis on enregistre
  - <image>.jpg : l'image avec les avions dessinés et leur classe prédite
  - <image>.xml : les prédictions au format Pascal VOC (même format que les
                  annotations fournies, avec en plus le score de confiance)
et une évaluation de la chaîne complète : un avion est « correct » si une
détection le recouvre (IoU >= 0.5) ET a la bonne classe.

Sortie : outputs/corpus_annote/ (+ corpus_annote.zip à mettre en lien dans le rapport)

Usage : python 09_corpus_annote.py [--images data/images_detection/test]
"""
import argparse
import glob
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
    images = sorted(glob.glob(os.path.join(args.images, "*.jpg")))
    n_gt = n_det = n_bonne_boite = n_correct = 0
    erreurs_classe = Counter()
    t0 = time.time()
    for n, chemin in enumerate(images, 1):
        nom = os.path.splitext(os.path.basename(chemin))[0]
        img = Image.open(chemin).convert("RGB")
        dets = pipe.analyser(img)
        dessiner(img, dets).save(os.path.join(args.out, nom + ".jpg"), quality=92)
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
        if n % 20 == 0 or n == len(images):
            print(f"  {n}/{len(images)} images ({(time.time() - t0) / n:.1f} s/image)")

    lignes = [f"Corpus annoté : {len(images)} images, {n_det} avions détectés",
              f"Détecteur {args.detecteur} (seuil {pipe.seuil}) + classifieur {args.classifieur}"]
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
    archive = shutil.make_archive(args.out, "zip", args.out)
    print(f"Corpus dans {os.path.abspath(args.out)}\nArchive : {os.path.abspath(archive)}")


if __name__ == "__main__":
    main()
