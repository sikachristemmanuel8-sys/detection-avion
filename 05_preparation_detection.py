"""Étape 4a - Préparation des données du détecteur « avion / fond ».

Idée de la détection par fenêtre glissante :
  on fait glisser un carré sur toute l'image, à plusieurs tailles, et un CNN
  répond pour chaque carré : « contient-il un avion bien cadré ? ».

Il faut donc apprendre au CNN deux classes à partir des annotations :
  - avion : carrés bien centrés sur un avion (IoU >= 0.5 avec la vraie boîte),
            avec un léger décalage / zoom aléatoire pour être robuste.
  - fond  : carrés qui ne recouvrent aucun avion à plus de 30 % (IoU < 0.3) :
              * pris au hasard dans l'image (piste, herbe, bâtiments...) ;
              * « négatifs difficiles » : carrés AUTOUR d'un avion mais mal
                cadrés (trop petit = un morceau d'avion, trop grand = l'avion
                + beaucoup de décor, ou décalé). Sans eux, le détecteur
                répondrait « avion » pour toutes les fenêtres proches d'un avion.

IoU (Intersection over Union) = aire commune / aire totale des deux boîtes.

On réutilise data/splits.json : mêmes images train / val / test que le classifieur.

Sortie (format ImageFolder, réutilisable par 03_classification.py) :
  data/fenetres/{train,val,test}/{avion,fond}/<image>_<k>.jpg
  data/images_detection/{val,test}/<id>.jpg (+ annotations) pour évaluer le détecteur

Usage : python 05_preparation_detection.py
"""
import argparse
import collections
import json
import math
import os
import random
import shutil

import numpy as np
from PIL import Image

from src.voc import DEFAULT_DATASET, load_dataset

SIZE = 96          # taille des fenêtres après redimensionnement (entrée du CNN)
IOU_POS = 0.5      # au-dessus : avion
IOU_NEG = 0.3      # en dessous : fond (entre les deux : ambigu, non utilisé)


def iou(win, boxes):
    """IoU entre une fenêtre (x0,y0,x1,y1) et un tableau de boîtes (N,4)."""
    if len(boxes) == 0:
        return np.zeros(0)
    ix0 = np.maximum(win[0], boxes[:, 0]); iy0 = np.maximum(win[1], boxes[:, 1])
    ix1 = np.minimum(win[2], boxes[:, 2]); iy1 = np.minimum(win[3], boxes[:, 3])
    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    a_win = (win[2] - win[0]) * (win[3] - win[1])
    a_box = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (a_win + a_box - inter)


def square(cx, cy, side, W, H):
    """Carré centré en (cx, cy), décalé si besoin pour rester dans l'image."""
    side = min(side, W, H)
    x0 = min(max(cx - side / 2, 0), W - side)
    y0 = min(max(cy - side / 2, 0), H - side)
    return np.array([x0, y0, x0 + side, y0 + side])


def save(img, win, path):
    img.crop(tuple(int(round(v)) for v in win)).resize((SIZE, SIZE), Image.BILINEAR).save(path, quality=92)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--splits", default="data/splits.json")
    p.add_argument("--pos-par-avion", type=int, default=2)
    p.add_argument("--neg-difficiles-par-avion", type=int, default=1)
    p.add_argument("--neg-aleatoires-par-image", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    anns, _ = load_dataset(args.dataset)
    by_id = {a.image_id: a for a in anns}
    with open(args.splits) as f:
        splits = json.load(f)

    out = os.path.join("data", "fenetres")
    if os.path.exists(out):
        shutil.rmtree(out)
    stats = collections.defaultdict(collections.Counter)

    for split, ids in splits.items():
        for c in ("avion", "fond"):
            os.makedirs(os.path.join(out, split, c))
        for n_img, image_id in enumerate(ids, 1):
            a = by_id[image_id]
            img = Image.open(a.image_path).convert("RGB")
            W, H = img.size
            gt = np.array([[b.xmin, b.ymin, b.xmax, b.ymax] for b in a.boxes])
            k = 0

            def emit(win, label, kind):
                nonlocal k
                save(img, win, os.path.join(out, split, label, f"{image_id}_{k}.jpg"))
                stats[split][kind] += 1
                k += 1

            for i, b in enumerate(a.boxes):
                cx, cy, side = (b.xmin + b.xmax) / 2, (b.ymin + b.ymax) / 2, max(b.w, b.h)
                # positifs : 1 carré exact + des carrés légèrement décalés / zoomés
                for j in range(args.pos_par_avion):
                    for _ in range(20):
                        if j == 0:
                            win = square(cx, cy, side, W, H)
                        else:
                            s = side * rng.uniform(0.9, 1.25)
                            win = square(cx + rng.uniform(-0.12, 0.12) * s, cy + rng.uniform(-0.12, 0.12) * s, s, W, H)
                        if iou(win, gt[i:i + 1])[0] >= IOU_POS:
                            emit(win, "avion", "avion")
                            break
                # négatifs difficiles : mal cadrés autour de l'avion
                for _ in range(args.neg_difficiles_par_avion):
                    for _ in range(30):
                        mode = rng.choice(("petit", "grand", "decale"))
                        if mode == "petit":
                            win = square(cx + rng.uniform(-0.3, 0.3) * side, cy + rng.uniform(-0.3, 0.3) * side,
                                         side * rng.uniform(0.35, 0.55), W, H)
                        elif mode == "grand":
                            win = square(cx, cy, side * rng.uniform(1.9, 2.8), W, H)
                        else:
                            ang = rng.uniform(0, 2 * math.pi)
                            d = side * rng.uniform(0.55, 0.9)
                            win = square(cx + d * math.cos(ang), cy + d * math.sin(ang), side, W, H)
                        if win[2] - win[0] >= 16 and iou(win, gt).max() < IOU_NEG:
                            emit(win, "fond", "fond_difficile")
                            break
            # négatifs aléatoires : tailles réparties de 32 à 450 px (échelle log)
            for _ in range(args.neg_aleatoires_par_image):
                for _ in range(30):
                    s = math.exp(rng.uniform(math.log(32), math.log(450)))
                    win = square(rng.uniform(0, W), rng.uniform(0, H), s, W, H)
                    if iou(win, gt).max() < IOU_NEG:
                        emit(win, "fond", "fond_aleatoire")
                        break
            if n_img % 200 == 0 or n_img == len(ids):
                print(f"  {split}: {n_img}/{len(ids)} images")

        # images complètes de val / test pour évaluer la détection
        if split in ("val", "test"):
            d = os.path.join("data", "images_detection", split)
            if os.path.exists(d):
                shutil.rmtree(d)
            os.makedirs(d)
            for image_id in ids:
                shutil.copy(by_id[image_id].image_path, d)
                shutil.copy(os.path.join(args.dataset, "annotations", image_id + ".xml"), d)

    print(f"\n{'split':>6} {'avion':>7} {'fond_difficile':>15} {'fond_aleatoire':>15}")
    for s in splits:
        print(f"{s:>6} {stats[s]['avion']:7d} {stats[s]['fond_difficile']:15d} {stats[s]['fond_aleatoire']:15d}")


if __name__ == "__main__":
    main()
