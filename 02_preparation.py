"""Étape 2 - Préparation des données : génération des vignettes (crops).

1. Découpage train / val / test PAR IMAGE (et pas par avion) :
   les avions d'une même image se ressemblent souvent (même aéroport, même
   éclairage, parfois même modèle garé en rang). Si on découpait par avion,
   des quasi-doublons se retrouveraient à la fois en train et en test, et le
   score de test serait trop optimiste (fuite de données).
   On utilise une stratification itérative pour que chaque classe soit
   répartie ~70/15/15 entre les splits (voir split_by_image).

2. Pour chaque avion : vignette carrée centrée sur la boîte, avec une marge
   de contexte (par défaut 15 %), ce qui garde la forme complète de l'avion
   même si la boîte est un peu serrée. Les zones hors image sont remplies en
   gris neutre. Les vignettes sont redimensionnées en --size x --size.

Sortie :
  data/crops/{train,val,test}/<classe>/<image>_<k>.jpg   (format ImageFolder)
  data/splits.json        : liste des images de chaque split
  data/crops/stats.txt    : nombre de vignettes par classe et par split

Usage : python 02_preparation.py [--dataset ...] [--size 224] [--margin 0.15]
"""
import argparse
import collections
import json
import os
import random
import shutil

import numpy as np
from PIL import Image

from src.voc import DEFAULT_DATASET, class_sort_key, load_dataset

SPLITS = ("train", "val", "test")


def split_by_image(anns, ratios, seed):
    """Stratification itérative (Sechidis et al., 2011), adaptée aux images
    qui contiennent plusieurs avions.

    Objectif : chaque split reçoit ~ratio x (nb d'avions de chaque classe).
    On traite d'abord les images contenant la classe la plus rare (c'est elle
    qui est la plus difficile à bien répartir) et on envoie chaque image dans
    le split qui a le plus grand « besoin » restant pour cette classe.
    """
    classes = sorted({b.label for a in anns for b in a.boxes}, key=class_sort_key)
    idx = {c: i for i, c in enumerate(classes)}
    per_img = np.zeros((len(anns), len(classes)))
    for i, a in enumerate(anns):
        for b in a.boxes:
            per_img[i, idx[b.label]] += 1

    ratios = np.array(ratios) / sum(ratios)
    need = np.outer(ratios, per_img.sum(0))          # [split, classe] avions encore attendus
    need_imgs = ratios * len(anns)                   # images encore attendues par split
    remaining = set(range(len(anns)))
    assign = {}
    rng = random.Random(seed)

    while remaining:
        left = per_img[list(remaining)].sum(0)
        left[left == 0] = np.inf
        c = int(np.argmin(left))                     # classe la plus rare parmi le reste
        imgs = [i for i in remaining if per_img[i, c] > 0]
        rng.shuffle(imgs)
        imgs.sort(key=lambda i: -per_img[i, c])      # images les plus chargées en premier
        for i in imgs:
            # split qui manque le plus de cette classe, puis d'images au total
            s = max(range(len(SPLITS)), key=lambda k: (need[k, c], need_imgs[k], rng.random()))
            assign[i] = s
            need[s] -= per_img[i]
            need_imgs[s] -= 1
            remaining.discard(i)

    return {s: [anns[i] for i in sorted(assign) if assign[i] == k] for k, s in enumerate(SPLITS)}


def square_crop(img, box, margin, size):
    cx, cy = (box.xmin + box.xmax) / 2, (box.ymin + box.ymax) / 2
    side = max(box.w, box.h) * (1 + 2 * margin)
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    side = int(round(side))
    canvas = Image.new("RGB", (side, side), (128, 128, 128))
    # partie de la vignette qui tombe dans l'image
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(img.width, x0 + side), min(img.height, y0 + side)
    canvas.paste(img.crop((sx0, sy0, sx1, sy1)), (sx0 - x0, sy0 - y0))
    return canvas.resize((size, size), Image.BICUBIC)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--out", default="data")
    p.add_argument("--size", type=int, default=224, help="taille des vignettes (224 = entrée standard des CNN ImageNet)")
    p.add_argument("--margin", type=float, default=0.15, help="marge de contexte autour de la boîte (fraction)")
    p.add_argument("--ratios", type=float, nargs=3, default=(0.7, 0.15, 0.15))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    anns, orphans = load_dataset(args.dataset)
    if orphans:
        print(f"Annotations ignorées (pas d'image) : {orphans}")

    splits = split_by_image(anns, args.ratios, args.seed)

    crops_dir = os.path.join(args.out, "crops")
    if os.path.exists(crops_dir):
        shutil.rmtree(crops_dir)
    os.makedirs(crops_dir)
    with open(os.path.join(args.out, "splits.json"), "w") as f:
        json.dump({s: sorted(a.image_id for a in v) for s, v in splits.items()}, f, indent=1)

    stats = {s: collections.Counter() for s in SPLITS}
    for s in SPLITS:
        for n, a in enumerate(splits[s], 1):
            img = Image.open(a.image_path).convert("RGB")
            for k, b in enumerate(a.boxes):
                d = os.path.join(crops_dir, s, b.label)
                os.makedirs(d, exist_ok=True)
                square_crop(img, b, args.margin, args.size).save(os.path.join(d, f"{a.image_id}_{k}.jpg"), quality=95)
                stats[s][b.label] += 1
            if n % 200 == 0 or n == len(splits[s]):
                print(f"  {s}: {n}/{len(splits[s])} images")

    classes = sorted(set().union(*stats.values()), key=class_sort_key)
    lines = [f"{'classe':>6} " + " ".join(f"{s:>7}" for s in SPLITS)]
    for c in classes:
        lines.append(f"{c:>6} " + " ".join(f"{stats[s][c]:7d}" for s in SPLITS))
    lines.append(f"{'TOTAL':>6} " + " ".join(f"{sum(stats[s].values()):7d}" for s in SPLITS))
    lines.append(f"{'images':>6} " + " ".join(f"{len(splits[s]):7d}" for s in SPLITS))
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(crops_dir, "stats.txt"), "w") as f:
        f.write(txt)


if __name__ == "__main__":
    main()
