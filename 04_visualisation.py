"""Étape 3 bis - Visualiser les vignettes classées par le modèle.

Produit dans outputs/visualisation/<arch>/ :
  - galerie_<split>.png      : une ligne par classe, N vignettes avec la prédiction
                               (cadre vert = correct, rouge = erreur)
  - erreurs_<split>.png      : toutes les vignettes mal classées (vraie -> prédite)
  - predictions_<split>.csv  : fichier, vraie classe, classe prédite, confiance

Usage : python 04_visualisation.py [--arch resnet50] [--split test] [--n 8]
"""
import argparse
import csv
import importlib
import math
import os
import random

import torch
from PIL import Image, ImageDraw, ImageFont

clf = importlib.import_module("03_classification")  # réutilise modèle + transformations

GREEN, RED, INK = (40, 160, 70), (220, 50, 50), (30, 30, 30)


def font(size):
    for name in ("DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def tile(path, true_c, pred_c, conf, thumb, f):
    """Vignette + cadre coloré + légende sous l'image."""
    ok = true_c == pred_c
    t = Image.new("RGB", (thumb, thumb + 22), "white")
    im = Image.open(path).convert("RGB").resize((thumb - 8, thumb - 8))
    t.paste(im, (4, 4))
    d = ImageDraw.Draw(t)
    d.rectangle((0, 0, thumb - 1, thumb - 1), outline=GREEN if ok else RED, width=4)
    txt = f"{pred_c} {conf:.0%}" if ok else f"{true_c}->{pred_c} {conf:.0%}"
    d.text((4, thumb + 3), txt, fill=GREEN if ok else RED, font=f)
    return t


def sheet(tiles, cols, title, f_title, row_labels=None, f_label=None):
    th_w, th_h = tiles[0][0].size if row_labels else tiles[0].size
    left = 60 if row_labels else 0
    if row_labels:
        rows = len(tiles)
    else:
        rows = math.ceil(len(tiles) / cols)
        tiles = [tiles[i * cols:(i + 1) * cols] for i in range(rows)]
    out = Image.new("RGB", (left + cols * (th_w + 6), 40 + rows * (th_h + 6)), "white")
    d = ImageDraw.Draw(out)
    d.text((10, 10), title, fill=INK, font=f_title)
    for r, row in enumerate(tiles):
        y = 40 + r * (th_h + 6)
        if row_labels:
            d.text((10, y + th_h // 2 - 10), row_labels[r], fill=INK, font=f_label)
        for c, t in enumerate(row):
            out.paste(t, (left + c * (th_w + 6), y))
    return out


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/crops")
    p.add_argument("--arch", default="resnet50")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--n", type=int, default=8, help="vignettes par classe dans la galerie")
    p.add_argument("--thumb", type=int, default=128)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(os.path.join("outputs", "classification", args.arch, "best_model.pt"), map_location=device)
    classes = ckpt["classes"]
    model = clf.build_model(ckpt["arch"], len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    _, eval_tf = clf.build_transforms(ckpt["size"])
    ds = clf.SortedImageFolder(os.path.join(args.data, args.split), eval_tf)
    assert ds.classes == classes
    loader = torch.utils.data.DataLoader(ds, batch_size=64, num_workers=2)

    probs = torch.cat([torch.softmax(model(x.to(device)), 1).cpu() for x, _ in loader])
    conf, pred = probs.max(1)
    rows = [(path, classes[y], classes[p], float(c))
            for (path, y), p, c in zip(ds.samples, pred.tolist(), conf.tolist())]

    out_dir = os.path.join("outputs", "visualisation", args.arch)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"predictions_{args.split}.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["fichier", "vraie_classe", "classe_predite", "confiance"])
        for path, t, pc, c in rows:
            w.writerow([os.path.relpath(path, args.data), t, pc, f"{c:.4f}"])

    f_small, f_title, f_label = font(12), font(20), font(16)
    n_err = sum(t != pc for _, t, pc, _ in rows)
    acc = 1 - n_err / len(rows)

    # galerie : N vignettes au hasard par classe
    rng = random.Random(0)
    gallery = []
    for c in classes:
        items = [r for r in rows if r[1] == c]
        gallery.append([tile(*r, args.thumb, f_small) for r in rng.sample(items, min(args.n, len(items)))])
    sheet(gallery, args.n, f"Vignettes {args.split} classées par {args.arch} - précision {acc:.1%}",
          f_title, row_labels=classes, f_label=f_label).save(os.path.join(out_dir, f"galerie_{args.split}.png"))

    # planche des erreurs, des plus « sûres d'elles » aux moins sûres
    errors = sorted((r for r in rows if r[1] != r[2]), key=lambda r: -r[3])
    if errors:
        sheet([tile(*r, args.thumb, f_small) for r in errors], 8,
              f"{len(errors)} erreurs sur {len(rows)} vignettes ({args.split}) - vraie -> prédite",
              f_title).save(os.path.join(out_dir, f"erreurs_{args.split}.png"))

    print(f"{len(rows)} vignettes, {n_err} erreurs, précision {acc:.2%}")
    print(f"Images enregistrées dans {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
