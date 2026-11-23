"""Étape 1 - Analyse des données.

Produit dans outputs/analyse/ :
  - resume.txt / resume.json : chiffres clés et contrôles de cohérence
  - 1_repartition_classes.png : nombre d'avions par classe (déséquilibre)
  - 2_avions_par_image.png    : combien d'avions par image
  - 3_taille_boites.png       : largeur x hauteur des boîtes
  - 4_taille_par_classe.png   : taille des avions selon la classe
  - 5_exemples_par_classe.png : mosaïque de vignettes, une ligne par classe
  - 6_image_annotee.png       : une image complète avec ses boîtes

Usage : python 01_analyse.py [--dataset "chemin/vers/student-dataset 2"]
"""
import argparse
import collections
import json
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from src.voc import DEFAULT_DATASET, class_sort_key, load_dataset

COLOR = "#2a6fdb"
INK = "#333333"
MUTED = "#888888"


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=12, color=INK)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#cccccc")
    ax.tick_params(colors=MUTED)
    ax.grid(axis="y", color="#eeeeee", linewidth=0.8)
    ax.set_axisbelow(True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--out", default="outputs/analyse")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    anns, orphans = load_dataset(args.dataset)
    boxes = [(a, b) for a in anns for b in a.boxes]
    classes = sorted({b.label for _, b in boxes}, key=class_sort_key)
    counts = collections.Counter(b.label for _, b in boxes)

    # ---------- Contrôles de cohérence ----------
    img_files = {os.path.splitext(f)[0] for f in os.listdir(os.path.join(args.dataset, "images")) if f.endswith(".jpg")}
    images_sans_annotation = sorted(img_files - {a.image_id for a in anns})
    taille_incoherente, boites_invalides = [], []
    for a in anns:
        real_w, real_h = Image.open(a.image_path).size
        if (real_w, real_h) != (a.width, a.height):
            taille_incoherente.append(a.image_id)
        for b in a.boxes:
            if b.w <= 0 or b.h <= 0 or b.xmin < 0 or b.ymin < 0 or b.xmax > a.width or b.ymax > a.height:
                boites_invalides.append((a.image_id, b.label))

    ws = np.array([b.w for _, b in boxes])
    hs = np.array([b.h for _, b in boxes])
    per_img = np.array([len(a.boxes) for a in anns])
    img_sizes = collections.Counter((a.width, a.height) for a in anns)

    resume = {
        "nb_images": len(anns),
        "nb_avions": len(boxes),
        "nb_classes": len(classes),
        "avions_par_classe": {c: counts[c] for c in classes},
        "ratio_desequilibre_max_min": round(max(counts.values()) / min(counts.values()), 2),
        "avions_par_image": {"min": int(per_img.min()), "moyenne": round(float(per_img.mean()), 2),
                              "mediane": float(np.median(per_img)), "max": int(per_img.max())},
        "tailles_images_top5": {f"{w}x{h}": n for (w, h), n in img_sizes.most_common(5)},
        "largeur_boites_px": {k: round(float(v), 1) for k, v in zip(["min", "p5", "mediane", "p95", "max"], np.percentile(ws, [0, 5, 50, 95, 100]))},
        "hauteur_boites_px": {k: round(float(v), 1) for k, v in zip(["min", "p5", "mediane", "p95", "max"], np.percentile(hs, [0, 5, 50, 95, 100]))},
        "boites_tronquees": sum(b.truncated for _, b in boxes),
        "controles": {
            "xml_sans_image": orphans,
            "images_sans_xml": images_sans_annotation,
            "taille_xml_differente_de_l_image": taille_incoherente,
            "boites_invalides": boites_invalides,
        },
    }
    with open(os.path.join(args.out, "resume.json"), "w", encoding="utf-8") as f:
        json.dump(resume, f, indent=2, ensure_ascii=False)

    lines = [
        f"Images annotées : {resume['nb_images']}",
        f"Avions annotés  : {resume['nb_avions']}  ({resume['nb_classes']} classes)",
        f"Avions/image    : {resume['avions_par_image']}",
        f"Déséquilibre    : la classe la plus fréquente a {resume['ratio_desequilibre_max_min']}x plus d'exemples que la plus rare",
        f"Taille images   : {resume['tailles_images_top5']}",
        f"Largeur boîtes  : {resume['largeur_boites_px']}",
        f"Hauteur boîtes  : {resume['hauteur_boites_px']}",
        f"Boîtes tronquées (bord d'image) : {resume['boites_tronquees']}",
        "",
        "Avions par classe :",
        *[f"  {c:>4} : {counts[c]:4d}" for c in classes],
        "",
        "Contrôles :",
        *[f"  {k} : {v if v else 'aucun'}" for k, v in resume["controles"].items()],
    ]
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(args.out, "resume.txt"), "w", encoding="utf-8") as f:
        f.write(txt)

    # ---------- 1. Répartition des classes ----------
    fig, ax = plt.subplots(figsize=(10, 4.5))
    vals = [counts[c] for c in classes]
    bars = ax.bar(classes, vals, color=COLOR, width=0.7)
    ax.axhline(np.mean(vals), color=MUTED, linestyle="--", linewidth=1)
    ax.text(len(classes) - 0.5, np.mean(vals), f" moyenne {np.mean(vals):.0f}", va="bottom", ha="right", color=MUTED, fontsize=9)
    for i in (int(np.argmin(vals)), int(np.argmax(vals))):
        ax.text(bars[i].get_x() + bars[i].get_width() / 2, vals[i], str(vals[i]), ha="center", va="bottom", fontsize=9, color=INK)
    style(ax, "Nombre d'avions annotés par classe", "Classe", "Nombre d'avions")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "1_repartition_classes.png"), dpi=150); plt.close(fig)

    # ---------- 2. Avions par image ----------
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(per_img, bins=range(1, per_img.max() + 2), color=COLOR, rwidth=0.85)
    style(ax, "Nombre d'avions par image", "Avions dans l'image", "Nombre d'images")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "2_avions_par_image.png"), dpi=150); plt.close(fig)

    # ---------- 3. Taille des boîtes ----------
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(ws, hs, s=8, alpha=0.25, color=COLOR, linewidths=0)
    lim = max(ws.max(), hs.max()) * 1.05
    ax.plot([0, lim], [0, lim], color=MUTED, linewidth=1, linestyle="--")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    style(ax, "Taille des boîtes (pixels)", "Largeur", "Hauteur")
    ax.grid(axis="x", color="#eeeeee", linewidth=0.8)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "3_taille_boites.png"), dpi=150); plt.close(fig)

    # ---------- 4. Taille par classe ----------
    fig, ax = plt.subplots(figsize=(10, 4.5))
    data = [[max(b.w, b.h) for _, b in boxes if b.label == c] for c in classes]
    ax.boxplot(data, tick_labels=classes, showfliers=False,
               medianprops=dict(color=COLOR, linewidth=2),
               boxprops=dict(color=MUTED), whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED))
    style(ax, "Taille des avions par classe (plus grand côté de la boîte)", "Classe", "Pixels")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "4_taille_par_classe.png"), dpi=150); plt.close(fig)

    # ---------- 5. Mosaïque d'exemples ----------
    rng = random.Random(0)
    n_ex, thumb = 8, 96
    by_class = collections.defaultdict(list)
    for a, b in boxes:
        by_class[b.label].append((a, b))
    mosaic = Image.new("RGB", (60 + n_ex * thumb, len(classes) * thumb), "white")
    draw = ImageDraw.Draw(mosaic)
    for r, c in enumerate(classes):
        draw.text((8, r * thumb + thumb // 2 - 6), c, fill=INK)
        for k, (a, b) in enumerate(rng.sample(by_class[c], min(n_ex, len(by_class[c])))):
            crop = Image.open(a.image_path).convert("RGB").crop((b.xmin, b.ymin, b.xmax, b.ymax))
            crop.thumbnail((thumb - 4, thumb - 4))
            mosaic.paste(crop, (60 + k * thumb + (thumb - crop.width) // 2, r * thumb + (thumb - crop.height) // 2))
    mosaic.save(os.path.join(args.out, "5_exemples_par_classe.png"))

    # ---------- 6. Une image annotée ----------
    a = max(anns, key=lambda x: len({b.label for b in x.boxes}))  # image la plus variée
    img = Image.open(a.image_path).convert("RGB")
    d = ImageDraw.Draw(img)
    for b in a.boxes:
        d.rectangle((b.xmin, b.ymin, b.xmax, b.ymax), outline=(255, 60, 60), width=2)
        d.text((b.xmin + 2, b.ymin + 2), b.label, fill=(255, 255, 0))
    img.save(os.path.join(args.out, f"6_image_annotee_{a.image_id}.png"))

    print(f"\nFigures enregistrées dans {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
