"""Étape 4c - Comparaison ResNet18 / ResNet50 (classification et détection).

ResNet18 et ResNet50 sont deux CNN de la même famille (réseaux résiduels) :
  - ResNet18 : 18 couches, ~11 millions de paramètres -> plus rapide, plus léger
  - ResNet50 : 50 couches, ~24 millions de paramètres -> plus de capacité, plus lent
La question : le réseau plus gros vaut-il le surcoût en temps de calcul ?

Lit les metrics.json produits par 03_classification.py et 06_detection.py :
  outputs/classification/resnet18, resnet50                     (20 classes)
  outputs/classification/detecteur_resnet18, detecteur_resnet50 (avion / fond)
  outputs/detection/detecteur_resnet18, detecteur_resnet50      (images complètes)

Sortie : outputs/comparaison/comparaison.txt et comparaison.png

Usage : python 07_comparaison.py
"""
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARCHS = ("resnet18", "resnet50")
COULEURS = {"resnet18": "#2a6fdb", "resnet50": "#e8792b"}
PARAMS_CONNUS = {"resnet18": 11.2, "resnet50": 23.5}  # si metrics.json absent (ancien entraînement)


def lire(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def lire_classif(arch):
    d = os.path.join("outputs", "classification", arch)
    m = lire(os.path.join(d, "metrics.json"))
    if m is None and os.path.exists(os.path.join(d, "rapport_test.txt")):
        # modèle entraîné avant l'ajout de metrics.json : on relit le rapport texte
        txt = open(os.path.join(d, "rapport_test.txt"), encoding="utf-8").read()
        r = re.search(r"accuracy ([\d.]+)\s+F1-macro ([\d.]+)", txt)
        if r:
            m = {"test_acc": float(r.group(1)), "test_f1_macro": float(r.group(2)),
                 "parametres_millions": PARAMS_CONNUS[arch], "duree_entrainement_min": None}
    return m


def fmt(v, pct=False, suffixe=""):
    if v is None:
        return "n/a"
    return f"{v:.1%}" if pct else f"{v}{suffixe}"


def main():
    out = os.path.join("outputs", "comparaison")
    os.makedirs(out, exist_ok=True)
    classif = {a: lire_classif(a) for a in ARCHS}
    det_cnn = {a: lire(os.path.join("outputs", "classification", f"detecteur_{a}", "metrics.json")) for a in ARCHS}
    det = {a: lire(os.path.join("outputs", "detection", f"detecteur_{a}", "metrics.json")) for a in ARCHS}

    g = lambda d, k: d.get(k) if d else None
    lignes = [
        ("CLASSIFICATION (20 modèles d'avions, test)", None, None),
        ("  Accuracy", *(fmt(g(classif[a], "test_acc"), True) for a in ARCHS)),
        ("  F1-macro", *(fmt(g(classif[a], "test_f1_macro"), True) for a in ARCHS)),
        ("  Paramètres", *(fmt(g(classif[a], "parametres_millions"), suffixe=" M") for a in ARCHS)),
        ("  Durée d'entraînement", *(fmt(g(classif[a], "duree_entrainement_min"), suffixe=" min") for a in ARCHS)),
        ("", None, None),
        ("DÉTECTION (fenêtre glissante, images de test, IoU >= 0.5)", None, None),
        ("  Précision", *(fmt(g(det[a], "precision"), True) for a in ARCHS)),
        ("  Rappel", *(fmt(g(det[a], "rappel"), True) for a in ARCHS)),
        ("  F1", *(fmt(g(det[a], "f1"), True) for a in ARCHS)),
        ("  Fausses alarmes", *(fmt(g(det[a], "fausses_alarmes")) for a in ARCHS)),
        ("  Avions manqués", *(fmt(g(det[a], "avions_manques")) for a in ARCHS)),
        ("  Temps par image", *(fmt(g(det[a], "secondes_par_image"), suffixe=" s") for a in ARCHS)),
        ("  Durée d'entraînement", *(fmt(g(det_cnn[a], "duree_entrainement_min"), suffixe=" min") for a in ARCHS)),
    ]
    txt = f"{'':48} {'ResNet18':>10} {'ResNet50':>10}\n"
    for nom, v18, v50 in lignes:
        txt += f"{nom:48} {v18:>10} {v50:>10}\n" if v18 is not None else f"{nom}\n"
    print(txt)
    with open(os.path.join(out, "comparaison.txt"), "w", encoding="utf-8") as f:
        f.write(txt)

    # ---- graphique : barres groupées, axe de 0 à 1 (on ne tronque pas l'axe)
    panneaux = [
        ("Classification (20 classes)", [("Accuracy", classif, "test_acc"), ("F1-macro", classif, "test_f1_macro")]),
        ("Détection (images complètes)", [("Précision", det, "precision"), ("Rappel", det, "rappel"), ("F1", det, "f1")]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [2, 3]})
    largeur = 0.36
    for ax, (titre, mesures) in zip(axes, panneaux):
        for k, a in enumerate(ARCHS):
            xs, vals = [], []
            for i, (_, src, cle) in enumerate(mesures):
                v = g(src[a], cle)
                if v is not None:
                    xs.append(i + (k - 0.5) * largeur); vals.append(v)
            barres = ax.bar(xs, vals, width=largeur - 0.03, color=COULEURS[a], label=a.replace("resnet", "ResNet"))
            for b, v in zip(barres, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.1%}", ha="center", va="bottom", fontsize=9, color="#333333")
        ax.set_xticks(range(len(mesures)), [m[0] for m in mesures])
        ax.set_ylim(0, 1.1); ax.set_title(titre, loc="left")
        ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#eeeeee"); ax.set_axisbelow(True)
    fig.suptitle("ResNet18 vs ResNet50", x=0.01, ha="left", fontsize=13)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=2, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(out, "comparaison.png"), dpi=150); plt.close(fig)
    print(f"Résultats dans {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
