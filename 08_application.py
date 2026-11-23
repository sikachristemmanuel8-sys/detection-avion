"""Étape 5 - Application avec interface graphique (Streamlit).

On dépose une image satellite -> l'application dessine chaque avion détecté
avec sa classe (A1..A20) et sa confiance, et affiche le tableau des avions.

Lancement :
  streamlit run 08_application.py        # puis ouvrir http://localhost:8501
"""
import glob
import hashlib
import json
import os
from collections import Counter

import pandas as pd
import streamlit as st
from PIL import Image

from src.pipeline import Pipeline, couleur, dessiner

DOSSIER_EXEMPLES = os.path.join("data", "images_detection", "test")


def afficher(fonction, *args, **kw):
    """Affichage sur toute la largeur, quelle que soit la version de Streamlit."""
    try:
        return fonction(*args, width="stretch", **kw)               # versions récentes
    except Exception:
        return fonction(*args, use_container_width=True, **kw)      # anciennes versions


def detecteurs_disponibles():
    """Détecteurs entraînés, le meilleur F1 (test) en premier."""
    res = []
    for nom in ("detecteur_resnet50", "detecteur_resnet18"):
        if os.path.exists(os.path.join("outputs", "classification", nom, "best_model.pt")):
            m = os.path.join("outputs", "detection", nom, "metrics.json")
            f1 = json.load(open(m)).get("f1", 0) if os.path.exists(m) else 0
            res.append((f1, nom))
    return [nom for _, nom in sorted(res, reverse=True)]


@st.cache_resource(show_spinner="Chargement des modèles...")
def charger_pipeline(detecteur, classifieur):
    # chargé une seule fois, puis gardé en mémoire entre les clics
    return Pipeline(detecteur=detecteur, classifieur=classifieur)


@st.cache_resource
def memoire():
    """Dictionnaire partagé : (détecteur, pas, empreinte de l'image) -> scores des fenêtres."""
    return {}


def scores_en_memoire(pipe, detecteur, pas, image, progres):
    """Partie lente (le CNN note toutes les fenêtres), gardée en mémoire :
    revenir sur une image ou bouger le curseur du seuil est ensuite instantané."""
    m = memoire()
    cle = (detecteur, pas, hashlib.md5(image.tobytes()).hexdigest())
    if cle not in m:
        if len(m) >= 50:                      # on garde au plus 50 images
            m.pop(next(iter(m)))
        m[cle] = pipe.scores_fenetres(image, pas, progres)
    return m[cle]


st.set_page_config(page_title="Détection d'avions", page_icon="✈️", layout="wide")
st.title("✈️ Détection et classification d'avions sur images satellites")

# ---------------------------------------------------------------- barre latérale : réglages
with st.sidebar:
    st.header("Réglages")
    detecteurs = detecteurs_disponibles()
    if not detecteurs:
        st.error("Aucun détecteur trouvé dans outputs/classification/")
        st.stop()
    detecteur = st.selectbox("Détecteur (avion / fond)", detecteurs,
                             format_func=lambda n: n.replace("detecteur_", "").replace("resnet", "ResNet"))
    pipe = charger_pipeline(detecteur, "resnet50")
    seuil = st.slider("Seuil de confiance du détecteur", 0.30, 0.999, float(pipe.seuil), 0.005,
                      help="Plus haut = moins de fausses alarmes, mais plus d'avions manqués. "
                           "La valeur de départ est celle choisie sur la validation.")
    rapide = st.toggle("Analyse rapide (≈ 2x plus vite)", value=False,
                       help="Fenêtres décalées de 30 % de leur taille au lieu de 20 % : deux fois moins de "
                            "fenêtres à analyser, mais quelques petits avions peuvent être manqués. "
                            "Les scores du rapport ont été mesurés en mode normal.")
    st.caption(f"Classifieur 20 classes : ResNet50 · calcul sur **{pipe.device.upper()}**")

# ---------------------------------------------------------------- choix de l'image
col_choix1, col_choix2 = st.columns(2)
with col_choix1:
    fichier = st.file_uploader("Dépose une image satellite", type=["jpg", "jpeg", "png"])
with col_choix2:
    exemples = sorted(glob.glob(os.path.join(DOSSIER_EXEMPLES, "*.jpg")))
    exemple = st.selectbox("... ou choisis une image de test", ["(aucune)"] + [os.path.basename(e) for e in exemples])

if fichier is not None:
    image, nom = Image.open(fichier).convert("RGB"), fichier.name
elif exemple != "(aucune)":
    image, nom = Image.open(os.path.join(DOSSIER_EXEMPLES, exemple)).convert("RGB"), exemple
else:
    st.info("Dépose une image ou choisis une image de test pour commencer.")
    st.stop()

# ---------------------------------------------------------------- analyse
pas = 0.3 if rapide else 0.2
barre = st.progress(0.0, text="Recherche des avions... (seulement la 1re fois pour chaque image)")
boites, scores = scores_en_memoire(pipe, detecteur, pas, image,
                                   lambda f: barre.progress(min(f, 1.0), text=f"Analyse des fenêtres : {f:.0%}"))
barre.empty()
with st.spinner("Classification des avions..."):
    detections = pipe.finaliser(image, boites, scores, seuil)

col_img1, col_img2 = st.columns(2)
afficher(col_img1.image, image, caption=f"Image d'origine : {nom}")
afficher(col_img2.image, dessiner(image, detections), caption="Avions détectés et classés")

if not detections:
    st.warning("Aucun avion détecté. Essaie de baisser le seuil dans la barre de gauche.")
    st.stop()

compte = Counter(d["classe"] for d in detections)
st.subheader(f"{len(detections)} avion(s) détecté(s)")
cols = st.columns(min(len(compte), 8))
for i, (classe, n) in enumerate(sorted(compte.items(), key=lambda kv: int(kv[0][1:]))):
    cols[i % len(cols)].markdown(
        f"<span style='display:inline-block;width:12px;height:12px;background:{couleur(classe)};"
        f"border-radius:2px'></span> **{classe}** × {n}", unsafe_allow_html=True)

tableau = pd.DataFrame([{"classe": d["classe"], "confiance classe": round(d["proba"], 3),
                         "score avion": round(d["score"], 3),
                         "x0": d["boite"][0], "y0": d["boite"][1], "x1": d["boite"][2], "y1": d["boite"][3]}
                        for d in detections])
tableau.index += 1
# tableau en HTML (st.dataframe a besoin de pyarrow, qui n'existe pas sur certains PC, ex. Windows ARM)
st.markdown(tableau.to_html(), unsafe_allow_html=True)
st.download_button("Télécharger le résultat (CSV)", tableau.to_csv().encode("utf-8"),
                   file_name=os.path.splitext(nom)[0] + "_avions.csv", mime="text/csv")
