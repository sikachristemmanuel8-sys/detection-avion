"""Chaîne complète : image satellite -> avions détectés + classe (A1..A20).

1. Détection : fenêtre glissante + CNN « avion / fond » (src/detection.py),
   avec les réglages choisis sur la validation par 06_detection.py.
2. Classification : chaque avion détecté est découpé exactement comme les
   vignettes d'entraînement (carré, marge 15 %, fond gris, 224 px) et passé
   au classifieur 20 classes.

Utilisé par 08_application.py (interface graphique) et 09_corpus_annote.py.
"""
import json
import os

import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torchvision import models, transforms
from torchvision.transforms.functional import to_tensor

from .detection import post_traitement, scorer_fenetres

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
MARGE = 0.15   # même marge que 02_preparation.py

# 20 couleurs bien distinctes (une par classe)
COULEURS = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6",
            "#bfef45", "#fabed4", "#469990", "#dcbeff", "#9a6324", "#fffac8", "#800000", "#aaffc3",
            "#808000", "#ffd8b1", "#000075", "#a9a9a9"]


def construire(arch, n_classes):
    """Même architecture que 03_classification.py, sans télécharger les poids
    ImageNet (on charge juste après nos propres poids entraînés)."""
    m = getattr(models, arch)(weights=None)
    if arch.startswith("resnet"):
        m.fc = nn.Linear(m.fc.in_features, n_classes)
    elif arch == "efficientnet_b0":
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_classes)
    elif arch == "convnext_tiny":
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, n_classes)
    return m


def charger(path, device):
    ckpt = torch.load(path, map_location="cpu")
    m = construire(ckpt["arch"], len(ckpt["classes"]))
    m.load_state_dict(ckpt["state_dict"])
    return m.to(device).eval(), ckpt


def vignette(img, boite, marge=MARGE, taille=224):
    """Découpe carrée autour d'une boîte (x0, y0, x1, y1), fond gris si on
    déborde de l'image - identique à square_crop() de 02_preparation.py."""
    x0, y0, x1, y1 = boite
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    cote = max(x1 - x0, y1 - y0) * (1 + 2 * marge)
    gx, gy = int(round(cx - cote / 2)), int(round(cy - cote / 2))
    cote = max(1, int(round(cote)))
    fond = Image.new("RGB", (cote, cote), (128, 128, 128))
    sx0, sy0 = max(0, gx), max(0, gy)
    sx1, sy1 = min(img.width, gx + cote), min(img.height, gy + cote)
    if sx1 > sx0 and sy1 > sy0:
        fond.paste(img.crop((sx0, sy0, sx1, sy1)), (sx0 - gx, sy0 - gy))
    return fond.resize((taille, taille), Image.BICUBIC)


class Pipeline:
    def __init__(self, racine=".", detecteur="detecteur_resnet50", classifieur="resnet50", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        cls_dir = os.path.join(racine, "outputs", "classification")

        self.det, ck = charger(os.path.join(cls_dir, detecteur, "best_model.pt"), self.device)
        self.idx_avion = ck["classes"].index("avion")
        self.taille_det = ck["size"]
        # réglages du post-traitement choisis sur la validation par 06_detection.py
        self.seuil, self.iou_nms, self.inclusion, self.fusion = 0.95, 0.3, 0.7, True
        reglages = os.path.join(racine, "outputs", "detection", detecteur, "metrics.json")
        if os.path.exists(reglages):
            m = json.load(open(reglages))
            self.seuil = m.get("seuil", self.seuil)
            self.iou_nms = m.get("iou_nms", self.iou_nms)
            self.inclusion = m.get("inclusion", self.inclusion)
            self.fusion = m.get("fusion", self.fusion)

        self.clf, ck = charger(os.path.join(cls_dir, classifieur, "best_model.pt"), self.device)
        self.classes = ck["classes"]
        self.taille_clf = ck["size"]
        self.tf = transforms.Compose([transforms.Resize((self.taille_clf, self.taille_clf)),
                                      transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
        self.noms = (detecteur, classifieur)

    @torch.no_grad()
    def scores_fenetres(self, img, pas=0.2, progres=None):
        """Partie lente (le CNN note toutes les fenêtres). Le résultat peut être
        gardé en mémoire : changer ensuite le seuil ne demande que finaliser()."""
        return scorer_fenetres(self.det, to_tensor(img.convert("RGB")), self.idx_avion,
                               taille_entree=self.taille_det, score_min=0.3, pas=pas, progres=progres)

    def analyser(self, img, seuil=None, pas=0.2, progres=None):
        """img : image PIL. Retourne une liste de dict
        {boite: [x0,y0,x1,y1], score: proba « avion », classe: 'A7', proba: proba de la classe}."""
        boites, scores = self.scores_fenetres(img, pas, progres)
        return self.finaliser(img, boites, scores, seuil)

    @torch.no_grad()
    def finaliser(self, img, boites, scores, seuil=None):
        """Partie rapide : seuil + NMS + suppression des incluses + fusion, puis classification."""
        img = img.convert("RGB")
        seuil = self.seuil if seuil is None else seuil
        boites, scores = post_traitement(boites, scores, max(seuil, 0.3), self.iou_nms, self.inclusion, self.fusion)
        if len(boites) == 0:
            return []
        boites[:, 0::2] = boites[:, 0::2].clamp(0, img.width)
        boites[:, 1::2] = boites[:, 1::2].clamp(0, img.height)
        lots = torch.stack([self.tf(vignette(img, b, taille=self.taille_clf)) for b in boites.tolist()])
        probas = []
        for lot in lots.split(64):
            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.device == "cuda"):
                probas.append(torch.softmax(self.clf(lot.to(self.device)).float(), 1).cpu())
        p, k = torch.cat(probas).max(1)
        res = [{"boite": [round(v) for v in b], "score": float(s), "classe": self.classes[int(c)], "proba": float(q)}
               for b, s, c, q in zip(boites.tolist(), scores.tolist(), k, p)]
        return sorted(res, key=lambda d: (d["boite"][1], d["boite"][0]))   # de haut en bas


def police(taille):
    try:
        return ImageFont.load_default(size=taille)
    except TypeError:   # anciennes versions de Pillow
        return ImageFont.load_default()


def couleur(classe):
    chiffres = "".join(c for c in classe if c.isdigit())
    return COULEURS[(int(chiffres) - 1) % len(COULEURS)] if chiffres else "#ffffff"


def dessiner(img, detections, epaisseur=3):
    """Dessine chaque avion avec sa classe (une couleur par classe)."""
    img = img.convert("RGB").copy()
    d = ImageDraw.Draw(img)
    f = police(max(12, img.width // 60))
    for det in detections:
        x0, y0, x1, y1 = det["boite"]
        c = couleur(det["classe"])
        d.rectangle((x0, y0, x1, y1), outline=c, width=epaisseur)
        txt = f"{det['classe']} {det['proba']:.0%}"
        l, t, r, b = d.textbbox((x0, y0), txt, font=f)
        ty = y0 - (b - t) - 4 if y0 - (b - t) - 4 >= 0 else y0
        d.rectangle((x0, ty, x0 + (r - l) + 6, ty + (b - t) + 4), fill=c)
        r_, g_, b_ = (int(c[i:i + 2], 16) for i in (1, 3, 5))
        clair = 0.299 * r_ + 0.587 * g_ + 0.114 * b_ > 140   # texte noir sur couleur claire, blanc sinon
        d.text((x0 + 3, ty + 1), txt, fill="black" if clair else "white", font=f)
    return img
