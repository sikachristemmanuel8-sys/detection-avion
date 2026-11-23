"""Détection d'avions par fenêtre glissante + CNN « avion / fond » + NMS.

1. Fenêtres : on génère des carrés couvrant toute l'image, à plusieurs tailles
   (de 40 à ~420 px, car les avions vont de ~30 à ~550 px), avec un pas de
   20 % de la taille (les fenêtres se chevauchent pour ne pas rater d'avion).
2. Score : chaque fenêtre est découpée + redimensionnée en 96x96 (sur GPU, avec
   roi_align), puis le CNN donne la probabilité « avion ».
3. NMS (Non-Maximum Suppression) : plusieurs fenêtres voisines détectent le
   même avion. On garde la mieux notée et on supprime celles qui la
   recouvrent trop (IoU > seuil). On recommence avec la suivante, etc.
"""
import torch
from torchvision.ops import nms, roi_align

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
TAILLES = (40, 52, 68, 88, 114, 148, 192, 250, 325, 423)  # x1.3 à chaque fois


def generer_fenetres(W, H, tailles=TAILLES, pas=0.2):
    boxes = []
    for s in tailles:
        if s > min(W, H):
            continue
        step = max(4, int(s * pas))
        xs = list(range(0, W - s + 1, step))
        ys = list(range(0, H - s + 1, step))
        if xs[-1] != W - s: xs.append(W - s)    # couvrir jusqu'au bord
        if ys[-1] != H - s: ys.append(H - s)
        boxes += [(x, y, x + s, y + s) for y in ys for x in xs]
    return torch.tensor(boxes, dtype=torch.float32)


@torch.no_grad()
def scorer_fenetres(model, img, idx_avion, taille_entree=96, score_min=0.3,
                    tailles=TAILLES, pas=0.2, batch=1024, progres=None):
    """Étapes 1 et 2 : toutes les fenêtres + leur score « avion ».
    Retourne seulement celles dont le score >= score_min (sur CPU).
    progres : fonction optionnelle appelée avec la fraction déjà traitée (0 à 1)."""
    device = next(model.parameters()).device
    model.eval()
    img = img.to(device)
    _, H, W = img.shape
    fenetres = generer_fenetres(W, H, tailles, pas).to(device)
    mean, std = MEAN.to(device), STD.to(device)
    scores = []
    for chunk in fenetres.split(batch):
        rois = torch.cat([torch.zeros(len(chunk), 1, device=device), chunk], 1)
        # sampling_ratio=-1 : moyenne sur tous les pixels de chaque case -> réduction propre
        patches = roi_align(img[None], rois, output_size=taille_entree, spatial_scale=1.0,
                            sampling_ratio=-1, aligned=True)
        patches = ((patches - mean) / std).contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(patches)
        scores.append(torch.softmax(logits.float(), 1)[:, idx_avion])
        if progres is not None:
            progres(sum(len(s) for s in scores) / len(fenetres))
    scores = torch.cat(scores)
    garde = scores >= score_min
    return fenetres[garde].cpu(), scores[garde].cpu()


def supprimer_inclus(boites, scores, ratio=0.7):
    """Supprime une boîte si elle est en grande partie (> ratio de la plus petite
    des deux) recouverte par une boîte mieux notée. La NMS classique (IoU) ne le
    fait pas : une petite fenêtre posée sur l'aile d'un avion a un IoU faible
    avec la bonne fenêtre, mais elle est presque entièrement dedans."""
    if len(boites) == 0:
        return torch.zeros(0, dtype=torch.long)
    ordre = scores.argsort(descending=True)
    b = boites[ordre]
    aire = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = torch.max(b[:, None, :2], b[None, :, :2])
    rb = torch.min(b[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(2)
    couvert = inter / torch.min(aire[:, None], aire[None, :])
    garde = torch.ones(len(b), dtype=torch.bool)
    for i in range(len(b)):
        if garde[i]:
            sup = couvert[i] > ratio
            sup[: i + 1] = False
            garde &= ~sup
    return ordre[garde]


def fusionner(gardees, toutes, scores_toutes, iou=0.5):
    """Fusion des fenêtres : chaque boîte gardée est remplacée par la moyenne
    (pondérée par le score) de toutes les fenêtres qui la recouvrent avec
    IoU >= iou. Les fenêtres « votent » pour la position de l'avion : le cadre
    est mieux centré qu'avec une seule fenêtre de la grille."""
    if len(gardees) == 0 or len(toutes) == 0:
        return gardees
    from torchvision.ops import box_iou
    poids = (box_iou(gardees, toutes) >= iou).float() * scores_toutes[None]
    return (poids @ toutes) / poids.sum(1, keepdim=True).clamp(min=1e-6)


def post_traitement(boites, scores, seuil, iou_nms=0.3, inclusion=0.7, fusion=True):
    """Étape 3 : seuil de score, NMS, suppression des boîtes incluses
    (inclusion=None pour la désactiver), puis fusion des fenêtres."""
    toutes, scores_toutes = boites, scores
    garde = scores >= seuil
    boites, scores = boites[garde], scores[garde]
    garde = nms(boites, scores, iou_nms)
    boites, scores = boites[garde], scores[garde]
    if inclusion is not None:
        garde = supprimer_inclus(boites, scores, inclusion)
        boites, scores = boites[garde], scores[garde]
    if fusion:
        boites = fusionner(boites, toutes, scores_toutes)
    return boites, scores


def detecter(model, img, idx_avion, seuil=0.5, iou_nms=0.3, inclusion=0.7, fusion=True, **kw):
    """img : tenseur (3, H, W) dans [0, 1]. Retourne (boîtes (N,4), scores (N,))."""
    boites, scores = scorer_fenetres(model, img, idx_avion, score_min=min(seuil, 0.3), **kw)
    return post_traitement(boites, scores, seuil, iou_nms, inclusion, fusion)


def iou_matrice(a, b):
    """IoU entre chaque boîte de a (N,4) et de b (M,4) -> (N, M)."""
    from torchvision.ops import box_iou
    return box_iou(a, b)


def apparier(pred, scores, gt, seuil_iou=0.5):
    """Associe les prédictions aux vrais avions (une vraie boîte ne peut être
    trouvée qu'une fois). Retourne, avec les prédictions triées par score :
    (boîtes, scores, tp = vrai/faux positif pour chaque prédiction,
     trouves = vrai/faux pour chaque vrai avion)."""
    ordre = scores.argsort(descending=True)
    pred, scores = pred[ordre], scores[ordre]
    deja = torch.zeros(len(gt), dtype=torch.bool)
    tp = torch.zeros(len(pred), dtype=torch.bool)
    if len(gt) and len(pred):
        ious = iou_matrice(pred, gt)
        for i in range(len(pred)):
            ious_i = ious[i].clone()
            ious_i[deja] = -1
            j = int(ious_i.argmax())
            if ious_i[j] >= seuil_iou:
                tp[i] = True
                deja[j] = True
    return pred, scores, tp, deja
