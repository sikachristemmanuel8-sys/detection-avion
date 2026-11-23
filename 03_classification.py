"""Étape 3 - Classification des vignettes (20 modèles d'avions).

Choix techniques (à savoir justifier à l'oral) :
  - Transfer Learning : CNN pré-entraîné sur ImageNet (autorisé par le sujet),
    puis ré-entraîné (fine-tuning) sur nos vignettes. Avec ~5 500 vignettes,
    entraîner un CNN from scratch sur-apprendrait vite ; les filtres
    pré-appris (bords, textures, formes) sont réutilisables.
  - Data Augmentation : vues satellites = vues de dessus, l'orientation d'un
    avion n'a pas de sens -> rotations à 360° + miroirs. Plus variations de
    luminosité/contraste et un léger flou (certaines images sont floues).
  - Déséquilibre (×7 entre A6 et A16) : la perte (cross-entropy) est pondérée
    par classe (poids ∝ 1/√fréquence) pour que les classes rares comptent.
  - Sélection du modèle sur le F1-macro de validation (moyenne des F1 par
    classe : une classe rare pèse autant qu'une fréquente), arrêt anticipé
    si plus d'amélioration (limite le sur-apprentissage).
  - Évaluation finale UNE SEULE FOIS sur le test : matrice de confusion,
    précision, rappel, F1 par classe.

Sortie dans outputs/classification/<arch>/ :
  best_model.pt, historique.json, courbes.png, matrice_confusion.png, rapport_test.txt

Usage :
  python 03_classification.py                      # resnet50, 25 époques
  python 03_classification.py --arch efficientnet_b0 --epochs 30
"""
import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # normalisation ImageNet


# ---------------------------------------------------------------- données
def build_transforms(size):
    train_tf = transforms.Compose([
        transforms.RandomRotation(180, fill=(128, 128, 128)),          # toutes orientations
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomResizedCrop(size, scale=(0.75, 1.0), ratio=(0.9, 1.1)),  # léger zoom / décalage
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02),
        transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_tf, eval_tf


def class_sort_key(label):
    digits = "".join(c for c in label if c.isdigit())
    return (int(digits) if digits else 10**9, label)


class SortedImageFolder(datasets.ImageFolder):
    """ImageFolder trie A1, A10, A11...; on veut A1, A2, ..., A20."""
    def find_classes(self, directory):
        classes = sorted((d.name for d in os.scandir(directory) if d.is_dir()), key=class_sort_key)
        return classes, {c: i for i, c in enumerate(classes)}


# ---------------------------------------------------------------- modèle
def build_model(arch, n_classes):
    if arch == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
    elif arch == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_classes)
    elif arch == "convnext_tiny":
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, n_classes)
    else:
        raise ValueError(arch)
    return m


def head_parameters(model, arch):
    if arch.startswith("resnet"):
        return list(model.fc.parameters())
    return list(model.classifier.parameters())


# ---------------------------------------------------------------- métriques
def confusion_matrix(y_true, y_pred, n):
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm  # lignes = vraie classe, colonnes = classe prédite


def per_class_metrics(cm):
    tp = np.diag(cm).astype(float)
    precision = np.divide(tp, cm.sum(0), out=np.zeros_like(tp), where=cm.sum(0) > 0)  # parmi les prédits X, combien sont X
    recall = np.divide(tp, cm.sum(1), out=np.zeros_like(tp), where=cm.sum(1) > 0)     # parmi les vrais X, combien trouvés
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    return precision, recall, f1


@torch.no_grad()
def evaluate(model, loader, device, n_classes, criterion=None):
    model.eval()
    preds, targets, loss_sum = [], [], 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        if criterion is not None:
            loss_sum += criterion(out, y).item() * len(y)
        preds.append(out.argmax(1).cpu())
        targets.append(y.cpu())
    preds, targets = torch.cat(preds).numpy(), torch.cat(targets).numpy()
    cm = confusion_matrix(targets, preds, n_classes)
    _, _, f1 = per_class_metrics(cm)
    return {"loss": loss_sum / len(targets), "acc": float((preds == targets).mean()),
            "f1_macro": float(f1.mean()), "cm": cm}


# ---------------------------------------------------------------- figures
def plot_history(hist, path):
    ep = range(1, len(hist["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, title in ((axes[0], "loss", "Perte (loss)"), (axes[1], "f1", "F1-macro")):
        ax.plot(ep, hist[f"train_{key}"], color="#2a6fdb", linewidth=2, label="entraînement")
        ax.plot(ep, hist[f"val_{key}"], color="#e8792b", linewidth=2, label="validation")
        ax.set_title(title, loc="left"); ax.set_xlabel("Époque")
        ax.spines[["top", "right"]].set_visible(False); ax.grid(color="#eeeeee"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_confusion(cm, classes, path):
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)  # normalisée par ligne = rappel sur la diagonale
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=90); ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Classe prédite"); ax.set_ylabel("Vraie classe")
    ax.set_title("Matrice de confusion (test, normalisée par ligne)", loc="left")
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.5 else "#333333")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/crops")
    p.add_argument("--arch", default="resnet50", choices=["resnet18", "resnet50", "efficientnet_b0", "convnext_tiny"])
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4, help="LR de la tête ; le backbone pré-entraîné utilise lr/10")
    p.add_argument("--patience", type=int, default=6, help="arrêt anticipé")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--name", default=None, help="nom du dossier de sortie (par défaut = arch)")
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    out_dir = os.path.join("outputs", "classification", args.name or args.arch)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Device : {device}")

    train_tf, eval_tf = build_transforms(args.size)
    ds = {s: SortedImageFolder(os.path.join(args.data, s), tf)
          for s, tf in (("train", train_tf), ("val", eval_tf), ("test", eval_tf))}
    classes = ds["train"].classes
    n = len(classes)
    assert ds["val"].classes == classes == ds["test"].classes
    loaders = {s: DataLoader(d, batch_size=args.batch, shuffle=(s == "train"), num_workers=args.workers,
                             pin_memory=(device == "cuda"), persistent_workers=args.workers > 0)
               for s, d in ds.items()}
    print({s: len(d) for s, d in ds.items()}, "vignettes")

    # pondération des classes : poids ∝ 1/sqrt(fréquence), moyenne = 1
    counts = np.bincount(ds["train"].targets, minlength=n)
    w = 1 / np.sqrt(counts); w = w / w.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device), label_smoothing=0.1)

    model = build_model(args.arch, n).to(device)
    head = head_parameters(model, args.arch)
    head_ids = {id(q) for q in head}
    backbone = [q for q in model.parameters() if id(q) not in head_ids]
    optimizer = torch.optim.AdamW([{"params": backbone, "lr": args.lr / 10},
                                   {"params": head, "lr": args.lr}], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # précision mixte (float16) sur GPU : ~2x plus rapide, même précision finale
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    hist = {k: [] for k in ("train_loss", "train_f1", "val_loss", "val_f1", "val_acc")}
    best_f1, best_epoch = -1.0, 0
    t_debut = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        loss_sum, preds, targets = 0.0, [], []
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()  # rétropropagation : calcul des gradients
            scaler.step(optimizer)         # descente de gradient : mise à jour des poids
            scaler.update()
            loss_sum += loss.item() * len(y)
            preds.append(out.argmax(1).cpu()); targets.append(y.cpu())
        scheduler.step()
        preds, targets = torch.cat(preds).numpy(), torch.cat(targets).numpy()
        train_f1 = float(per_class_metrics(confusion_matrix(targets, preds, n))[2].mean())

        val = evaluate(model, loaders["val"], device, n, criterion)
        hist["train_loss"].append(loss_sum / len(targets)); hist["train_f1"].append(train_f1)
        hist["val_loss"].append(val["loss"]); hist["val_f1"].append(val["f1_macro"]); hist["val_acc"].append(val["acc"])
        flag = ""
        if val["f1_macro"] > best_f1:
            best_f1, best_epoch = val["f1_macro"], epoch
            torch.save({"arch": args.arch, "classes": classes, "size": args.size,
                        "state_dict": model.state_dict()}, os.path.join(out_dir, "best_model.pt"))
            flag = "  <- meilleur"
        print(f"[{epoch:2d}/{args.epochs}] train loss {hist['train_loss'][-1]:.3f} F1 {train_f1:.3f} | "
              f"val loss {val['loss']:.3f} acc {val['acc']:.3f} F1 {val['f1_macro']:.3f} | {time.time() - t0:.0f}s{flag}")
        plot_history(hist, os.path.join(out_dir, "courbes.png"))
        with open(os.path.join(out_dir, "historique.json"), "w") as f:
            json.dump(hist, f, indent=1)
        if epoch - best_epoch >= args.patience:
            print(f"Arrêt anticipé : pas d'amélioration depuis {args.patience} époques.")
            break
    duree_min = (time.time() - t_debut) / 60

    # ---- évaluation finale sur le test avec le meilleur modèle
    ckpt = torch.load(os.path.join(out_dir, "best_model.pt"), map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    test = evaluate(model, loaders["test"], device, n)
    precision, recall, f1 = per_class_metrics(test["cm"])
    support = test["cm"].sum(1)
    lines = [f"Modèle : {args.arch} (meilleure époque {best_epoch}, F1-macro val {best_f1:.3f})",
             f"TEST  accuracy {test['acc']:.3f}   F1-macro {test['f1_macro']:.3f}", "",
             f"{'classe':>6} {'précision':>10} {'rappel':>8} {'F1':>6} {'nb':>5}"]
    for i, c in enumerate(classes):
        lines.append(f"{c:>6} {precision[i]:10.3f} {recall[i]:8.3f} {f1[i]:6.3f} {support[i]:5d}")
    # confusions les plus fréquentes : utile pour l'analyse d'erreurs à l'oral
    off = test["cm"].copy(); np.fill_diagonal(off, 0)
    lines += ["", "Confusions les plus fréquentes (vraie -> prédite : nb) :"]
    for k in np.argsort(off, axis=None)[::-1][:8]:
        i, j = divmod(int(k), n)
        if off[i, j]:
            lines.append(f"  {classes[i]} -> {classes[j]} : {off[i, j]}")
    txt = "\n".join(lines)
    print("\n" + txt)
    with open(os.path.join(out_dir, "rapport_test.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    plot_confusion(test["cm"], classes, os.path.join(out_dir, "matrice_confusion.png"))
    # chiffres clés pour la comparaison des modèles (07_comparaison.py)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"arch": args.arch, "classes": classes, "test_acc": test["acc"],
                   "test_f1_macro": test["f1_macro"], "val_f1_macro": best_f1, "meilleure_epoque": best_epoch,
                   "epoques": len(hist["train_loss"]), "duree_entrainement_min": round(duree_min, 1),
                   "parametres_millions": round(sum(q.numel() for q in model.parameters()) / 1e6, 1)}, f, indent=1)
    print(f"Durée d'entraînement : {duree_min:.1f} min")
    print(f"\nRésultats dans {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
