# Détection et classification d'avions sur images satellites

Projet du cours « Introduction à l'IA » (IPSA). L'application trouve les avions sur une image satellite et donne le modèle de chacun parmi 20 classes (A1 à A20).

## Fonctionnement

Le système travaille en deux étapes.

```
image satellite
   │
   ├─ 1. DÉTECTION : fenêtre glissante + CNN « avion / fond » (ResNet18)
   │     ~19 000 fenêtres carrées de 10 tailles → score « avion »
   │     → seuil → NMS → suppression des boîtes incluses → fusion des fenêtres
   │
   └─ 2. CLASSIFICATION : chaque avion détecté est découpé (carré, marge 15 %, 224 px)
         → CNN 20 classes (ResNet50) → A1 … A20
```

Les deux réseaux sont des **ResNet pré-entraînés sur ImageNet** (transfer learning), puis réentraînés sur nos données.

## Résultats sur le jeu de test

| | ResNet18 | ResNet50 |
|---|---|---|
| **Classification** : accuracy | 97,9 % | **98,2 %** |
| **Classification** : F1-macro | 97,3 % | **98,0 %** |
| **Détection** : précision | **87,2 %** | 84,3 % |
| **Détection** : rappel | **91,4 %** | 89,7 % |
| **Détection** : F1 (IoU ≥ 0,5) | **89,3 %** | 86,9 % |
| Détection : temps par image (GPU T4) | **1,8 s** | 5,6 s |

**Choix final : ResNet18 pour détecter, ResNet50 pour classifier.**

**Chaîne complète** (269 images de test, 1181 avions) : un avion compte comme correct s'il est bien encadré (IoU ≥ 0,5) **et** bien classé.

| Précision | Rappel | F1 | Avions bien localisés et bien classés | Temps (GPU T4) |
|---|---|---|---|---|
| **85,2 %** | **89,3 %** | **87,2 %** | 1055 sur 1081 (97,6 %) | 1,7 s/image |

- Pour classer 20 modèles d'avions sur des vignettes de 224 px, le plus gros réseau est légèrement meilleur.
- Pour séparer « avion » et « fond » sur de petites fenêtres de 96 px, ResNet50 surapprend, alors que ResNet18 est plus précis et 3 fois plus rapide.

Le graphique de comparaison se trouve dans `outputs/comparaison/comparaison.png`.

## Données

- 1331 images satellites et 7870 avions annotés au format Pascal VOC (XML), répartis en 20 classes déséquilibrées : 785 avions pour A16, 107 pour A6.
- Les images sont découpées en 70 % entraînement, 15 % validation et 15 % test. Ce découpage se fait **par image**, pour éviter les fuites de données, et il est **stratifié** : chaque classe est présente dans les mêmes proportions dans les trois ensembles.

## Organisation du code

| Fichier | Rôle |
|---|---|
| `01_analyse.py` | Analyse du jeu de données : statistiques et graphiques (`outputs/analyse/`) |
| `02_preparation.py` | Découpage train/val/test et vignettes de chaque avion (`data/crops/`) |
| `03_classification.py` | Entraînement d'un CNN : classifieur 20 classes, ou détecteur avion/fond |
| `04_visualisation.py` | Galerie des prédictions et des erreurs du classifieur |
| `05_preparation_detection.py` | Fenêtres « avion » et « fond » (dont des négatifs difficiles) pour le détecteur |
| `06_detection.py` | Évaluation du détecteur et choix des réglages (seuil, NMS…) sur la validation |
| `07_comparaison.py` | Comparaison entre ResNet18 et ResNet50 |
| `08_application.py` | **Application Streamlit** (interface graphique) |
| `09_corpus_annote.py` | Corpus de test annoté par le modèle : images, XML et score de la chaîne complète |
| `src/` | Lecture des annotations VOC, fenêtre glissante, chaîne complète |
| `colab_*.ipynb` | Notebooks d'entraînement sur Google Colab (GPU) |

## Lancer l'application

Les modèles entraînés (fichiers `best_model.pt`) ne sont pas dans le dépôt, car ils sont trop lourds. Il faut les placer dans `outputs/classification/<nom du modèle>/`.

```bash
pip install -r requirements.txt
streamlit run 08_application.py      # puis ouvrir http://localhost:8501
```

Sous Windows, il suffit aussi de double-cliquer sur `lancer_application.bat`.

## Reproduire les résultats

```bash
python 01_analyse.py
python 02_preparation.py
python 03_classification.py --arch resnet50                      # classifieur (GPU conseillé)
python 05_preparation_detection.py
python 03_classification.py --data data/fenetres --arch resnet18 --size 96 --batch 128 --lr 1e-3 --epochs 12 --patience 4 --name detecteur_resnet18
python 06_detection.py --name detecteur_resnet18
python 09_corpus_annote.py
```
