# Main — Entraînement et évaluation des modèles de downscaling

Ce dossier contient la partie principale du pipeline de descente d’échelle statistique.  
Il permet de charger les données climatiques, préparer les prédicteurs et la cible, entraîner un modèle IA, puis générer des prédictions sous forme de fichiers NetCDF.

Le pipeline est conçu pour plusieurs approches :

```text
ERA5 2° → MSWT 0.1°
LMDZ250 → LMDZ35
LMDZ35 interpolé à 2° → LMDZ35
```

---

# 1. Organisation du dossier

```text
temp/main/
│
├── configs/
│   ├── cnn/
│   │   └── config.yaml
│   ├── glm/
│   │   └── config.yaml
│   └── unet1/
│       ├── era5_to_mswt.yaml
│       ├── lmdz250_to_lmdz35.yaml
│       └── lmdz35_2deg_to_lmdz35.yaml
│
├── scripts/
│   ├── job_cpu.sh
│   └── job_gpu.sh
│
├── src/
│   ├── core/
│   │   ├── config.py
│   │   ├── evaluation.py
│   │   ├── losses.py
│   │   ├── training.py
│   │   └── utils.py
│   │
│   ├── data/
│   │   ├── data_loading.py
│   │   ├── interpolation.py
│   │   └── preprocessing.py
│   │
│   └── models/
│       ├── cnn.py
│       ├── glm.py
│       ├── unet_arch.py
│       └── unet_arch1.py
│
├── train.py
├── eval.py
└── README.md
```

---

# 2. Rôle du dossier `main`

Le dossier `main` assure les étapes suivantes :

```text
1. Lecture du fichier YAML de configuration
2. Chargement des prédicteurs
3. Chargement du predictand / target
4. Masquage spatial sur la région d’étude
5. Interpolation des prédicteurs vers une grille commune
6. Normalisation des prédicteurs
7. Conversion des unités de la cible
8. Entraînement du modèle IA
9. Sauvegarde des modèles entraînés
10. Évaluation du modèle sur la période de test
11. Sauvegarde des prédictions en NetCDF
12. Génération de figures de diagnostic
```

---

# 3. Approches supportées

## 3.1 ERA5 → MSWT

Cette approche correspond à la chaîne initiale de descente d’échelle :

```text
Prédicteurs : ERA5
Cible       : MSWT
```

Les prédicteurs ERA5 sont interpolés vers une grille grossière de 2°, puis le modèle apprend à reconstruire la température MSWT à fine résolution.

```text
ERA5 2° → MSWT 0.1°
```

Dans le YAML :

```yaml
general:
  src: "era5"
  target: "mswt"
```

---

## 3.2 LMDZ250 → LMDZ35

Cette approche apprend la relation entre une simulation climatique grossière et une simulation régionale plus fine.

```text
Prédicteurs : LMDZ250
Cible       : LMDZ35
```

Le modèle apprend :

```text
LMDZ250 → LMDZ35
```

Dans le YAML :

```yaml
general:
  src: "lmdz250"
  target: "lmdz35"
```

---

## 3.3 LMDZ35 interpolé à 2° → LMDZ35

Cette approche consiste à dégrader LMDZ35 vers une grille grossière de 2°, puis à apprendre à reconstruire LMDZ35.

```text
Prédicteurs : LMDZ35 interpolé à 2°
Cible       : LMDZ35
```

Le modèle apprend :

```text
LMDZ35 2° → LMDZ35
```

Dans le YAML :

```yaml
general:
  src: "lmdz35"
  target: "lmdz35"
```

---

# 5. Section `general`

La section `general` définit l’expérience.

```yaml
general:
  verbose: true
  experiment: "unet1_lmdz250_to_lmdz35"
  variable: "temp"
  target: "lmdz35"
  src: "lmdz250"
  model_type: "unet1"
  interpolation_type: "linear"
  variables: ["z", "q", "t", "u", "v"]
  levels: [500, 700, 850, 1000]
  resolution: 2.0
```

## Champs importants

| Champ | Rôle |
|---|---|
| `experiment` | Nom du dossier de résultats |
| `variable` | Variable étudiée, ici `"temp"` |
| `src` | Source des prédicteurs |
| `target` | Cible utilisée comme référence |
| `model_type` | Modèle utilisé : `unet`, `unet1`, `cnn`, `glm` |
| `variables` | Variables atmosphériques utilisées comme prédicteurs |
| `levels` | Niveaux de pression utilisés |
| `resolution` | Résolution cible des prédicteurs après interpolation |

---

# 6. Section `paths`

La section `paths` indique où se trouvent les données et où sauvegarder les résultats.

## 6.1 ERA5 → MSWT

```yaml
paths:
  root_dir: ""
  results_dir: "./temp/results/"
  shapefile_path: "./morocco_shapefile/morocco_unified_fixed_v2.shp"
  era5_predictor_pattern: "./DATA1/predictors_era5/{var}_1979-2020_levels.nc"
  mswt_path: "./DATA1/predictand_mswet/mswt_1979_2020.nc"
```

## 6.2 LMDZ250 → LMDZ35

```yaml
paths:
  root_dir: ""
  results_dir: "./temp/results/"
  shapefile_path: "./morocco_shapefile/morocco_unified_fixed_v2.shp"
  lmdz250_predictor_pattern: "./DATA1/lmdz_250/present/all_Mor/{lmdz_var}-{suffix}.nc"
  lmdz35_path: "./DATA1/predictand_lmdz/t2m_hist_ATM_DA_1979-2014_lat19to38_lon-20to1.nc"
```

## 6.3 LMDZ35 2° → LMDZ35

```yaml
paths:
  root_dir: ""
  results_dir: "./temp/results/"
  shapefile_path: "./morocco_shapefile/morocco_unified_fixed_v2.shp"
  lmdz35_predictor_pattern: "./DATA1/lmdz_35/present_1979_1985/all_Mor/{lmdz_var}-{suffix}.nc"
  lmdz35_path: "./DATA1/predictand_lmdz/t2m_hist_ATM_DA_1979-2014_lat19to38_lon-20to1.nc"
```

---

# 7. Noms des variables LMDZ

Dans les fichiers LMDZ bruts, les noms des variables ne sont pas directement `z`, `q`, `t`, `u`, `v`.

Le code utilise le mapping suivant :

```text
z → geop
q → rhum + temp, puis conversion en humidité spécifique
t → temp
u → vitu
v → vitv
```

Donc, pour un pattern :

```yaml
lmdz250_predictor_pattern: "./DATA1/lmdz_250/present/all_Mor/{lmdz_var}-{suffix}.nc"
```

le code cherche automatiquement :

```text
geop-hist.nc
rhum-hist.nc
temp-hist.nc
vitu-hist.nc
vitv-hist.nc
```

Pour la variable `q`, elle n’est pas lue directement comme un fichier `q`.  
Elle est reconstruite à partir de :

```text
rhum + temp + niveau de pression
```

---

# 8. Fichiers principaux

## 8.1 `train.py`

Script principal d’entraînement.

Commande :

```bash
python -m temp.main.train <config.yaml>
```

ou :

```bash
python temp/main/train.py <config.yaml>
```

Exemple :

```bash
python temp/main/train.py temp/main/configs/unet1/lmdz250_to_lmdz35.yaml
```

Étapes effectuées :

```text
1. Lire le fichier YAML
2. Charger les données
3. Prétraiter les données
4. Construire le modèle
5. Entraîner le modèle
6. Sauvegarder le meilleur modèle et le dernier modèle
```

---

## 8.2 `eval.py`

Script principal d’évaluation.

Commande :

```bash
python -m temp.main.eval <config.yaml>
```

ou :

```bash
python temp/main/eval.py <config.yaml>
```

Exemple :

```bash
python temp/main/eval.py temp/main/configs/unet1/lmdz250_to_lmdz35.yaml
```

Étapes effectuées :

```text
1. Lire le fichier YAML
2. Charger les données de test
3. Appliquer le même prétraitement
4. Charger le modèle entraîné
5. Générer les prédictions
6. Sauvegarder les prédictions en NetCDF
7. Générer des figures de diagnostic
```

---

# 9. Dossier `src/core`

## 9.1 `config.py`

Ce fichier lit le YAML et transforme les informations en objet `cfg`.

Il définit notamment :

```text
cfg.src
cfg.target
cfg.variable
cfg.model_type
cfg.predictor_pattern
cfg.target_path
cfg.results_dir
cfg.experiment
```

Il choisit automatiquement le pattern des prédicteurs :

```text
src = era5    → era5_predictor_pattern
src = lmdz250 → lmdz250_predictor_pattern
src = lmdz35  → lmdz35_predictor_pattern
src = lmdz    → lmdz_predictor_pattern
```

Et le chemin de la cible :

```text
target = mswt   → mswt_path
target = lmdz35 → lmdz35_path
target = lmdz   → lmdz_path
```

---

## 9.2 `training.py`

Ce fichier contient la boucle d’entraînement.

Il construit le modèle selon :

```yaml
model_type: "unet1"
```

Le modèle prend automatiquement :

```text
in_channels = nombre de prédicteurs
out_shape   = taille spatiale de la cible
```

Donc le même modèle peut fonctionner pour :

```text
ERA5 2° → MSWT 0.1°
LMDZ250 → LMDZ35
LMDZ35 2° → LMDZ35
```

---

## 9.3 `evaluation.py`

Ce fichier :

```text
1. Charge le modèle entraîné
2. Génère les prédictions
3. Sauvegarde un fichier NetCDF
4. Produit des figures de diagnostic
```

Les prédictions de température sont sauvegardées avec la variable :

```text
air_temperature
```

---

## 9.4 `losses.py`

Contient les fonctions de perte.

Pour la température, la loss principale est :

```text
GaussianLoss
```

Le modèle prédit deux sorties :

```text
canal 0 : moyenne prédite
canal 1 : log-variance
```

La loss gaussienne est utilisée pour modéliser l’incertitude.

---

## 9.5 `utils.py`

Contient des fonctions utilitaires :

```text
- affichage verbose
- masquage spatial
- sauvegarde / chargement des modèles
- chemins d’expérience
- figures de diagnostic
- courbes de loss
```

---

# 10. Dossier `src/data`

## 10.1 `data_loading.py`

Ce fichier charge les données.

Il gère :

```text
ERA5
MSWT
LMDZ250
LMDZ35
```

Il effectue :

```text
1. Lecture des fichiers NetCDF
2. Détection des variables
3. Masquage spatial
4. Sélection temporelle
5. Sélection des niveaux de pression
6. Conversion rhum → q pour LMDZ
7. Interpolation des prédicteurs
8. Construction de X, y_train, y_test
```

Sortie principale :

```text
X       : prédicteurs, shape = time × channels × lat × lon
y_train : cible train, shape = time × lat × lon
y_test  : cible test, shape = time × lat × lon
```

---

## 10.2 `interpolation.py`

Contient la fonction :

```python
interpolate_to_target_resolution(...)
```

Elle interpole les prédicteurs vers la résolution demandée dans le YAML :

```yaml
resolution: 2.0
```

Donc les prédicteurs sont ramenés vers une grille commune, par exemple :

```text
ERA5 natif → 2°
LMDZ250 → 2°
LMDZ35 natif → 2°
```

---

## 10.3 `preprocessing.py`

Ce fichier prépare les données pour PyTorch.

Il effectue :

```text
1. Séparation train/test
2. Normalisation des prédicteurs
3. Conversion des unités de la cible
4. Conversion en tenseurs PyTorch
```

Modes de normalisation disponibles :

```text
global
channel
gridbox
```

Le mode utilisé dans les expériences est généralement :

```yaml
norm_mode: "gridbox"
```

---

# 11. Dossier `src/models`

## 11.1 `unet_arch1.py`

Architecture UNet principale utilisée dans les expériences récentes.

Elle contient :

```text
- encodeur
- bottleneck
- décodeur
- skip connections
- tête vers la grille cible
- sortie gaussienne optionnelle
```

Le point important est que la sortie dépend de :

```python
target_size
```

Donc l’architecture s’adapte automatiquement à la taille de la cible :

```text
MSWT
LMDZ35
```

---

## 11.2 `unet_arch.py`

Autre version de UNet.

Elle est gardée pour compatibilité avec les anciennes expériences.

---

## 11.3 `cnn.py`

Modèle CNN de type Baño-like.

Il extrait des caractéristiques sur la grille grossière, puis projette vers la grille cible.

---

## 11.4 `glm.py`

Baseline statistique de type GLM local.

Il entraîne un modèle local pour chaque point de grille cible à partir des voisins de la grille d’entrée.

---

# 12. Résultats générés

Pour une expérience :

```yaml
experiment: "unet1_lmdz250_to_lmdz35"
```

les sorties sont sauvegardées dans :

```text
<results_dir>/unet1_lmdz250_to_lmdz35/
```

Exemple :

```text
temp/results_test_app1/unet1_lmdz250_to_lmdz35/
```

Structure typique :

```text
unet1_lmdz250_to_lmdz35/
├── models/
│   ├── unet1_temp_best.pth
│   └── unet1_temp_last.pth
│
├── output_data/
│   └── unet1_predictions_lmdz35.nc
│
├── output_figs/
│   ├── losses.png
│   ├── spatial_distribution.png
│   └── monthly_means.png
│
└── config.txt
```

---

# 13. Commandes principales

## 13.1 Entraîner ERA5 → MSWT

```bash
python temp/main/train.py temp/main/configs/unet1/era5_to_mswt.yaml
```

## 13.2 Évaluer ERA5 → MSWT

```bash
python temp/main/eval.py temp/main/configs/unet1/era5_to_mswt.yaml
```

---

## 13.3 Entraîner LMDZ250 → LMDZ35

```bash
python temp/main/train.py temp/main/configs/unet1/lmdz250_to_lmdz35.yaml
```

## 13.4 Évaluer LMDZ250 → LMDZ35

```bash
python temp/main/eval.py temp/main/configs/unet1/lmdz250_to_lmdz35.yaml
```

---

## 13.5 Entraîner LMDZ35 2° → LMDZ35

```bash
python temp/main/train.py temp/main/configs/unet1/lmdz35_to_lmdz35.yaml
```

## 13.6 Évaluer LMDZ35 2° → LMDZ35

```bash
python temp/main/eval.py temp/main/configs/unet1/lmdz35_to_lmdz35.yaml
```

---

# 14. Mini-test rapide

Pour vérifier que le code fonctionne sans lancer un entraînement long, on peut utiliser une courte période :

```yaml
dates:
  train:
    start: "1980-01-01"
    end: "1980-01-31"
  test:
    start: "1980-02-01"
    end: "1980-02-05"
```

Et réduire l’entraînement :

```yaml
training:
  epochs: 1
  batch_size: 1

  validation:
    enable: false
```

Cela permet de tester :

```text
chargement des données
prétraitement
construction du modèle
entraînement
sauvegarde
évaluation
```

sans attendre longtemps.

---

# 15. Points importants à vérifier

## 15.1 Compatibilité temporelle

Les prédicteurs et la cible doivent couvrir la même période.

Exemple :

```text
Si les prédicteurs LMDZ35 sont disponibles seulement sur 1979–1985,
alors le train/test doit rester dans 1979–1985.
```

Sinon, `x_test` peut être vide.

---

## 15.2 Compatibilité spatiale

Le domaine utilisé est :

```yaml
region:
  lon_min: -18
  lon_max: 0
  lat_min: 21
  lat_max: 36
```

Le masque spatial est appliqué à la fois aux prédicteurs et à la cible.

Un log correct doit montrer :

```text
[AFTER MASK]
shape : (..., lat > 0, lon > 0)
```

Si une dimension vaut 0, cela signifie qu’il y a un problème de coordonnées ou de domaine.

---

## 15.3 Variables LMDZ

Pour les fichiers LMDZ bruts, les noms réels sont :

```text
geop
rhum
temp
vitu
vitv
```

Le code les associe aux variables génériques :

```text
z
q
t
u
v
```

---

## 15.4 Target LMDZ35

Le fichier cible LMDZ35 actuel est correctement structuré avec :

```text
lat : environ 19 → 38
lon : environ -20 → 1
variable : t2m
unités : K
```

La conversion en Celsius est faite pendant le prétraitement.

---

# 16. Résumé des approches

| Approche | `src` | `target` | Prédicteurs | Cible |
|---|---|---|---|---|
| ERA5 → MSWT | `era5` | `mswt` | ERA5 interpolé à 2° | MSWT |
| LMDZ250 → LMDZ35 | `lmdz250` | `lmdz35` | LMDZ250 interpolé à 2° | LMDZ35 |
| LMDZ35 2° → LMDZ35 | `lmdz35` | `lmdz35` | LMDZ35 interpolé à 2° | LMDZ35 natif |

---

# 17. Workflow recommandé

Pour chaque expérience :

```text
1. Préparer le fichier YAML dans configs/unet1/
2. Lancer train.py
3. Vérifier que le modèle est sauvegardé
4. Lancer eval.py
5. Vérifier que le NetCDF de prédiction est sauvegardé
6. Vérifier les figures output_figs/
7. Passer au postprocessing
```

Exemple complet :

```bash
python temp/main/train.py temp/main/configs/unet1/lmdz250_to_lmdz35.yaml
python temp/main/eval.py temp/main/configs/unet1/lmdz250_to_lmdz35.yaml
bash temp/postproc/run_postproc.sh lmdz250_to_lmdz35
```

---

# 18. Notes finales

Le dossier `main` ne calcule pas les métriques finales comme Bias, RMSE, Corrélation ou Extrêmes.  
Ces métriques sont calculées dans :

```text
temp/postproc/
```

Le dossier `main` sert à produire :

```text
modèles entraînés
prédictions NetCDF
figures de diagnostic directes
```

Le dossier `postproc` sert ensuite à évaluer scientifiquement ces prédictions.