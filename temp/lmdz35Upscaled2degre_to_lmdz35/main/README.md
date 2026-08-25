# Main pipeline

Ce dossier contient la partie principale du pipeline de descente d'echelle statistique.

L'etat actuel du projet documente une seule approche supportee :

```text
ERA5 -> MSWT
```

## Structure utile

```text
temp/era5_mswt/main/
├── configs/
│   ├── cnn/
│   ├── CNN/
│   ├── CNN/
│   │   ├── config_arch.yaml
│   │   ├── config_arch1.yaml
│   │   └── autres configurations classiques
│   └── phy_ai/
│       ├── cnn/
│       │   ├── config_cnn1_xiong_continuity.yaml
│       │   ├── config_cnn1_xiong_directional.yaml
│       │   ├── config_cnn1_serifi_gradient.yaml
│       │   ├── config_cnn10_xiong_continuity.yaml
│       │   ├── config_cnn10_xiong_directional.yaml
│       │   └── config_cnn10_serifi_gradient.yaml
│       ├── CNN/
│       └── CNN/
│           ├── config_arch1_xiong_continuity.yaml
│           ├── config_arch1_xiong_directional.yaml
│           └── config_arch1_serifi_gradient.yaml
├── scripts/
│   ├── job_cpu.sh
│   └── job_gpu.sh
├── src/
├── train.py
└── eval.py
```

## Configurations CNN et CNN

Les configurations CNN classiques sont regroupees dans :

```text
temp/era5_mswt/main/configs/CNN/
```

Correspondance principale :

```text
model_type: "CNN"  -> temp/era5_mswt/main/configs/CNN/config_arch.yaml
model_type: "CNN" -> temp/era5_mswt/main/configs/CNN/config_arch1.yaml
```

Les configurations CNN avec contraintes physiques ou structurelles sont
regroupees dans :

```text
temp/era5_mswt/main/configs/phy_ai/CNN/
```

Voir `configs/README.md` pour les expériences Xiong et Serifi et leurs
commandes.

Les variantes Physics-Informed CNN1 et CNN10 sont regroupées dans :

```text
temp/era5_mswt/main/configs/phy_ai/cnn/
```

Elles déclinent séparément les losses `xiong_continuity`,
`xiong_directional` et `serifi_gradient` pour les deux modes CNN.

Chaque combinaison CNN/loss possède aussi une configuration réduite suffixée
`_test.yaml`, par exemple :

```text
config_cnn1_xiong_continuity_test.yaml
config_cnn10_xiong_continuity_test.yaml
```

Ces six variantes utilisent 20 époques, un batch de 4, l'entraînement sur
1980--1984, l'évaluation sur 1985 et le dossier `./temp/results_test/`. Elles
conservent les paramètres d'architecture propres à CNN1 ou CNN10.

La configuration classique de test/validation reste disponible ici :

```text
temp/era5_mswt/main/configs/CNN/test_arch1.yaml
```

## Approche supportee

### ERA5 -> MSWT

Cette approche apprend la relation entre les predicteurs ERA5 et la cible MSWT.

```yaml
general:
  src: "era5"
  target: "mswt"
  variable: "temp"
  model_type: "CNN"
```

Les variables d'entree attendues sont :

```text
z, q, t, u, v
```

Les niveaux verticaux utilises par les configs actuelles sont :

```text
500, 700, 850, 1000 hPa
```

## Lancer l'entrainement

Depuis la racine du projet :

```bash
python temp/era5_mswt/main/train.py temp/era5_mswt/main/configs/CNN/test_arch1.yaml
```

Avec l'environnement local Windows :

```powershell
.\climate_env\Scripts\python.exe temp/era5_mswt/main/train.py temp/era5_mswt/main/configs/CNN/test_arch1.yaml
```

## Lancer l'evaluation

Depuis la racine du projet :

```bash
python temp/era5_mswt/main/eval.py temp/era5_mswt/main/configs/CNN/test_arch1.yaml
```

Avec l'environnement local Windows :

```powershell
.\climate_env\Scripts\python.exe temp/era5_mswt/main/eval.py temp/era5_mswt/main/configs/CNN/test_arch1.yaml
```

## Jobs SLURM

CPU :

```bash
bash temp/era5_mswt/main/scripts/job_cpu.sh
```

GPU :

```bash
bash temp/era5_mswt/main/scripts/job_gpu.sh
```

Les deux scripts utilisent par defaut :

```text
temp/era5_mswt/main/configs/CNN/test_arch1.yaml
```

On peut changer la config sans modifier le script :

```bash
MAIN_CONFIG=temp/era5_mswt/main/configs/CNN/test_arch1.yaml bash temp/era5_mswt/main/scripts/job_cpu.sh
```

## Sorties attendues

Les sorties principales sont placees sous le dossier de resultats configure dans le YAML :

```text
results_dir / experiment /
├── models/
├── output_data/
└── plots/
```

Les noms de modeles restent bases sur :

```text
model_type + variable + tag
```

La reorganisation des configs ne change donc pas les noms des modeles sauvegardes.

## Notes

- Les noms d'experiences restent definis dans les fichiers YAML.
- `model_type` continue a selectionner l'architecture du modele.
- La resolution automatique des configs CNN est geree par `resolve_model_config_path` dans `temp/era5_mswt/main/src/core/config.py`.
