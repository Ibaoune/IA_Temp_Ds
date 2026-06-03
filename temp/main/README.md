# Main pipeline

Ce dossier contient la partie principale du pipeline de descente d'echelle statistique.

L'etat actuel du projet documente une seule approche supportee :

```text
ERA5 -> MSWT
```

Les anciennes approches non presentes dans le projet ne sont plus documentees ici.

## Structure utile

```text
temp/main/
├── configs/
│   ├── unet/
│   │   ├── config_arch.yaml
│   │   ├── test_arch.yaml
│   │   ├── config_arch1.yaml
│   │   └── test_arch1.yaml
│   ├── cnn/
│   └── glm/
├── scripts/
│   ├── job_cpu.sh
│   └── job_gpu.sh
├── src/
├── train.py
└── eval.py
```

## Configurations U-Net

Les configs U-Net sont regroupees dans :

```text
temp/main/configs/unet/
```

Correspondance principale :

```text
model_type: "unet"  -> temp/main/configs/unet/config_arch.yaml
model_type: "unet1" -> temp/main/configs/unet/config_arch1.yaml
```

Pour les jobs de test/validation utilises actuellement :

```text
temp/main/configs/unet/test_arch1.yaml
```

## Approche supportee

### ERA5 -> MSWT

Cette approche apprend la relation entre les predicteurs ERA5 et la cible MSWT.

```yaml
general:
  src: "era5"
  target: "mswt"
  variable: "temp"
  model_type: "unet1"
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
python temp/main/train.py temp/main/configs/unet/test_arch1.yaml
```

Avec l'environnement local Windows :

```powershell
.\climate_env\Scripts\python.exe temp/main/train.py temp/main/configs/unet/test_arch1.yaml
```

## Lancer l'evaluation

Depuis la racine du projet :

```bash
python temp/main/eval.py temp/main/configs/unet/test_arch1.yaml
```

Avec l'environnement local Windows :

```powershell
.\climate_env\Scripts\python.exe temp/main/eval.py temp/main/configs/unet/test_arch1.yaml
```

## Jobs SLURM

CPU :

```bash
bash temp/main/scripts/job_cpu.sh
```

GPU :

```bash
bash temp/main/scripts/job_gpu.sh
```

Les deux scripts utilisent par defaut :

```text
temp/main/configs/unet/test_arch1.yaml
```

On peut changer la config sans modifier le script :

```bash
MAIN_CONFIG=temp/main/configs/unet/test_arch1.yaml bash temp/main/scripts/job_cpu.sh
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
- La resolution automatique des configs U-Net est geree par `resolve_model_config_path` dans `temp/main/src/core/config.py`.
