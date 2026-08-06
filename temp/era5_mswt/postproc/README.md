# Postprocessing

Ce dossier contient les metriques et figures de post-traitement pour les predictions du pipeline.

L'etat actuel du projet documente la configuration principale :

```text
ERA5 -> MSWT
```

Les configs utilisees par defaut s'appellent :

```text
config.yaml
```

Chaque dossier de metrique contient sa propre config `config.yaml`, qui pointe vers la config principale :

```text
temp/main/configs/unet/config_arch1.yaml
```

## Structure

```text
temp/postproc/
├── mean/
│   ├── bias/
│   ├── rmse/
│   └── correlation/
├── extreme/
│   ├── b02/
│   ├── b98/
│   ├── cams/
│   └── wams/
├── temporal/
│   ├── ac1/
│   └── rstd/
├── bano_compare/
├── explore_config.yaml
└── run_postproc.sh
```

## Lancer tout le postprocessing

Depuis la racine du projet :

```bash
bash temp/postproc/run_postproc.sh
```

Equivalent explicite :

```bash
bash temp/postproc/run_postproc.sh config
```

Avec un groupe seulement :

```bash
bash temp/postproc/run_postproc.sh config mean
bash temp/postproc/run_postproc.sh config extreme
bash temp/postproc/run_postproc.sh config temporal
```

Groupes disponibles :

```text
all
mean
extreme
temporal
```

## Metriques mean

Bias :

```bash
python -m temp.postproc.mean.bias.bias temp/postproc/mean/bias/config.yaml
python -m temp.postproc.mean.bias.plot temp/postproc/mean/bias/config.yaml
```

RMSE :

```bash
python -m temp.postproc.mean.rmse.rmse temp/postproc/mean/rmse/config.yaml
python -m temp.postproc.mean.rmse.plot temp/postproc/mean/rmse/config.yaml
```

Correlation :

```bash
python -m temp.postproc.mean.correlation.corr temp/postproc/mean/correlation/config.yaml
python -m temp.postproc.mean.correlation.plot temp/postproc/mean/correlation/config.yaml
```

## Metriques extreme

B02 :

```bash
python -m temp.postproc.extreme.b02.b02 temp/postproc/extreme/b02/config.yaml
python -m temp.postproc.extreme.b02.plot temp/postproc/extreme/b02/config.yaml
```

B98 :

```bash
python -m temp.postproc.extreme.b98.b98 temp/postproc/extreme/b98/config.yaml
python -m temp.postproc.extreme.b98.plot temp/postproc/extreme/b98/config.yaml
```

CAMS :

```bash
python -m temp.postproc.extreme.cams.cams temp/postproc/extreme/cams/config.yaml
python -m temp.postproc.extreme.cams.plot temp/postproc/extreme/cams/config.yaml
```

WAMS :

```bash
python -m temp.postproc.extreme.wams.wams temp/postproc/extreme/wams/config.yaml
python -m temp.postproc.extreme.wams.plot temp/postproc/extreme/wams/config.yaml
```

## Metriques temporal

AC1 :

```bash
python -m temp.postproc.temporal.ac1.ac1 temp/postproc/temporal/ac1/config.yaml
python -m temp.postproc.temporal.ac1.plot temp/postproc/temporal/ac1/config.yaml
```

RSTD :

```bash
python -m temp.postproc.temporal.rstd.rstd temp/postproc/temporal/rstd/config.yaml
python -m temp.postproc.temporal.rstd.plot temp/postproc/temporal/rstd/config.yaml
```

## Sorties

Chaque module sauvegarde ses resultats dans le dossier de resultats configure par le YAML principal.

Les sorties typiques sont :

```text
summary JSON/CSV
NetCDF de metrique
figures PNG
```

## Notes

- `run_postproc.sh` utilise `config.yaml` par defaut.
- Si une autre config est passee par erreur au script, elle est ignoree et `config.yaml` est utilisee.
- Les chemins `main_config_path` doivent pointer vers `temp/main/configs/unet/config_arch1.yaml` pour l'experience courante.
