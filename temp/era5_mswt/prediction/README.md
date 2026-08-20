# Pipeline de prédiction

Ce dossier exécute les prédictions de température, compare les distributions des prédicteurs et produit les diagnostics post-traités. Les imports et chemins sont résolus depuis l’emplacement des scripts : les commandes ne dépendent donc pas du répertoire courant.

## Architecture

```text
temp/era5_mswt/
├── main/                 # modèles, chargement des données et configurations
├── postproc/             # utilitaires cartographiques partagés
├── prediction/
│   ├── config.yaml       # configuration unique de la prédiction
│   ├── predict.py        # prédiction LMDZ configurée
│   ├── predict_era5_downscaled.py
│   ├── predictors_dist.py
│   ├── check_datasets.py
│   └── postproc/         # métriques et figures de la prédiction
└── results/              # modèles, statistiques et sorties
```

`backend_paths.py` relie `prediction/` aux modules de `main/` et de `postproc/`. Ne pas ajouter manuellement de chemin absolu à `sys.path`.

## Configuration

Modifier `config.yaml` avant l’exécution. Les chemins relatifs sont interprétés depuis `prediction/` :

- `paths.root_dir` : racine des NetCDF (`DATA1` par défaut) ;
- `paths.*_pattern` et `paths.mswt_path` : chemins relatifs à `root_dir` ;
- `paths.shapefile_path` : shapefile du Maroc ;
- `prediction.models_dir` : répertoire contenant l’expérience entraînée ;
- `prediction.output_dir` : destination des NetCDF prédits ;
- `dates.test` et `dates.reference` : périodes de prédiction et de correction du biais.

Le modèle attendu est `results/<experiment>/best_model.pth` (avec les statistiques de normalisation de l’expérience). Vérifier que les fichiers de données correspondant aux motifs configurés existent.

## Exécution

Depuis la racine du dépôt :

```bash
python -m temp.era5_mswt.prediction.check_datasets
python -m temp.era5_mswt.prediction.predict
python -m temp.era5_mswt.prediction.predict_era5_downscaled
python -m temp.era5_mswt.prediction.predictors_dist
```

Sur Slurm :

```bash
export CONDA_ENV_NAME=env_torch   # facultatif si l’environnement est déjà actif
sbatch temp/era5_mswt/prediction/predict.sh
```

## Post-traitement

Les modules doivent être lancés avec `-m` afin que leurs imports relatifs fonctionnent. Exemple :

```bash
python -m temp.era5_mswt.prediction.postproc.climatology.compute_climatology
python -m temp.era5_mswt.prediction.postproc.climatology.plot
python -m temp.era5_mswt.prediction.postproc.bias.compute_bias
python -m temp.era5_mswt.prediction.postproc.bias.plot
python -m temp.era5_mswt.prediction.postproc.rmse.compute_rmse
python -m temp.era5_mswt.prediction.postproc.rmse.plot
python -m temp.era5_mswt.prediction.postproc.correlation.compute_correlation
python -m temp.era5_mswt.prediction.postproc.correlation.plot
python -m temp.era5_mswt.prediction.postproc.distributions.compute_distributions
python -m temp.era5_mswt.prediction.postproc.distributions.plot
python -m temp.era5_mswt.prediction.postproc.monthly_cycle.compute_monthly_cycle
python -m temp.era5_mswt.prediction.postproc.monthly_cycle.plot
python -m temp.era5_mswt.prediction.postproc.seasonal_cycle.compute_seasonal_cycle
python -m temp.era5_mswt.prediction.postproc.seasonal_cycle.plot
python -m temp.era5_mswt.prediction.postproc.summary.compute_summary_tables
python -m temp.era5_mswt.prediction.postproc.summary.plot
```

Les résultats des métriques sont écrits sous `prediction.output_dir/metrics/<métrique>/{data,plots}`.

## Dépendances principales

Python 3.10+, PyTorch, xarray, NumPy, pandas, PyYAML, matplotlib, SciPy, dask, netCDF4 ou h5netcdf, geopandas, shapely et cartopy. Les versions doivent rester compatibles avec l’environnement utilisé pour entraîner le modèle.

## Vérifications rapides

```bash
python -m compileall -q temp/era5_mswt/prediction
python -c "from temp.era5_mswt.prediction.backend_paths import MAIN_DIR, POSTPROC_DIR; print(MAIN_DIR, POSTPROC_DIR)"
```

`check_datasets.py` génère `quick_hist_summary.csv` dans ce dossier. Une prédiction complète nécessite les NetCDF et le checkpoint ; les vérifications statiques seules ne les remplacent pas.
