from __future__ import annotations

from importlib import import_module
from pathlib import Path
import sys


PREDICTION_DIR = Path(__file__).resolve().parent
ERA5_MSWT_ROOT = PREDICTION_DIR.parent
TEMP_ROOT = ERA5_MSWT_ROOT.parent
PROJECT_ROOT = TEMP_ROOT.parent
REPOSITORY_ROOT = PROJECT_ROOT
WORKSPACE_ROOT = REPOSITORY_ROOT.parent

ERA5_MSWT_MAIN_DIR = ERA5_MSWT_ROOT / "main"
ERA5_MSWT_POSTPROC_DIR = ERA5_MSWT_ROOT / "postproc"

LEGACY_MAIN_DIR = TEMP_ROOT / "main"
LEGACY_POSTPROC_DIR = TEMP_ROOT / "postproc"

if ERA5_MSWT_MAIN_DIR.exists():
    MAIN_DIR = ERA5_MSWT_MAIN_DIR
    MAIN_PACKAGE = "temp.era5_mswt.main"
else:
    MAIN_DIR = LEGACY_MAIN_DIR
    MAIN_PACKAGE = "temp.main"

if ERA5_MSWT_POSTPROC_DIR.exists():
    POSTPROC_DIR = ERA5_MSWT_POSTPROC_DIR
    POSTPROC_PACKAGE = "temp.era5_mswt.postproc"
else:
    POSTPROC_DIR = LEGACY_POSTPROC_DIR
    POSTPROC_PACKAGE = "temp.postproc"

CONFIGS_DIR = MAIN_DIR / "configs"


def ensure_backend_on_path() -> None:
    for path in (REPOSITORY_ROOT, MAIN_DIR):
        path_str = str(path)
        while path_str in sys.path:
            sys.path.remove(path_str)
        sys.path.insert(0, path_str)


def import_main_module(module_path: str):
    ensure_backend_on_path()
    return import_module(f"{MAIN_PACKAGE}.{module_path}")


def import_postproc_module(module_path: str):
    ensure_backend_on_path()
    return import_module(f"{POSTPROC_PACKAGE}.{module_path}")
