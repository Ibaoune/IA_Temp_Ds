import os
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import yaml

# ==========================================================
# CONFIG
# ==========================================================
PREDICTION_DIR = Path(__file__).resolve().parent
with open(PREDICTION_DIR / "config.yaml", "r", encoding="utf-8") as config_file:
    _CONFIG = yaml.safe_load(config_file) or {}

_root_value = Path(_CONFIG.get("paths", {}).get("root_dir", "../../../../DATA1"))
ROOT = _root_value if _root_value.is_absolute() else (PREDICTION_DIR / _root_value).resolve()

TARGET_FOLDERS = {
    "LMDZ_35_hist": ROOT / "lmdz_35" / "present_1979_1985" / "all_Mor",
    "LMDZ_250_hist": ROOT / "lmdz_250" / "present" / "all_Mor",
    "ERA5_predictors": ROOT / "predictors_era5",
    "MSWT_predictand": ROOT / "predictand_mswet",
}

OUTPUT_CSV = PREDICTION_DIR / "quick_hist_summary.csv"


# ==========================================================
# HELPERS
# ==========================================================
def open_dataset_safe(path):
    """Ouvre un fichier NetCDF avec plusieurs moteurs si besoin."""
    last_error = None
    for engine in [None, "netcdf4", "h5netcdf", "scipy"]:
        try:
            kwargs = {"decode_times": True}
            if engine is not None:
                kwargs["engine"] = engine
            return xr.open_dataset(path, **kwargs)
        except Exception as e:
            last_error = e
    raise last_error


def find_time_name(ds):
    candidates = ["time", "time_counter", "valid_time", "t"]
    for name in candidates:
        if name in ds.coords or name in ds.dims:
            return name

    for name in list(ds.coords) + list(ds.dims):
        if "time" in name.lower():
            return name
    return None


def find_main_var(ds, time_name=None):
    """Choisit la variable principale à explorer."""
    data_vars = list(ds.data_vars)
    if not data_vars:
        return None

    # priorité à une variable qui contient le temps
    if time_name is not None:
        for v in data_vars:
            if time_name in ds[v].dims:
                return v

    return data_vars[0]


def safe_year_info(time_da):
    try:
        years = np.unique(time_da.dt.year.values)
        years = [int(y) for y in years]
        return years
    except Exception:
        return []


def format_years(years):
    if not years:
        return "N/A"
    if len(years) <= 10:
        return str(years)
    return f"{years[:5]} ... {years[-5:]}"


def analyze_file(nc_path):
    result = {
        "file": nc_path.name,
        "full_path": str(nc_path),
        "variable": None,
        "units": None,
        "dims": None,
        "shape": None,
        "time_name": None,
        "start_date": None,
        "end_date": None,
        "n_years": None,
        "years": None,
        "missing_values": None,
        "total_values": None,
        "missing_pct": None,
        "vmin": None,
        "vmax": None,
        "status": "OK",
    }

    ds = None
    try:
        ds = open_dataset_safe(nc_path)

        print(f"\nFile: {nc_path.name}")
        print("Data variables:", list(ds.data_vars))
        print("Coords:", list(ds.coords))
        print("Dims:", dict(ds.sizes))

        # --------------------------------------------------
        # 1) Trouver la coordonnée temps
        # --------------------------------------------------
        time_name = find_time_name(ds)
        result["time_name"] = time_name

        # --------------------------------------------------
        # 2) Choisir la vraie variable principale
        #    on exclut les variables de bornes
        # --------------------------------------------------
        excluded = {"time_counter_bnds", "time_bnds", "bounds", "lat_bnds", "lon_bnds"}

        candidates = [
            v for v in ds.data_vars
            if v not in excluded and not v.endswith("_bnds")
        ]

        if not candidates:
            result["status"] = "No main variable found"
            print("No main variable found")
            return result

        # priorité à une variable qui contient le temps
        var_name = None
        if time_name is not None:
            for v in candidates:
                if time_name in ds[v].dims:
                    var_name = v
                    break

        if var_name is None:
            var_name = candidates[0]

        da = ds[var_name]

        units = (
            da.attrs.get("units")
            or da.attrs.get("unit")
            or da.attrs.get("Units")
            or "N/A"
        )

        result["variable"] = var_name
        result["units"] = units
        result["dims"] = str(dict(ds.sizes))
        result["shape"] = str(tuple(da.shape))

        print("Chosen variable:", var_name)
        print("Units:", units)
        print("Shape:", da.shape)
        print("Dims of variable:", da.dims)

        # --------------------------------------------------
        # 3) NaN / taille totale
        # --------------------------------------------------
        missing_values = int(da.isnull().sum().values)
        total_values = int(da.size)
        missing_pct = 100.0 * missing_values / total_values if total_values > 0 else 0.0

        result["missing_values"] = missing_values
        result["total_values"] = total_values
        result["missing_pct"] = round(missing_pct, 6)

        print("NaN count:", missing_values)

        # --------------------------------------------------
        # 4) Min / Max
        # --------------------------------------------------
        try:
            vmin = float(da.min(skipna=True).values)
            vmax = float(da.max(skipna=True).values)
            result["vmin"] = vmin
            result["vmax"] = vmax
            print("Min:", vmin)
            print("Max:", vmax)
        except Exception as e:
            print(f"Min/Max not available: {e}")

        # --------------------------------------------------
        # 5) Infos temporelles
        # --------------------------------------------------
        if time_name is not None and time_name in ds:
            t = ds[time_name]

            if t.size > 0:
                try:
                    time_idx = pd.to_datetime(t.values)
                except Exception:
                    try:
                        import cftime
                        if isinstance(t.values[0], cftime.datetime):
                            time_idx = pd.Index([
                                pd.Timestamp(x.strftime("%Y-%m-%d %H:%M:%S"))
                                for x in t.values
                            ])
                        else:
                            time_idx = None
                    except Exception:
                        time_idx = None

                if time_idx is not None:
                    result["start_date"] = time_idx[0].strftime("%Y-%m-%d")
                    result["end_date"] = time_idx[-1].strftime("%Y-%m-%d")

                    years = sorted(pd.Index(time_idx.year).unique().tolist())
                    result["n_years"] = len(years)

                    if len(years) <= 10:
                        result["years"] = str(years)
                    else:
                        result["years"] = f"{years[:5]} ... {years[-5:]}"

                    print("Start:", result["start_date"])
                    print("End  :", result["end_date"])

        return result

    except Exception as e:
        result["status"] = f"ERROR: {e}"
        print(f"[ERROR] Failed to process file: {e}")
        return result

    finally:
        if ds is not None:
            ds.close()


# ==========================================================
# MAIN
# ==========================================================
def quick_explore():
    rows = []

    print("\n" + "=" * 80)
    print("QUICK EXPLORATION - HISTORICAL DATA ONLY")
    print("=" * 80)

    for label, folder in TARGET_FOLDERS.items():
        print(f"\n[{label}]")
        print(f"Folder: {folder}")

        if not folder.exists():
            print("  -> Folder not found")
            continue

        nc_files = sorted(folder.glob("*.nc"))
        if not nc_files:
            print("  -> No .nc files found")
            continue

        for nc_file in nc_files:
            info = analyze_file(nc_file)
            info["dataset_group"] = label
            rows.append(info)

            print(f"\n  File           : {info['file']}")
            print(f"  Variable       : {info['variable']}")
            print(f"  Dims           : {info['dims']}")
            print(f"  Units          : {info['units']}")
            print(f"  Shape          : {info['shape']}")
            print(f"  Time coord     : {info['time_name']}")
            print(f"  Start date     : {info['start_date']}")
            print(f"  End date       : {info['end_date']}")
            print(f"  Number of years: {info['n_years']}")
            print(f"  Years          : {info['years']}")
            print(f"  Missing values : {info['missing_values']} / {info['total_values']} ({info['missing_pct']}%)")
            print(f"  Status         : {info['status']}")

    if rows:
        df = pd.DataFrame(rows)
        cols = [
            "dataset_group", "file", "variable", "units", "shape",
            "start_date", "end_date", "n_years", "years",
            "missing_values", "total_values", "missing_pct", "status"
        ]
        df[cols].to_csv(OUTPUT_CSV, index=False)

        print("\n" + "=" * 80)
        print("SUMMARY TABLE")
        print("=" * 80)
        print(df[cols].to_string(index=False))
        print(f"\nCSV saved to: {OUTPUT_CSV}")
    else:
        print("\nNo results collected.")


if __name__ == "__main__":
    quick_explore()
