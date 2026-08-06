from __future__ import annotations

import argparse
from pathlib import Path
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt

# Add the main project to path
POSTPROC_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = POSTPROC_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from era5_mswt.postproc.bano_compare.common_bano import (
    get_lon_lat,
    plot_map_bano,
    open_metric_field,
)
from era5_mswt.postproc.bano_compare.fig3_unet_maps import (
    FIG3_SPECS,
    get_bano_norm_and_ticks,
    add_bano_colorbar,
)

def get_config(yaml_path: str):
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)

def load_main_config(config_path: str):
    abs_path = (POSTPROC_DIR / "bano_compare" / config_path).resolve()
    with open(abs_path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Multi-model maps")
    parser.add_argument("compare_config", nargs="?", default="compare_sets.yaml")
    args = parser.parse_args()

    cfg_path = POSTPROC_DIR / "bano_compare" / args.compare_config
    if not cfg_path.exists():
        print(f"Error: {cfg_path} not found.")
        sys.exit(1)

    cmp_cfg = get_config(str(cfg_path))
    spatial_domain = cmp_cfg.get("spatial", {}).get("eval_domain", "land")
    display_domain = "land"
    out_base = POSTPROC_DIR / "results" / cmp_cfg.get("output", {}).get("folder_name", "multi_compare")
    out_base.mkdir(parents=True, exist_ok=True)
    dpi = int(cmp_cfg.get("output", {}).get("dpi", 300))

    sets = cmp_cfg.get("comparison_sets", {})
    if not sets:
        print("No comparison sets defined.")
        return

    for set_name, models in sets.items():
        print(f"\n{'='*50}\nProcessing set: {set_name}\n{'='*50}")
        out_dir = out_base / set_name
        out_dir.mkdir(exist_ok=True)
        
        num_models = len(models)
        # Figure size: width=14, height=3 * num_models
        fig, axes = plt.subplots(num_models, 4, figsize=(14, 3.2 * num_models), dpi=dpi, squeeze=False)
        
        for i, m in enumerate(models):
            m_path = m["path"]
            label = m["label"]
            try:
                main_cfg = load_main_config(m_path)
                exp_name = main_cfg.get("general", {}).get("experiment", "")
                exp_path = POSTPROC_DIR / "results" / exp_name

                shapefile_path = main_cfg.get("paths", {}).get("shapefile_path", None)

                for j, spec in enumerate(FIG3_SPECS):
                    ax = axes[i, j]
                    try:
                        da, path, var_name = open_metric_field(
                            exp_path=exp_path,
                            metric_name=spec["metric_name"],
                            eval_domain=spatial_domain,
                            patterns=spec["patterns"],
                            preferred_vars=spec["vars"],
                        )
                        lons, lats = get_lon_lat(da)
                        arr = da.values
                        
                        _, norm, _ = get_bano_norm_and_ticks(spec["color_kind"])
                        
                        im = plot_map_bano(
                            ax=ax, arr=arr, stats_arr=arr, lons=lons, lats=lats,
                            shapefile_path=shapefile_path, title="", cmap=spec["cmap"],
                            norm=norm, unit=spec["unit"], color_kind=spec["color_kind"],
                            display_domain=display_domain,
                        )
                        
                        # Add title only to the top row
                        if i == 0:
                            ax.set_title(spec["title"], fontsize=12, fontweight="bold", pad=8)
                            
                        # Add colorbar only to the last row
                        if i == num_models - 1:
                            add_bano_colorbar(fig, im, ax, spec["color_kind"])
                            
                    except Exception as e:
                        print(f"    [WARNING] {label} - {spec['metric_name']} failed: {e}")
                        ax.text(0.5, 0.5, "Data unavailable", transform=ax.transAxes, ha="center")
                        ax.axis("off")
                        
            except Exception as exc:
                print(f"  [ERROR] Failed to process model {label}: {exc}")

            # Add row label
            axes[i, 0].text(-0.25, 0.50, label, transform=axes[i, 0].transAxes, rotation=90, 
                            ha="center", va="center", fontsize=14, fontweight="normal")

        fig.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.08, wspace=0.1, hspace=0.1)
        fig.suptitle(f"Map Comparison: {set_name}", fontsize=16, fontweight='bold')
        
        out_path = out_dir / "fig3_multi_maps.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"[SAVED] {out_path}")

if __name__ == "__main__":
    main()
