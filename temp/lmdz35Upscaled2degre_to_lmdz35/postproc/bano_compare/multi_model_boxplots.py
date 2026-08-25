from __future__ import annotations

import argparse
from pathlib import Path
import yaml
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add the main project to path so we can import modules
POSTPROC_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = POSTPROC_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from era5_mswt.main.src.core.utils import build_experiment_path
from era5_mswt.postproc.bano_compare.common_bano import (
    ensure_bano_output_dir,
    flatten_valid,
    open_metric_field,
)
from era5_mswt.postproc.bano_compare.fig2_metrics_boxplots import (
    METRIC_SPECS,
    compute_auto_ylim
)

def get_config(yaml_path: str):
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)

def load_main_config(config_path: str):
    abs_path = (POSTPROC_DIR / "bano_compare" / config_path).resolve()
    with open(abs_path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Multi-model metrics boxplots")
    parser.add_argument("compare_config", nargs="?", default="compare_sets.yaml")
    args = parser.parse_args()

    cfg_path = POSTPROC_DIR / "bano_compare" / args.compare_config
    if not cfg_path.exists():
        print(f"Error: {cfg_path} not found.")
        sys.exit(1)

    cmp_cfg = get_config(str(cfg_path))
    spatial_domain = cmp_cfg.get("spatial", {}).get("eval_domain", "land")
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
        
        # Load data for all models
        model_labels = [m["label"] for m in models]
        model_data = {spec["metric_name"]: [] for spec in METRIC_SPECS}

        for m in models:
            m_path = m["path"]
            try:
                main_cfg = load_main_config(m_path)
                exp_name = main_cfg.get("general", {}).get("experiment", "")
                exp_path = POSTPROC_DIR / "results" / exp_name
                
                print(f"  [{m['label']}] Loading metrics from {exp_path}")

                for spec in METRIC_SPECS:
                    metric_name = spec["metric_name"]
                    try:
                        da, path, var_name = open_metric_field(
                            exp_path=exp_path,
                            metric_name=metric_name,
                            eval_domain=spatial_domain,
                            patterns=spec["patterns"],
                            preferred_vars=spec["vars"],
                        )
                        values = flatten_valid(da.values)
                        model_data[metric_name].append(values)
                    except Exception as e:
                        print(f"    [WARNING] {m['label']} - {metric_name} failed: {e}")
                        model_data[metric_name].append(np.array([]))

            except Exception as exc:
                print(f"  [ERROR] Failed to process model {m['label']}: {exc}")
                for metric_name in model_data:
                    model_data[metric_name].append(np.array([]))

        # Plotting
        num_models = len(models)
        fig, axes = plt.subplots(3, 3, figsize=(14, 10), dpi=dpi)
        axes_flat = axes.ravel()
        
        # Define colors for models (tab20 has 20 distinct colors)
        cmap = plt.cm.get_cmap("tab20")
        colors = [cmap(i % 20) for i in range(num_models)]

        for ax, spec in zip(axes_flat, METRIC_SPECS):
            metric_name = spec["metric_name"]
            data_list = model_data[metric_name]
            
            if not any(d.size > 0 for d in data_list):
                ax.text(0.5, 0.5, "No valid data", transform=ax.transAxes, ha="center", va="center")
                continue
                
            # Compute global ylim for this metric
            all_valid = np.concatenate([d for d in data_list if d.size > 0])
            ylim = compute_auto_ylim(all_valid, ref=spec["ref"])
            
            box = ax.boxplot(
                data_list,
                patch_artist=True,
                showfliers=False,
                whis=(5, 95),
                medianprops=dict(color="black", linewidth=1.3),
                boxprops=dict(linewidth=0.8),
                whiskerprops=dict(color="0.45", linestyle=":", linewidth=0.9),
                capprops=dict(color="0.45", linewidth=0.8),
            )
            
            for patch, color in zip(box['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_edgecolor("0.45")

            ax.axhline(spec["ref"], color="indianred", linewidth=1.2, linestyle="--")
            
            ax.set_xticks(range(1, num_models + 1))
            # Rotate labels if there are many models
            ax.set_xticklabels(model_labels, rotation=45 if num_models > 4 else 0, ha="right" if num_models > 4 else "center", fontsize=8)
            ax.set_ylabel(spec["ylabel"], fontsize=11, labelpad=6)
            ax.set_ylim(*ylim)
            
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)
                spine.set_color("0.45")
                
            ax.text(0.93, 0.94, spec["panel"], transform=ax.transAxes, ha="right", va="top", fontsize=11)

        fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.15, wspace=0.35, hspace=0.45)
        
        # Add legend or super title
        fig.suptitle(f"Metric Comparison: {set_name}", fontsize=14, fontweight='bold')
        
        out_path = out_dir / "fig2_metrics_boxplots.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"[SAVED] {out_path}")

if __name__ == "__main__":
    main()
