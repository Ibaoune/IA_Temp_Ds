from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from ...main.src.core.utils import build_experiment_path

from .common_bano import (
    DEFAULT_CONFIG,
    get_main_cfg_and_bano_cfg,
    ensure_bano_output_dir,
    flatten_valid,
    open_metric_field,
)


# ==========================================================
# Baño-style metric layout
# ==========================================================

METRIC_SPECS = [
    # Fixed y-limits inspired by the Baño-Medina figure, expanded only as needed.
    {
        "panel": "(a)",
        "title": "",
        "ylabel": "RMSE (°C)",
        "metric_name": "rmse",
        "patterns": [
            "rmse_annual_daily_first_mean_period.nc",
            "rmse_annual_mean_period.nc",
        ],
        "vars": ["rmse"],
        "ylim": (0.7, 2.76),
        "ref": 1.05,
    },
    {
        "panel": "(b)",
        "title": "",
        "ylabel": "Cor. deseasonal",
        "metric_name": "corr_d",
        "patterns": [
            "corr_d_annual_mean_period.nc",
        ],
        "vars": ["corr_d"],
        "ylim": (0.86, 1.00),
        "ref": 0.96,
    },
    {
        "panel": "(c)",
        "title": "",
        "ylabel": "Standard dev. ratio",
        "metric_name": "rstd",
        "patterns": [
            "rstd_annual_mean_period.nc",
        ],
        "vars": ["rstd", "std_ratio", "ratio_std"],
        "ylim": (0.68, 1.03),
        "ref": 1.00,
    },
    {
        "panel": "(d)",
        "title": "",
        "ylabel": "Bias (°C)",
        "metric_name": "bias",
        "patterns": [
            "bias_annual_daily_first_mean_period.nc",
            "bias_annual_mean_first_mean_period.nc",
            "bias_annual_mean_period.nc",
        ],
        "vars": ["bias"],
        "ylim": (-1.82, 0.91),
        "ref": 0.0,
    },
    {
        "panel": "(e)",
        "title": "",
        "ylabel": "Bias P02 (°C)",
        "metric_name": "b02",
        "patterns": [
            "b02_annual_mean_period.nc",
        ],
        "vars": ["b02", "bias_p02", "bp02"],
        "ylim": (-0.95, 2.66),
        "ref": 0.55,
    },
    {
        "panel": "(f)",
        "title": "",
        "ylabel": "Bias P98 (°C)",
        "metric_name": "b98",
        "patterns": [
            "b98_annual_mean_period.nc",
        ],
        "vars": ["b98", "bias_p98", "bp98"],
        "ylim": (-3.68, 1.0),
        "ref": 0.10,
    },
    {
        "panel": "(g)",
        "title": "",
        "ylabel": "Bias AC1",
        "metric_name": "ac1",
        "patterns": [
            "ac1_annual_mean_period.nc",
        ],
        "vars": ["bac1", "bias_ac1", "ac1"],
        "ylim": (-0.4, 0.4),
        "ref": 0.0,
    },
    {
        "panel": "(h)",
        "title": "",
        "ylabel": "Bias WAMS (days)",
        "metric_name": "wams",
        "patterns": [
            "wams_annual_mean_period.nc",
        ],
        "vars": ["bwams", "bias_wams", "wams"],
        "ylim": (-3, 5),
        "ref": 0.0,
    },
    {
        "panel": "(i)",
        "title": "",
        "ylabel": "Bias CAMS (days)",
        "metric_name": "cams",
        "patterns": [
            "cams_annual_mean_period.nc",
        ],
        "vars": ["bcams", "bias_cams", "cams"],
        "ylim": (-3, 5),
        "ref": 0.0,
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Produce Baño-style Figure 2: metric boxplots."
    )
    parser.add_argument(
        "bano_config",
        nargs="?",
        default=str(DEFAULT_CONFIG),
        help="Path to bano_compare config.yaml",
    )
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()

def compute_auto_ylim(values: np.ndarray, ref: float | None = None, pad_frac: float = 0.12):
    """
    Compute automatic y-limits from actual metric values.
    The reference line is included in the limits.
    """
    values = flatten_valid(values)

    if values.size == 0:
        if ref is None:
            return (-1.0, 1.0)
        return (ref - 1.0, ref + 1.0)

    data_min = float(np.nanmin(values))
    data_max = float(np.nanmax(values))

    if ref is not None and np.isfinite(ref):
        data_min = min(data_min, float(ref))
        data_max = max(data_max, float(ref))

    if data_min == data_max:
        pad = 0.5 if data_min == 0 else abs(data_min) * 0.2
    else:
        pad = pad_frac * (data_max - data_min)

    return data_min - pad, data_max + pad

def make_bano_boxplot(
    ax,
    values: np.ndarray,
    label: str,
    ylabel: str,
    ylim: tuple[float, float] | None,
    ref: float,
    panel: str,
):
    """
    One-panel Baño-style boxplot, with automatic limits adapted to our results.
    """
    values = flatten_valid(values)

    if ylim is None:
        ylim = compute_auto_ylim(values, ref=ref)

    if values.size == 0:
        ax.text(
            0.5,
            0.5,
            "No valid data",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
        )
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_ylim(*ylim)
        return

    ax.boxplot(
        [values],
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        whis=(5, 95),
        medianprops=dict(color="black", linewidth=1.3),
        boxprops=dict(facecolor="0.80", edgecolor="0.45", linewidth=0.8),
        whiskerprops=dict(color="0.45", linestyle=":", linewidth=0.9),
        capprops=dict(color="0.45", linewidth=0.8),
    )

    ax.axhline(ref, color="indianred", linewidth=0.9)

    ax.set_xticks([])
    ax.set_xticklabels([])

    ax.set_ylabel(ylabel, fontsize=12, labelpad=6)
    ax.set_ylim(*ylim)

    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("0.45")

    ax.tick_params(axis="both", labelsize=9)

    ax.text(
        0.93,
        0.94,
        panel,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
    )

def main():
    args = parse_args()

    bano_cfg, cfg, main_cfg_path = get_main_cfg_and_bano_cfg(args.bano_config)

    exp_path = Path(build_experiment_path(cfg))
    out_dir = ensure_bano_output_dir(
        cfg,
        folder_name=bano_cfg.get("output", {}).get("folder_name", "bano_compare"),
    )
    dpi = int(bano_cfg.get("output", {}).get("dpi", 300))

    spatial_domain = bano_cfg.get("spatial", {}).get("eval_domain", "land")
    model_label = bano_cfg.get("experiment", {}).get("model_label", "UNet1")

    print("=== Baño-style Figure 2: metrics boxplots ===")
    print(f"Main config    : {main_cfg_path}")
    print(f"Experiment root: {exp_path}")
    print(f"Spatial domain : {spatial_domain}")
    print(f"Output dir     : {out_dir}")

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(8.0, 8.7),
        dpi=dpi,
    )

    axes_flat = axes.ravel()

    for ax, spec in zip(axes_flat, METRIC_SPECS):
        metric_name = spec["metric_name"]

        try:
            da, path, var_name = open_metric_field(
                exp_path=exp_path,
                metric_name=metric_name,
                eval_domain=spatial_domain,
                patterns=spec["patterns"],
                preferred_vars=spec["vars"],
            )

            print(f"[OK] {metric_name}: {path.name} | variable={var_name}")

            values = da.values

        except Exception as exc:
            print(f"[WARNING] Could not load {metric_name}: {exc}")
            values = np.array([])

        make_bano_boxplot(
            ax=ax,
            values=values,
            label=model_label,
            ylabel=spec["ylabel"],
            ylim=spec["ylim"],
            ref=spec["ref"],
            panel=spec["panel"],
        )
    
    # Single model label for all boxplots
    fig.text(
        0.5,
        0.035,
        model_label,
        ha="center",
        va="center",
        fontsize=12,
    )

    fig.subplots_adjust(
        left=0.10,
        right=0.98,
        top=0.98,
        bottom=0.10,
        wspace=0.42,
        hspace=0.22,
    )

    out_path = out_dir / "fig2_metrics_boxplots.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"[SAVED] {out_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
