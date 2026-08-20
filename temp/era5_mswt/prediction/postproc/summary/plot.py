from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..common import PRED_CONFIG, get_metric_dirs, load_prediction_settings


MODEL_ORDER = ["MSWT", "ERA5_downscaled", "LMDZ_35", "LMDZ_250"]
COLORS = {
    "MSWT": "#2f2f2f",
    "ERA5_downscaled": "#4c78a8",
    "LMDZ_35": "#59a14f",
    "LMDZ_250": "#e15759",
}


def _ordered_series(series: pd.Series) -> pd.Series:
    labels = [model for model in MODEL_ORDER if model in series.index]
    return series.reindex(labels)


def _metric_series(df: pd.DataFrame, axis: str, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype=float)

    sub = df[(df["analysis_axis"] == axis) & df[column].notna()]
    if sub.empty:
        return pd.Series(dtype=float)

    return _ordered_series(sub.groupby("model")[column].mean())


def _fallback_rmse_series(df: pd.DataFrame) -> pd.Series:
    if "rmse_clim" not in df.columns:
        return pd.Series(dtype=float)

    sub = df[
        df["analysis_axis"].isin(["monthly_cycle", "seasonal_cycle"])
        & df["rmse_clim"].notna()
        & (df["model"] != "MSWT")
    ]
    if sub.empty:
        return pd.Series(dtype=float)

    return _ordered_series(sub.groupby("model")["rmse_clim"].mean())


def _draw_bar(ax, series: pd.Series, title: str, ylabel: str, zero_line: bool = False) -> bool:
    series = series.dropna()
    if series.empty:
        ax.set_visible(False)
        return False

    colors = [COLORS.get(model, "#8f8f8f") for model in series.index]
    ax.bar(series.index, series.values, color=colors, edgecolor="#404040", linewidth=0.5)
    if zero_line:
        ax.axhline(0.0, color="#2f2f2f", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_title(title, fontweight="bold", pad=10)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.28)
    ax.tick_params(axis="x", rotation=25)
    return True


def main() -> None:
    settings = load_prediction_settings()
    data_dir, plot_dir = get_metric_dirs(settings, "summary")
    csv_path = data_dir / (
        f"global_postproc_summary_{settings.selected_start}_{settings.selected_end}_{settings.output_tag}.csv"
    )

    print("=== Plotting global post-processing summary ===")
    print(f"Prediction config : {PRED_CONFIG}")
    print(f"Selected period   : {settings.selected_start} -> {settings.selected_end}")
    print(f"Mode              : {settings.output_tag}")
    print(f"Evaluation domain : {settings.eval_domain}")
    print(f"Display domain    : {settings.display_domain}")
    print(f"Input CSV         : {csv_path}")
    print(f"Output plot dir   : {plot_dir}")

    if not csv_path.exists():
        print(f"[WARNING] Missing global summary CSV: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("[WARNING] Global summary CSV is empty; no summary plot produced.")
        return

    mean_temp = _metric_series(df, "climatology", "mean_temp")
    mean_bias = _metric_series(df, "climatology", "bias_mean")
    mean_rmse = _metric_series(df, "rmse", "rmse_mean")
    if mean_rmse.empty:
        mean_rmse = _fallback_rmse_series(df)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.2), dpi=180)
    plotted = [
        _draw_bar(axes[0], mean_temp, "Mean temperature", "Temperature (C)"),
        _draw_bar(axes[1], mean_bias, "Mean bias", "Bias (C)", zero_line=True),
        _draw_bar(axes[2], mean_rmse, "Mean RMSE", "RMSE (C)"),
    ]

    if not any(plotted):
        plt.close(fig)
        print("[WARNING] No compatible columns found for summary plot.")
        return

    fig.suptitle(f"Post-processing summary over land ({settings.output_tag})", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig_path = plot_dir / "global_postproc_summary_barplots_land.png"
    fig.savefig(fig_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[SUCCESS] Figure saved to: {fig_path}")


if __name__ == "__main__":
    main()
