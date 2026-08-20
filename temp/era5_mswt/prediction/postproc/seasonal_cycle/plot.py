from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..common import PRED_CONFIG, get_metric_dirs, load_prediction_settings


MODEL_ORDER = ["MSWT", "ERA5_downscaled", "LMDZ_35", "LMDZ_250"]
SEASONS = ["DJF", "MAM", "JJA", "SON"]
COLORS = {
    "MSWT": "#2f2f2f",
    "ERA5_downscaled": "#4c78a8",
    "LMDZ_35": "#59a14f",
    "LMDZ_250": "#e15759",
}


def _ordered_models(df: pd.DataFrame, include_obs: bool) -> list[str]:
    available = set(df["model"])
    return [model for model in MODEL_ORDER if model in available and (include_obs or model != "MSWT")]


def _plot_grouped_bars(
    df: pd.DataFrame,
    column: str,
    ylabel: str,
    title: str,
    fig_path: Path,
    include_obs: bool,
) -> None:
    models = _ordered_models(df, include_obs=include_obs)
    if not models:
        print(f"[WARNING] No models available for {fig_path.name}; skipping.")
        return

    x = np.arange(len(SEASONS))
    width = min(0.18, 0.76 / max(len(models), 1))
    offsets = (np.arange(len(models)) - (len(models) - 1) / 2.0) * width

    fig, ax = plt.subplots(figsize=(10.5, 5.8), dpi=180)
    for offset, model in zip(offsets, models):
        sub = df[df["model"] == model].set_index("season").reindex(SEASONS)
        ax.bar(
            x + offset,
            sub[column].values,
            width=width,
            color=COLORS.get(model),
            label=model,
            edgecolor="#404040",
            linewidth=0.5,
        )

    if column == "bias":
        ax.axhline(0.0, color="#2f2f2f", linestyle="--", linewidth=1.0, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(SEASONS)
    ax.set_xlabel("Season")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold", pad=10)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.28)
    ax.legend(frameon=False, ncols=2)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[SUCCESS] Figure saved to: {fig_path}")


def main() -> None:
    settings = load_prediction_settings()
    data_dir, plot_dir = get_metric_dirs(settings, "seasonal_cycle")
    csv_path = data_dir / (
        f"seasonal_cycle_summary_{settings.selected_start}_{settings.selected_end}_{settings.output_tag}.csv"
    )

    print("=== Plotting seasonal cycle diagnostics ===")
    print(f"Prediction config : {PRED_CONFIG}")
    print(f"Selected period   : {settings.selected_start} -> {settings.selected_end}")
    print(f"Mode              : {settings.output_tag}")
    print(f"Evaluation domain : {settings.eval_domain}")
    print(f"Display domain    : {settings.display_domain}")
    print(f"Input CSV         : {csv_path}")
    print(f"Output plot dir   : {plot_dir}")

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing seasonal cycle CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    print("Models found      :", _ordered_models(df, include_obs=True))

    _plot_grouped_bars(
        df=df,
        column="mean_temperature",
        ylabel="Temperature (C)",
        title=f"Seasonal temperature over land ({settings.output_tag})",
        fig_path=plot_dir / "seasonal_temperature_barplot_land.png",
        include_obs=True,
    )
    _plot_grouped_bars(
        df=df,
        column="bias",
        ylabel="Bias (C)",
        title=f"Seasonal bias over land ({settings.output_tag})",
        fig_path=plot_dir / "seasonal_bias_barplot_land.png",
        include_obs=False,
    )
    _plot_grouped_bars(
        df=df,
        column="rmse_clim",
        ylabel="RMSE (C)",
        title=f"Seasonal climatological RMSE over land ({settings.output_tag})",
        fig_path=plot_dir / "seasonal_rmse_barplot_land.png",
        include_obs=False,
    )


if __name__ == "__main__":
    main()
