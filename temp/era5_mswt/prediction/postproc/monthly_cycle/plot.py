from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..common import PRED_CONFIG, get_metric_dirs, load_prediction_settings


MODEL_ORDER = ["MSWT", "ERA5_downscaled", "LMDZ_35", "LMDZ_250"]
MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
COLORS = {
    "MSWT": "#2f2f2f",
    "ERA5_downscaled": "#4c78a8",
    "LMDZ_35": "#59a14f",
    "LMDZ_250": "#e15759",
}


def _ordered_models(df: pd.DataFrame, include_obs: bool) -> list[str]:
    models = []
    for model in MODEL_ORDER:
        if model in set(df["model"]) and (include_obs or model != "MSWT"):
            models.append(model)
    return models


def _plot_cycle(
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

    fig, ax = plt.subplots(figsize=(10.5, 5.8), dpi=180)
    x = np.arange(1, 13)

    for model in models:
        sub = df[df["model"] == model].sort_values("month")
        ax.plot(
            sub["month"],
            sub[column],
            marker="o",
            linewidth=2.1,
            markersize=5.5,
            color=COLORS.get(model),
            label=model,
        )

    if column == "bias":
        ax.axhline(0.0, color="#2f2f2f", linestyle="--", linewidth=1.0, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(MONTH_LABELS)
    ax.set_xlabel("Month")
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
    data_dir, plot_dir = get_metric_dirs(settings, "monthly_cycle")
    csv_path = data_dir / (
        f"monthly_cycle_summary_{settings.selected_start}_{settings.selected_end}_{settings.output_tag}.csv"
    )

    print("=== Plotting monthly cycle diagnostics ===")
    print(f"Prediction config : {PRED_CONFIG}")
    print(f"Selected period   : {settings.selected_start} -> {settings.selected_end}")
    print(f"Mode              : {settings.output_tag}")
    print(f"Evaluation domain : {settings.eval_domain}")
    print(f"Display domain    : {settings.display_domain}")
    print(f"Input CSV         : {csv_path}")
    print(f"Output plot dir   : {plot_dir}")

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing monthly cycle CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    print("Models found      :", _ordered_models(df, include_obs=True))

    _plot_cycle(
        df=df,
        column="mean_temperature",
        ylabel="Temperature (C)",
        title=f"Monthly temperature cycle over land ({settings.output_tag})",
        fig_path=plot_dir / "monthly_temperature_cycle_land.png",
        include_obs=True,
    )
    _plot_cycle(
        df=df,
        column="bias",
        ylabel="Bias (C)",
        title=f"Monthly bias cycle over land ({settings.output_tag})",
        fig_path=plot_dir / "monthly_bias_cycle_land.png",
        include_obs=False,
    )
    _plot_cycle(
        df=df,
        column="rmse_clim",
        ylabel="RMSE (C)",
        title=f"Monthly climatological RMSE over land ({settings.output_tag})",
        fig_path=plot_dir / "monthly_rmse_cycle_land.png",
        include_obs=False,
    )


if __name__ == "__main__":
    main()
