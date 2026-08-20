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
QUANTILE_COLUMNS = ["q01", "q05", "q25", "q50", "q75", "q95", "q99"]
QUANTILE_LABELS = ["1", "5", "25", "50", "75", "95", "99"]


def _ordered_models(df: pd.DataFrame) -> list[str]:
    available = set(df["model"])
    return [model for model in MODEL_ORDER if model in available]


def _plot_histogram(hist_df: pd.DataFrame, fig_path: Path, mode: str) -> None:
    models = _ordered_models(hist_df)
    if not models:
        print(f"[WARNING] No models available for {fig_path.name}; skipping.")
        return

    fig, ax = plt.subplots(figsize=(10.5, 5.8), dpi=180)
    for model in models:
        sub = hist_df[hist_df["model"] == model].sort_values("bin_center")
        ax.plot(
            sub["bin_center"],
            sub["density"],
            linewidth=2.1,
            color=COLORS.get(model),
            label=model,
        )

    ax.set_xlabel("Daily temperature (C)")
    ax.set_ylabel("Density")
    ax.set_title(f"Daily temperature distribution over land ({mode})", fontweight="bold", pad=10)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.28)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[SUCCESS] Figure saved to: {fig_path}")


def _plot_boxplot(summary_df: pd.DataFrame, fig_path: Path, mode: str) -> None:
    rows = []
    labels = []
    for model in _ordered_models(summary_df):
        row = summary_df[summary_df["model"] == model].iloc[0]
        required = [row["min"], row["q25"], row["q50"], row["q75"], row["max"]]
        if not np.all(np.isfinite(required)):
            continue
        rows.append(
            {
                "label": model,
                "whislo": float(row["min"]),
                "q1": float(row["q25"]),
                "med": float(row["q50"]),
                "q3": float(row["q75"]),
                "whishi": float(row["max"]),
                "mean": float(row["mean"]),
                "fliers": [],
            }
        )
        labels.append(model)

    if not rows:
        print(f"[WARNING] No finite boxplot stats available for {fig_path.name}; skipping.")
        return

    fig, ax = plt.subplots(figsize=(10.0, 5.8), dpi=180)
    bp = ax.bxp(rows, showfliers=False, showmeans=True, patch_artist=True, widths=0.48)
    for patch, label in zip(bp["boxes"], labels):
        patch.set_facecolor(COLORS.get(label, "#8f8f8f"))
        patch.set_alpha(0.85)
        patch.set_edgecolor("#404040")
    for median in bp["medians"]:
        median.set_color("#202020")
        median.set_linewidth(1.7)
    for mean in bp["means"]:
        mean.set_marker("D")
        mean.set_markerfacecolor("white")
        mean.set_markeredgecolor("#202020")
        mean.set_markersize(5.0)

    ax.set_ylabel("Daily temperature (C)")
    ax.set_title(f"Daily temperature boxplot over land ({mode})", fontweight="bold", pad=10)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.28)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[SUCCESS] Figure saved to: {fig_path}")


def _plot_quantiles(summary_df: pd.DataFrame, fig_path: Path, mode: str) -> None:
    models = _ordered_models(summary_df)
    if not models:
        print(f"[WARNING] No models available for {fig_path.name}; skipping.")
        return

    x = np.arange(len(QUANTILE_COLUMNS))
    fig, ax = plt.subplots(figsize=(10.5, 5.8), dpi=180)
    for model in models:
        row = summary_df[summary_df["model"] == model].iloc[0]
        y = [row[col] for col in QUANTILE_COLUMNS]
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2.1,
            markersize=5.5,
            color=COLORS.get(model),
            label=model,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(QUANTILE_LABELS)
    ax.set_xlabel("Quantile (%)")
    ax.set_ylabel("Daily temperature (C)")
    ax.set_title(f"Daily temperature quantiles over land ({mode})", fontweight="bold", pad=10)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.28)
    ax.legend(frameon=False, ncols=2)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[SUCCESS] Figure saved to: {fig_path}")


def main() -> None:
    settings = load_prediction_settings()
    data_dir, plot_dir = get_metric_dirs(settings, "distributions")
    summary_path = data_dir / (
        f"distributions_summary_{settings.selected_start}_{settings.selected_end}_{settings.output_tag}.csv"
    )
    hist_path = data_dir / (
        f"distributions_histogram_{settings.selected_start}_{settings.selected_end}_{settings.output_tag}.csv"
    )

    print("=== Plotting daily temperature distributions ===")
    print(f"Prediction config : {PRED_CONFIG}")
    print(f"Selected period   : {settings.selected_start} -> {settings.selected_end}")
    print(f"Mode              : {settings.output_tag}")
    print(f"Evaluation domain : {settings.eval_domain}")
    print(f"Display domain    : {settings.display_domain}")
    print(f"Input summary CSV : {summary_path}")
    print(f"Input histogram   : {hist_path}")
    print(f"Output plot dir   : {plot_dir}")

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing distributions summary CSV: {summary_path}")
    if not hist_path.exists():
        raise FileNotFoundError(f"Missing distributions histogram CSV: {hist_path}")

    summary_df = pd.read_csv(summary_path)
    hist_df = pd.read_csv(hist_path)
    print("Models found      :", _ordered_models(summary_df))

    _plot_histogram(
        hist_df=hist_df,
        fig_path=plot_dir / "temperature_distribution_histogram_land.png",
        mode=settings.output_tag,
    )
    _plot_boxplot(
        summary_df=summary_df,
        fig_path=plot_dir / "temperature_boxplot_land.png",
        mode=settings.output_tag,
    )
    _plot_quantiles(
        summary_df=summary_df,
        fig_path=plot_dir / "temperature_quantiles_lineplot_land.png",
        mode=settings.output_tag,
    )


if __name__ == "__main__":
    main()
