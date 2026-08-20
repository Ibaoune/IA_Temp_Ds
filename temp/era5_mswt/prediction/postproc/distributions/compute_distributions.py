from __future__ import annotations

import gc
import numpy as np
import pandas as pd

from ..common import (
    PRED_CONFIG,
    get_metric_data_dir,
    load_aligned_comparison_data,
    load_prediction_settings,
    valid_pixel_count,
)
from ...backend_paths import import_postproc_module

flatten_valid = import_postproc_module("map_utils").flatten_valid


QUANTILES = {
    "q01": 0.01,
    "q05": 0.05,
    "q25": 0.25,
    "q50": 0.50,
    "q75": 0.75,
    "q95": 0.95,
    "q99": 0.99,
}
N_BINS = 60


def _distribution_stats(values: np.ndarray) -> dict:
    if values.size == 0:
        stats = {
            "count": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
        }
        stats.update({name: np.nan for name in QUANTILES})
        return stats

    stats = {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }
    stats.update({name: float(np.quantile(values, q)) for name, q in QUANTILES.items()})
    return stats


def _common_edges(summary_rows: list[dict]) -> np.ndarray:
    mins = [row["min"] for row in summary_rows if np.isfinite(row["min"])]
    maxs = [row["max"] for row in summary_rows if np.isfinite(row["max"])]
    if not mins or not maxs:
        return np.linspace(0.0, 1.0, N_BINS + 1)

    vmin = float(np.min(mins))
    vmax = float(np.max(maxs))
    if vmax <= vmin:
        pad = 0.5 if vmin == 0 else abs(vmin) * 0.01
        return np.linspace(vmin - pad, vmax + pad, N_BINS + 1)

    return np.linspace(vmin, vmax, N_BINS + 1)


def main() -> None:
    settings = load_prediction_settings()
    data_dir = get_metric_data_dir(settings, "distributions")

    print("=== Computing daily temperature distributions ===")
    print(f"Prediction config : {PRED_CONFIG}")
    print(f"Reference file    : {settings.reference_path}")
    print(f"Prediction dir    : {settings.prediction_dir}")
    print(f"Selected period   : {settings.selected_start} -> {settings.selected_end}")
    print(f"Mode              : {settings.output_tag}")
    print(f"Prediction mode   : {settings.prediction_tag}")
    print(f"Evaluation domain : {settings.eval_domain}")
    print(f"Display domain    : {settings.display_domain}")
    print(f"Output dir        : {data_dir}")

    if not settings.reference_path.exists():
        raise FileNotFoundError(f"Reference file not found: {settings.reference_path}")

    sources = load_aligned_comparison_data(settings)
    print("Models found      :", [source["label"] for source in sources])

    summary_rows = []

    for source in sources:
        label = source["label"]
        initial_pixels = valid_pixel_count(source.get("initial_spatial_mask", source["spatial_mask"]))
        common_pixels = valid_pixel_count(source["spatial_mask"])
        values = flatten_valid(source["data"].values)

        row = {
            "model": label,
            "start_date": settings.selected_start,
            "end_date": settings.selected_end,
            "eval_domain": settings.eval_domain,
            "display_domain": settings.display_domain,
            "initial_spatial_pixels": initial_pixels,
            "common_valid_pixels": common_pixels,
            **_distribution_stats(values),
        }
        summary_rows.append(row)

        print(
            f"  {label}: time={source['data'].sizes.get('time', 0)}, "
            f"initial_pixels={initial_pixels}, common_pixels={common_pixels}, values={values.size}"
        )
        del values
        gc.collect()

    summary_df = pd.DataFrame(summary_rows)
    out_summary = data_dir / (
        f"distributions_summary_{settings.selected_start}_{settings.selected_end}_{settings.output_tag}.csv"
    )
    summary_df.to_csv(out_summary, index=False)
    print(f"[SUCCESS] Summary CSV saved to: {out_summary}")

    edges = _common_edges(summary_rows)
    hist_rows = []
    for source in sources:
        label = source["label"]
        values = flatten_valid(source["data"].values)
        counts, _ = np.histogram(values, bins=edges)
        widths = np.diff(edges)
        total = counts.sum()
        densities = counts / (total * widths) if total > 0 else np.zeros_like(counts, dtype=float)

        for left, right, count, density in zip(edges[:-1], edges[1:], counts, densities):
            hist_rows.append(
                {
                    "model": label,
                    "bin_left": float(left),
                    "bin_right": float(right),
                    "bin_center": float((left + right) / 2.0),
                    "count": int(count),
                    "density": float(density),
                    "eval_domain": settings.eval_domain,
                    "display_domain": settings.display_domain,
                }
            )
        del values
        gc.collect()

    hist_df = pd.DataFrame(hist_rows)
    out_hist = data_dir / (
        f"distributions_histogram_{settings.selected_start}_{settings.selected_end}_{settings.output_tag}.csv"
    )
    hist_df.to_csv(out_hist, index=False)
    print(f"[SUCCESS] Histogram CSV saved to: {out_hist}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
