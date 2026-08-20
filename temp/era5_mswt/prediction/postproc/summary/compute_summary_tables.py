from __future__ import annotations

import pandas as pd

from ..common import (
    PRED_CONFIG,
    get_metric_data_dir,
    load_prediction_settings,
)


SUMMARY_INPUTS = [
    ("climatology", "climatology_summary_{start}_{end}_{mode}.csv"),
    ("monthly_cycle", "monthly_cycle_summary_{start}_{end}_{mode}.csv"),
    ("seasonal_cycle", "seasonal_cycle_summary_{start}_{end}_{mode}.csv"),
    ("distributions", "distributions_summary_{start}_{end}_{mode}.csv"),
    ("bias", "bias_summary_{start}_{end}_{mode}.csv"),
    ("rmse", "rmse_summary_{start}_{end}_{mode}.csv"),
    ("correlation", "corr_d_summary_{start}_{end}_{mode}.csv"),
]


def main() -> None:
    settings = load_prediction_settings()
    output_dir = get_metric_data_dir(settings, "summary")

    print("=== Computing global post-processing summary ===")
    print(f"Prediction config : {PRED_CONFIG}")
    print(f"Selected period   : {settings.selected_start} -> {settings.selected_end}")
    print(f"Mode              : {settings.output_tag}")
    print(f"Evaluation domain : {settings.eval_domain}")
    print(f"Display domain    : {settings.display_domain}")
    print(f"Output dir        : {output_dir}")

    frames = []
    for axis, pattern in SUMMARY_INPUTS:
        data_dir = get_metric_data_dir(settings, axis, create=False)
        path = data_dir / pattern.format(
            start=settings.selected_start,
            end=settings.selected_end,
            mode=settings.output_tag,
        )
        if not path.exists():
            print(f"[WARNING] Missing summary CSV for {axis}: {path}")
            continue

        df = pd.read_csv(path)
        df.insert(0, "analysis_axis", axis)
        df.insert(1, "source_csv", str(path))
        frames.append(df)
        print(f"  Loaded {axis}: {path}")

    if frames:
        global_df = pd.concat(frames, ignore_index=True, sort=False)
    else:
        global_df = pd.DataFrame(columns=["analysis_axis", "source_csv"])
        print("[WARNING] No input summary CSV found; writing an empty global summary.")

    out_csv = output_dir / (
        f"global_postproc_summary_{settings.selected_start}_{settings.selected_end}_{settings.output_tag}.csv"
    )
    global_df.to_csv(out_csv, index=False)
    print(f"[SUCCESS] Global summary CSV saved to: {out_csv}")


if __name__ == "__main__":
    main()
