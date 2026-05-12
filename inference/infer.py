#!/usr/bin/env python
import argparse
import os
from pathlib import Path

import numpy as np

from inference_utils import (
    check_metadata_compatibility,
    choose_device,
    find_checkpoint,
    load_model_from_checkpoint,
    load_training_config,
    load_yaml,
    plot_prediction_summary,
    prepare_strain_for_inference,
    read_downloader_hdf5,
    triggers_for_threshold_and_width,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run basic AttenGW inference on one downloader HDF5 file."
    )

    parser.add_argument("--config", type=str, required=True,
                        help="Inference YAML config.")

    # Common path overrides
    parser.add_argument("--checkpoint_run_dir", type=str, default=None,
                        help="Override checkpoint.run_dir from the YAML.")
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="Override checkpoint.ckpt_path from the YAML.")
    parser.add_argument("--input_file", type=str, default=None,
                        help="Override input.file from the YAML.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output.output_dir from the YAML.")

    # Common inference overrides
    parser.add_argument("--thresholds", type=float, nargs="+", default=None,
                        help="Override inference.thresholds from the YAML.")
    parser.add_argument("--min_width_samples", type=int, nargs="+", default=None,
                        help="Override inference.min_width_samples from the YAML.")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override inference.batch_size from the YAML.")
    parser.add_argument("--device", type=str, default=None,
                        help="Override inference.device from the YAML, e.g. auto/cpu/cuda.")

    # Output behavior
    parser.add_argument("--make_plot", action="store_true",
                        help="Force output.make_plot=True.")
    parser.add_argument("--no_plot", action="store_true",
                        help="Force output.make_plot=False.")
    parser.add_argument("--save_predictions", action="store_true",
                        help="Force output.save_predictions=True.")

    return parser.parse_args()


def print_file_summary(input_file, attrs, prep_info):
    print("\nInput file")
    print(f"  path: {input_file}")
    print(f"  mode: {attrs.get('mode', 'unknown')}")
    print(f"  whitened: {attrs.get('whiten', 'unknown')}")
    print(f"  gps_start: {attrs.get('gps_start', 'unknown')}")
    print(f"  gps_end: {attrs.get('gps_end', 'unknown')}")
    print(f"  effective_gps_start_after_preprocessing: {prep_info['effective_gps_start']}")
    print(f"  sample_rate: {prep_info['sample_rate']} Hz")
    print(f"  trim_samples: {prep_info['trim_samples']}")

def format_all_results_text(input_file, run_dir, ckpt_path, attrs, prep_info, all_results):
    lines = []

    lines.append("=" * 100)
    lines.append("ATTENGW INFERENCE SUMMARY")
    lines.append("=" * 100)
    lines.append(f"Input file:   {input_file}")
    lines.append(f"Training run: {run_dir}")
    lines.append(f"Checkpoint:   {ckpt_path}")
    lines.append("")
    lines.append("File metadata:")
    lines.append(f"  mode: {attrs.get('mode', 'unknown')}")
    lines.append(f"  whitened: {attrs.get('whiten', 'unknown')}")
    lines.append(f"  gps_start: {attrs.get('gps_start', 'unknown')}")
    lines.append(f"  gps_end: {attrs.get('gps_end', 'unknown')}")
    lines.append(f"  effective_gps_start_after_preprocessing: {prep_info['effective_gps_start']}")
    lines.append(f"  sample_rate: {prep_info['sample_rate']} Hz")
    lines.append(f"  trim_samples: {prep_info['trim_samples']}")
    lines.append("")
    lines.append("-" * 100)
    lines.append("TRIGGER COUNT SUMMARY")
    lines.append("-" * 100)
    lines.append(
        "Each cell shows: number of triggers; first up to 3 peak samples."
    )
    lines.append("")

    thresholds = sorted({k[0] for k in all_results.keys()})
    widths = sorted({k[1] for k in all_results.keys()})

    # Header
    col_width = 26
    header = f"{'threshold':>12}"
    for width in widths:
        header += f" | {'width=' + str(width):^{col_width}}"
    lines.append(header)
    lines.append("-" * len(header))

    # Rows
    for threshold in thresholds:
        row = f"{threshold:>12}"
        for width in widths:
            triggers = all_results.get((threshold, width), [])
            first_samples = [str(t["peak_sample"]) for t in triggers[:3]]
            if first_samples:
                cell = f"n={len(triggers)}; samples={','.join(first_samples)}"
            else:
                cell = "n=0"
            row += f" | {cell:<{col_width}}"
        lines.append(row)

    lines.append("")
    lines.append("=" * 100)
    lines.append("DETAILED TRIGGERS")
    lines.append("=" * 100)

    for threshold in thresholds:
        for width in widths:
            triggers = all_results.get((threshold, width), [])
            lines.append("")
            lines.append("-" * 100)
            lines.append(f"Threshold: {threshold}")
            lines.append(f"min_width_samples: {width}")
            lines.append(f"Triggers found: {len(triggers)}")
            lines.append("-" * 100)

            if not triggers:
                lines.append("No triggers found.")
                continue

            for i, trig in enumerate(triggers, start=1):
                lines.append("")
                lines.append(f"Trigger {i}:")
                lines.append(f"  peak score: {trig['peak_score']:.6f}")
                lines.append(f"  peak sample: {trig['peak_sample']}")
                lines.append(f"  time from processed file start: {trig['peak_time_s']:.6f} s")
                lines.append(f"  GPS time: {trig['peak_gps']:.6f}")
                lines.append(f"  UTC time: {trig['peak_utc']}")
                lines.append(
                    f"  trigger interval: {trig['left_time_s']:.6f} – "
                    f"{trig['right_time_s']:.6f} s after processed file start"
                )
                lines.append(f"  offset stream: {trig['offset']} samples")
                lines.append(f"  mean cut used: {trig['mean_cut']:.6f}")

    return "\n".join(lines) + "\n"

def print_triggers(threshold, width, triggers):
    print("\n" + "=" * 80)
    print(f"Threshold {threshold}, min_width_samples {width}")
    print(f"Triggers found: {len(triggers)}")

    if not triggers:
        return

    for i, trig in enumerate(triggers, start=1):
        print(f"\nTrigger {i}")
        print(f"  peak score: {trig['peak_score']:.6f}")
        print(f"  peak sample: {trig['peak_sample']}")
        print(f"  time from processed file start: {trig['peak_time_s']:.6f} s")
        print(f"  GPS time: {trig['peak_gps']:.6f}")
        print(f"  UTC time: {trig['peak_utc']}")
        print(
            "  trigger interval: "
            f"{trig['left_time_s']:.6f} – {trig['right_time_s']:.6f} s "
            "after processed file start"
        )
        print(f"  offset stream: {trig['offset']} samples")
        print(f"  mean cut used: {trig['mean_cut']:.6f}")


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    
    if args.checkpoint_run_dir is not None:
        cfg["checkpoint"]["run_dir"] = args.checkpoint_run_dir
    
    if args.ckpt_path is not None:
        cfg["checkpoint"]["ckpt_path"] = args.ckpt_path
    
    if args.input_file is not None:
        cfg["input"]["file"] = args.input_file
    
    if args.output_dir is not None:
        cfg["output"]["output_dir"] = args.output_dir
    
    if args.thresholds is not None:
        cfg["inference"]["thresholds"] = args.thresholds
    
    if args.min_width_samples is not None:
        cfg["inference"]["min_width_samples"] = args.min_width_samples
    
    if args.batch_size is not None:
        cfg["inference"]["batch_size"] = args.batch_size
    
    if args.device is not None:
        cfg["inference"]["device"] = args.device
    
    if args.make_plot:
        cfg["output"]["make_plot"] = True
    
    if args.no_plot:
        cfg["output"]["make_plot"] = False
    
    if args.save_predictions:
        cfg["output"]["save_predictions"] = True

    run_dir = cfg["checkpoint"]["run_dir"]
    ckpt_path = find_checkpoint(run_dir, cfg["checkpoint"].get("ckpt_path"))
    training_config = load_training_config(run_dir)

    input_file = cfg["input"]["file"]
    output_dir = cfg["output"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    device = choose_device(cfg["inference"].get("device", "auto"))
    print(f"Using device: {device}")

    model = load_model_from_checkpoint(ckpt_path, device)

    data, attrs = read_downloader_hdf5(input_file)
    check_metadata_compatibility(training_config, attrs, cfg.get("sanity_checks", {}))

    strain_L1, strain_H1, prep_info = prepare_strain_for_inference(
        data=data,
        attrs=attrs,
        training_config=training_config,
        inference_config=cfg,
    )

    print_file_summary(input_file, attrs, prep_info)
    print("\nModel")
    print(f"  training run: {run_dir}")
    print(f"  checkpoint: {ckpt_path}")
    print(f"  prepared samples: {len(strain_L1)}")

    inf = cfg["inference"]
    thresholds = inf["thresholds"]
    widths = inf["min_width_samples"]

    all_results = {}

    for threshold in thresholds:
        for width in widths:
            triggers, pred_by_offset = triggers_for_threshold_and_width(
                model=model,
                strain_L1=strain_L1,
                strain_H1=strain_H1,
                sample_rate=float(prep_info["sample_rate"]),
                effective_gps_start=float(prep_info["effective_gps_start"]),
                segment_length=int(inf["segment_length"]),
                stride=int(inf["stride"]),
                offsets=inf["offsets"],
                batch_size=int(inf["batch_size"]),
                threshold=float(threshold),
                min_width_samples=int(width),
                mean_margin=float(inf.get("mean_margin", 0.05)),
                mean_cap=float(inf.get("mean_cap", 0.95)),
                merge_tolerance_s=float(inf.get("merge_tolerance_s", 0.25)),
                device=device,
            )

            key = f"thr_{threshold}_width_{width}"
            all_results[(float(threshold), int(width))] = triggers
            print_triggers(threshold, width, triggers)
            

            if cfg["output"].get("make_plot", False):
                stem = Path(input_file).stem
                plots_dir = os.path.join(output_dir, "plots")
                os.makedirs(plots_dir, exist_ok=True)
            
                plot_path = os.path.join(plots_dir, f"{stem}_{key}_prediction.png")
                plot_prediction_summary(
                    strain_L1=strain_L1,
                    strain_H1=strain_H1,
                    pred_by_offset=pred_by_offset,
                    triggers=triggers,
                    sample_rate=float(prep_info["sample_rate"]),
                    output_path=plot_path,
                    title=f"{stem}: threshold={threshold}, width={width}",
                )
                print(f"Saved plot: {plot_path}")

            if cfg["output"].get("save_predictions", False):
                stem = Path(input_file).stem
                pred_path = os.path.join(output_dir, f"{stem}_{key}_predictions.npz")
                np.savez_compressed(
                    pred_path,
                    **{f"offset_{k}": v for k, v in pred_by_offset.items()},
                )
                print(f"Saved predictions: {pred_path}")
                
    stem = Path(input_file).stem
    summary_txt = os.path.join(output_dir, f"{stem}_inference_summary.txt")

    with open(summary_txt, "w") as f:
        f.write(
            format_all_results_text(
                input_file=input_file,
                run_dir=run_dir,
                ckpt_path=ckpt_path,
                attrs=attrs,
                prep_info=prep_info,
                all_results=all_results,
            )
        )

    print(f"\nSaved combined inference summary: {summary_txt}")

    print("\nInference finished.")


if __name__ == "__main__":
    main()