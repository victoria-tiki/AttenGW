#!/usr/bin/env python
import argparse
import os
import glob
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
    format_noise_metrics_text,
    resolve_training_noise_files,
    build_training_noise_reference,
    compute_noise_domain_metrics,
    triggers_for_threshold_and_width,
)



def parse_args():
    parser = argparse.ArgumentParser(
        description="Run basic AttenGW inference on one or more downloader HDF5 files."
    )

    parser.add_argument("--config", type=str, required=True,
                        help="Inference YAML config.")

    # Common path overrides
    parser.add_argument("--checkpoint_run_dir", type=str, default=None,
                        help="Override checkpoint.run_dir from the YAML.")
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="Override checkpoint.ckpt_path from the YAML.")
    parser.add_argument("--input_path", type=str, default=None,
                        help="Override input.path from the YAML. Can be one file or a glob.")
    parser.add_argument("--input_file", type=str, default=None,
                        help="Backward-compatible alias for --input_path.")
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
    parser.add_argument("--plot_only_if_triggers", action="store_true",
                        help="Only save plots for threshold/width combinations with triggers.")
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
    print(f"  checkpoint_noise_is_whitened: {prep_info.get('checkpoint_noise_is_whitened', 'unknown')}")
    print(f"  whitened_during_inference: {prep_info.get('whitened_during_inference', 'unknown')}")
    print(f"  mean_subtraction: {prep_info.get('mean_subtraction', 'unknown')}")
    print(f"  std_normalization: {prep_info.get('std_normalization', 'unknown')}")

def resolve_input_files(input_path):
    """
    Resolve one input path or glob pattern into a sorted list of files.
    """
    matches = sorted(glob.glob(input_path))
    if matches:
        return matches

    if os.path.exists(input_path):
        return [input_path]

    raise FileNotFoundError(f"No input files found for: {input_path}")


def format_all_results_text(input_file, run_dir, ckpt_path, attrs, prep_info, all_results, noise_metrics=None,):
    """
    Human-readable text output for one input file.

    Summary comes first as a compact threshold/width grid, followed by detailed
    trigger information.
    """
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
    lines.append(f"  checkpoint_noise_is_whitened: {prep_info.get('checkpoint_noise_is_whitened', 'unknown')}")
    lines.append(f"  whitened_during_inference: {prep_info.get('whitened_during_inference', 'unknown')}")
    lines.append(f"  mean_subtraction: {prep_info.get('mean_subtraction', 'unknown')}")
    lines.append(f"  std_normalization: {prep_info.get('std_normalization', 'unknown')}")
    lines.append("")
    lines.extend(format_noise_metrics_text(noise_metrics))
    lines.append("")
    lines.append("-" * 100)
    lines.append("TRIGGER COUNT SUMMARY")
    lines.append("-" * 100)
    lines.append("Each cell shows: number of triggers; first up to 3 peak samples.")
    lines.append("")

    thresholds = sorted({key[0] for key in all_results.keys()})
    widths = sorted({key[1] for key in all_results.keys()})

    col_width = 28
    header = f"{'threshold':>12}"
    for width in widths:
        header += f" | {'width=' + str(width):^{col_width}}"
    lines.append(header)
    lines.append("-" * len(header))

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


def format_all_files_summary_text(file_summaries):
    """
    Compact summary across all input files.
    """
    lines = []
    lines.append("=" * 100)
    lines.append("ATTENGW MULTI-FILE INFERENCE SUMMARY")
    lines.append("=" * 100)
    lines.append("Each row is one input file and one threshold/width setting.")
    lines.append("first_samples lists the first up to 3 trigger peak samples.")
    lines.append("")

    header = (
        f"{'file':<45} | {'threshold':>9} | {'width':>6} | "
        f"{'n_trig':>6} | {'first_samples':<30}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for item in file_summaries:
        file_name = Path(item["input_file"]).name
        if len(file_name) > 45:
            file_name = "..." + file_name[-42:]

        first_samples = ",".join(str(x) for x in item["first_samples"])
        lines.append(
            f"{file_name:<45} | "
            f"{item['threshold']:>9} | "
            f"{item['width']:>6} | "
            f"{item['n_triggers']:>6} | "
            f"{first_samples:<30}"
        )

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

    # Apply CLI overrides. YAML remains the default source of truth.
    if args.checkpoint_run_dir is not None:
        cfg["checkpoint"]["run_dir"] = args.checkpoint_run_dir

    if args.ckpt_path is not None:
        cfg["checkpoint"]["ckpt_path"] = args.ckpt_path

    input_override = args.input_path or args.input_file
    if input_override is not None:
        cfg["input"]["path"] = input_override

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

    if args.plot_only_if_triggers:
        cfg["output"]["plot_only_if_triggers"] = True

    if args.save_predictions:
        cfg["output"]["save_predictions"] = True

    run_dir = cfg["checkpoint"]["run_dir"]
    ckpt_path = find_checkpoint(run_dir, cfg["checkpoint"].get("ckpt_path"))
    training_config = load_training_config(run_dir)

    # Backward-compatible: allow either input.path or old input.file.
    input_cfg = cfg["input"]
    input_path = input_cfg.get("path") or input_cfg.get("file")
    if input_path is None:
        raise ValueError("Inference config must define input.path or input.file.")

    input_files = resolve_input_files(input_path)
    
    # Optional noise-domain reference, used only for summary diagnostics.
    noise_similarity_cfg = cfg.get("noise_similarity", {})
    training_noise_files = resolve_training_noise_files(
        noise_similarity_cfg.get("training_noise_path")
    )
    noise_reference = None
    if training_noise_files:
        print(f"Resolved {len(training_noise_files)} training noise file(s) for similarity metrics.")
        noise_reference = build_training_noise_reference(
            training_noise_files=training_noise_files,
            training_config=training_config,
            similarity_config=noise_similarity_cfg,
        )
    else:
        print(
            "No training noise files configured for similarity metrics. "
            "Set noise_similarity.training_noise_path to enable them."
        )

    output_dir = cfg["output"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    device = choose_device(cfg["inference"].get("device", "auto"))
    print(f"Using device: {device}")
    print(f"Resolved {len(input_files)} input file(s).")

    model = load_model_from_checkpoint(ckpt_path=ckpt_path,device=device,training_config=training_config)

    inf = cfg["inference"]
    thresholds = inf["thresholds"]
    widths = inf["min_width_samples"]

    multi_file_summaries = []

    for file_idx, input_file in enumerate(input_files, start=1):
        print("\n" + "#" * 100)
        print(f"Processing file {file_idx}/{len(input_files)}: {input_file}")
        print("#" * 100)

        data, attrs = read_downloader_hdf5(input_file)
        check_metadata_compatibility(training_config, attrs, cfg.get("sanity_checks", {}))

        strain_L1, strain_H1, prep_info = prepare_strain_for_inference(
            data=data,
            attrs=attrs,
            training_config=training_config,
            inference_config=cfg,
        )
        
        noise_metrics = compute_noise_domain_metrics(
            eval_noise_file=input_file,
            eval_strain_L1=strain_L1,
            eval_strain_H1=strain_H1,
            eval_attrs=attrs,
            training_reference=noise_reference,
            training_config=training_config,
            similarity_config=noise_similarity_cfg,
            prep_info=prep_info,
        )

        print_file_summary(input_file, attrs, prep_info)
        print("\nModel")
        print(f"  training run: {run_dir}")
        print(f"  checkpoint: {ckpt_path}")
        print(f"  prepared samples: {len(strain_L1)}")

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

                multi_file_summaries.append(
                    {
                        "input_file": input_file,
                        "threshold": float(threshold),
                        "width": int(width),
                        "n_triggers": len(triggers),
                        "first_samples": [t["peak_sample"] for t in triggers[:3]],
                    }
                )

                should_plot = cfg["output"].get("make_plot", False)
                if cfg["output"].get("plot_only_if_triggers", False) and not triggers:
                    should_plot = False

                if should_plot:
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
                    noise_metrics=noise_metrics,
                )
            )

        print(f"\nSaved inference summary: {summary_txt}")

    # If multiple files were processed, also save one compact cross-file summary.
    if len(input_files) > 1:
        all_files_txt = os.path.join(output_dir, "all_files_inference_summary.txt")
        with open(all_files_txt, "w") as f:
            f.write(format_all_files_summary_text(multi_file_summaries))
        print(f"\nSaved multi-file summary: {all_files_txt}")

    print("\nInference finished.")

if __name__ == "__main__":
    main()
