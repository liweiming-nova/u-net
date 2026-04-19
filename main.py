import argparse

from src.plot_metrics import plot_comparison_curves_from_json, plot_training_curves_from_json
from src.predict import predict
from src.prepare_data import prepare_data
from src.train import train
from src.utils import load_yaml


def parse_args():
    parser = argparse.ArgumentParser(description="U-Net retinal vessel segmentation")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--mode", type=str, choices=["train", "predict", "prepare-data", "plot-metrics", "plot-compare"], default="train")
    parser.add_argument("--metrics-path", type=str, default=None)
    parser.add_argument("--baseline-metrics-path", type=str, default=None)
    parser.add_argument("--improved-metrics-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--baseline-label", type=str, default="Baseline U-Net")
    parser.add_argument("--improved-label", type=str, default="Improved U-Net")
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_yaml(args.config)

    if args.mode == "train":
        train(cfg, resume_path=args.resume_path, resume=not args.no_resume)
    elif args.mode == "predict":
        predict(args.config)
    elif args.mode == "plot-metrics":
        metrics_path = args.metrics_path or f'{cfg["train"]["save_dir"]}/metrics.json'
        curve_paths = plot_training_curves_from_json(metrics_path, args.output_dir or cfg["train"]["save_dir"])
        print(curve_paths)
    elif args.mode == "plot-compare":
        baseline_metrics_path = args.baseline_metrics_path
        improved_metrics_path = args.improved_metrics_path or args.metrics_path or f'{cfg["train"]["save_dir"]}/metrics.json'
        if not baseline_metrics_path:
            raise ValueError("--baseline-metrics-path is required for plot-compare mode")
        curve_paths = plot_comparison_curves_from_json(
            baseline_metrics_path=baseline_metrics_path,
            improved_metrics_path=improved_metrics_path,
            output_dir=args.output_dir or "plots_compare",
            baseline_label=args.baseline_label,
            improved_label=args.improved_label,
        )
        print(curve_paths)
    else:
        prepare_data(args.config)
