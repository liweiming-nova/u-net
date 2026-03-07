import argparse

from src.predict import predict
from src.train import train
from src.utils import load_yaml


def parse_args():
    parser = argparse.ArgumentParser(description="U-Net retinal vessel segmentation")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--mode", type=str, choices=["train", "predict"], default="train")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_yaml(args.config)

    if args.mode == "train":
        train(cfg)
    else:
        predict(args.config)
