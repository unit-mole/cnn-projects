from __future__ import annotations
import argparse
from pathlib import Path
import _bootstrap  # noqa: F401
from src.experiment_runner import run_experiment

if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--config",type=Path); args=parser.parse_args(); run_experiment(args.config)
