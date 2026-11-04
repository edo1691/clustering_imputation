"""Command-line entry point for full and individual workflow stages."""

import argparse
import json
import sys
from typing import Optional, Sequence

from .config import PIPELINE_STEPS, load_config
from . import pipeline, stages


_STAGE_FUNCTIONS = {
    "generate_dataset": stages.generate_dataset,
    "split_dataset": stages.split_dataset,
    "fit_preprocessor": stages.fit_preprocessor,
    "train_model": stages.train_model,
    "build_prototypes": stages.build_prototypes,
    "transform_test": stages.transform_test,
    "evaluate": stages.evaluate,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate synthetic data and build constrained cluster representations."
    )
    parser.add_argument("--config", required=True, help="Path to YAML configuration.")
    parser.add_argument("--step", choices=PIPELINE_STEPS, help="Run one stage only.")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        results = (
            [_STAGE_FUNCTIONS[args.step](config)]
            if args.step
            else pipeline.run(config)
        )
        for result in results:
            print(
                json.dumps(
                    {"stage": result.stage, "summary": result.summary},
                    sort_keys=True,
                    default=str,
                )
            )
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        print("Error: {}".format(exc), file=sys.stderr)
        return 1
    return 0
