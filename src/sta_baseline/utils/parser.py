#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Argument parser functions."""

import argparse
import sys

from fvcore.common.config import CfgNode

from sta_baseline.config.defaults import get_cfg


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the SlowFast training pipeline."""
    parser = argparse.ArgumentParser(description="Provide SlowFast video training and testing pipeline.")
    parser.add_argument("--job_name", default="", type=str)
    parser.add_argument("--on_cluster", default=False, action="store_true")
    parser.add_argument("--working_directory", default="", type=str)
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument(
        "--shard_id",
        help="The shard id of current node, Starts from 0 to num_shards - 1",
        default=0,
        type=int,
    )
    parser.add_argument("--num_shards", help="Number of shards using by the job", default=1, type=int)
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default="configs/Kinetics/SLOWFAST_4x16_R50.yaml",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="See ego4d/config/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> CfgNode:
    """Given the arguemnts, load and initialize the configs.

    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    # Setup cfg.
    cfg = get_cfg()
    # Load config from cfg.
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    # Inherit parameters from args.
    if hasattr(args, "num_shards"):
        cfg.NUM_SHARDS = args.num_shards
    if hasattr(args, "rng_seed"):
        cfg.RNG_SEED = args.rng_seed
    if hasattr(args, "output_dir"):
        cfg.OUTPUT_DIR = args.output_dir
    if hasattr(args, "fast_dev_run"):
        cfg.FAST_DEV_RUN = args.fast_dev_run

    return cfg
