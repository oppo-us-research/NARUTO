"""
MIT License

Copyright (c) 2024 OPPO

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import argparse
import mmengine

def override_cfg(
        args: argparse.Namespace,
        cfg : mmengine.Config
    ) -> mmengine.Config:
    """override configuration

    Args:
        args: arguments
        cfg : configuration

    Returns:
        cfg : updated configuration
    """
    if hasattr(args, "seed") and args.seed is not None:
        ### random seed ###
        cfg.general.seed = args.seed

    if hasattr(args, "result_dir") and args.result_dir is not None:
        ### output/result directory ###
        cfg.dirs.result_dir = args.result_dir

    if hasattr(args, "enable_vis") and args.enable_vis is not None:
        ### output/result directory ###
        enable_vis = args.enable_vis == 1
        cfg.visualizer.vis_rgbd = enable_vis
    return cfg


def argument_parsing() -> argparse.Namespace:
    """parse arguments

    Returns:
        args: arguments
        
    """
    parser = argparse.ArgumentParser(
            description="Arguments to run NARUTO."
        )
    parser.add_argument("--cfg", type=str, default="configs/default.py",
                        help="NARUTO config")
    parser.add_argument("--result_dir", type=str, default=None, 
                        help="result directory")
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed; also used as the initial pose idx for Replica")
    parser.add_argument("--enable_vis", type=int, default=None,
                        help="enable visualization. 1: True, 0: False")
    args = parser.parse_args()
    return args


def load_cfg(args: argparse.Namespace) -> mmengine.Config:
    """argument parsing and load configuration

    Args:
        args: arguments

    Returns:
        cfg : configuration

    """
    cfg = mmengine.Config.fromfile(args.cfg)
    cfg = override_cfg(args, cfg)
    return cfg
