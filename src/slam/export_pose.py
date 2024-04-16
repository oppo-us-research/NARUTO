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
import numpy as np
import torch


def argument_parsing() -> argparse.Namespace:
    """parse arguments

    Returns:
        args: arguments
        
    """
    parser = argparse.ArgumentParser(
            description="Arguments to export pose from checkpoint."
        )
    parser.add_argument("--ckpt", type=str, help="Checkpoint path")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = argument_parsing()

    ##################################################
    ### load checkpoint
    ##################################################
    ckpt_path = args.ckpt
    npy_path = ckpt_path + ".pose.npy"
    ckpt = torch.load(ckpt_path)

    ##################################################
    ### read pose and save as npy
    ##################################################
    poses_tensor = ckpt['pose']
    poses = []
    for i in range(len(poses_tensor)):
        poses.append(poses_tensor[i].detach().cpu().numpy())
    poses = np.stack(poses)
    np.save(npy_path, poses)
    