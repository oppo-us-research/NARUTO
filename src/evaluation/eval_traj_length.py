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
import os
import sys
import torch
from tqdm import tqdm
sys.path.append(os.getcwd())

from src.utils.general_utils import update_results_file


def argument_parsing() -> argparse.Namespace:
    """parse arguments

    Returns:
        args: arguments
        
    """
    parser = argparse.ArgumentParser(
            description="Arguments to calculate trajectory from the poses stored in checkpoint."
        )
    parser.add_argument("--ckpt", type=str, help="Checkpoint path")
    parser.add_argument("--result_txt", type=str, help="result txt")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = argument_parsing()

    ##################################################
    ### load checkpoint
    ##################################################
    print(f"==> loading checkpoint: [{args.ckpt}]")
    ckpt_path = args.ckpt
    ckpt = torch.load(ckpt_path)

    ##################################################
    ### read pose and calculate trajectory and save in result
    ##################################################
    poses_tensor = ckpt['pose']
    traj_len = 0
    
    for i in tqdm(range(len(poses_tensor)), desc="==> Calculating traj. length: "):
        if i == 0:
            cur_pose = poses_tensor[i]
            continue
        rel_pose = torch.inverse(poses_tensor[i]) @ cur_pose
        traj_len += torch.norm(rel_pose[:3, 3]).item()
        cur_pose = poses_tensor[i]
    
    print(f"==> Total trajectory length: {traj_len:.2f}")
    
    ##################################################
    ### write/update result to the result txt
    ##################################################
    result = {"traj_len(m)": traj_len}
    update_results_file(result, args.result_txt)
    