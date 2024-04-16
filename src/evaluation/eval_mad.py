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
import trimesh
import torch
import sys, os
sys.path.append(os.getcwd())

from src.naruto.cfg_loader import load_cfg
from src.slam.coslam.coslam import CoSLAMNaruto as CoSLAM
from src.utils.general_utils import InfoPrinter, update_results_file

def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh

##################################################
### argparse
##################################################
parser = argparse.ArgumentParser(
        description="Arguments to the active sim slam."
    )
parser.add_argument("--cfg", type=str,
                    help="experiement config file path")
parser.add_argument("--ckpt", type=str,
                    help="ckpt path")
parser.add_argument("--gt_mesh", type=str,
                    help="gt mesh path")
parser.add_argument("--result_txt", type=str,
                    help="result text file")
args = parser.parse_args()

gt_meshfile = args.gt_mesh

##################################################
### load model
##################################################
info_printer = InfoPrinter("NARUTO", 0)
main_cfg = load_cfg(args)
slam = CoSLAM(main_cfg, info_printer) 
slam.load_ckpt(args.ckpt)

##################################################
### Sample GT mesh points
##################################################
mesh_gt = trimesh.load(gt_meshfile, process=False)
if gt_meshfile.endswith(".obj"):
    mesh_gt = as_mesh(mesh_gt)
gt_pc = trimesh.sample.sample_surface(mesh_gt, 200000, seed=0)[0] # N,3
gt_pc = torch.from_numpy(gt_pc).unsqueeze(1).cuda().float() # N, 1, 3
pred_sdf = slam.predict_sdf(gt_pc)
mad = pred_sdf.abs().mean().item() * 10 # unit: cm

##################################################
### print and save result
##################################################
print(f"MAD : {mad:.2f} cm")
result = {"MAD": mad}
update_results_file(result, args.result_txt)
