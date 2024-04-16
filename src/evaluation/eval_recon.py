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


''' modified from third_parties/neural_slam_eval/eval_recon.py '''
import argparse
import os
import sys
import trimesh
sys.path.append(os.getcwd())

from src.utils.general_utils import update_results_file
from third_parties.neural_slam_eval.eval_recon import calc_3d_mesh_metric, get_align_transformation


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arguments to evaluate the reconstruction."
    )
    parser.add_argument("--rec_mesh", type=str,
                        help="reconstructed mesh file path")
    parser.add_argument("--gt_mesh", type=str,
                        help="ground truth mesh file path")
    parser.add_argument("--align",
                        action="store_true", help="Align meshes")
    parser.add_argument("--result_txt", type=str, help="result txt")
    args = parser.parse_args()

    ##################################################
    ### load mesh
    ##################################################
    mesh_gt = trimesh.load(args.gt_mesh, process=False)
    mesh_rec = trimesh.load(args.rec_mesh, process=False)

    if args.gt_mesh.endswith('obj'):
        mesh_gt = as_mesh(mesh_gt)
    
    ##################################################
    ### align reconstructed mesh to GT mesh
    ##################################################
    if args.align:
        transformation = get_align_transformation(args.rec_mesh, args.gt_mesh)
        mesh_rec = mesh_rec.apply_transform(transformation)

    ##################################################
    ### evaluate
    ##################################################
    eval_result = calc_3d_mesh_metric(mesh_gt, mesh_rec)
    
    ##################################################
    ### print and save result
    ##################################################
    print(eval_result)
    update_results_file(eval_result, args.result_txt)
