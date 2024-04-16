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
from glob import glob
import numpy as np
import open3d as o3d
import os, sys
import time


sys.path.append(os.getcwd())
from src.visualization.o3d_utils import (
    create_camera_frustum, 
    save_camera_parameters, 
    load_camera_parameters_from_json, 
    create_dashed_line
    )


def argument_parsing() -> argparse.Namespace:
    """parse arguments

    Returns:
        args: arguments
        
    """
    parser = argparse.ArgumentParser(
            description="Arguments to visualize trajectory."
        )
    parser.add_argument("--mesh_file", type=str, default="", 
                        help="mesh file")
    parser.add_argument("--traj_dir", type=str, default="", 
                        help="trajectory pose dir")
    parser.add_argument("--cam_json", type=str, default=None, 
                        help="trajectory pose file")
    parser.add_argument("--out_dir", type=str, default=None, 
                        help="output directory to save rendered image")
    parser.add_argument("--with_interact", type=int, default=0, 
                        help="with interaction for visualization")
    args = parser.parse_args()
    return args

### arguments ###
args = argument_parsing()
mesh_file = args.mesh_file
window_hw = (1024, 1024)
cam_json = args.cam_json

mesh = o3d.io.read_triangle_mesh(mesh_file)
# camera_trajectory = np.load(traj_file)
pose_files = sorted(glob(os.path.join(args.traj_dir, "*.npy")))
camera_trajectory = np.stack([np.load(filepath) for filepath in pose_files])

### initialize window ###
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window(width=window_hw[1], height=window_hw[0])

### Add mesh ###
vis.add_geometry(mesh)

### set a view direction ###
if cam_json is not None:
    vis_cam_param = load_camera_parameters_from_json(cam_json)
view_control = vis.get_view_control()

### Add trajecotry ###
skip_step = 5
cam_traj_subset = camera_trajectory[::skip_step]
# cam_traj_subset = camera_trajectory[:10]

for step, pose in enumerate(cam_traj_subset):
    ##################################################
    ### Add Camera ###
    ##################################################
    intrinsic = np.array([[300, 0, 300],
                        [0, 300, 300],
                        [0, 0, 1]])
    ### Create camera frustum ###
    if step == 0:
        color = [1, 0, 0]
    elif step == len(cam_traj_subset) - 1:
        color = [0, 0, 1]
    else:
        color = [0, 1, 0]
    camera_frustum = create_camera_frustum(color=color, extrinsic=pose, intrinsic=intrinsic, scale=1)
    vis.add_geometry(camera_frustum)

    ##################################################
    ### Add line
    ##################################################
    if step > 0:
        points = [pose[:3, 3] for pose in cam_traj_subset[step-1:step+1]]
        line_set = create_dashed_line(points, color=[0, 0, 0])
        vis.add_geometry(line_set)

    ##################################################
    ### set camera view
    ##################################################
    if cam_json is not None:
        view_control.convert_from_pinhole_camera_parameters(vis_cam_param, allow_arbitrary=True)

    ##################################################
    ### update visualizer
    ##################################################
    vis.poll_events()
    vis.update_renderer()

    ##################################################
    ### save visualization
    ##################################################
    if args.out_dir is not None:
        os.makedirs(args.out_dir, exist_ok=True)
        render_filepath = os.path.join(args.out_dir, f"{step*skip_step:04}.png")
        vis.capture_screen_image(render_filepath)
    
    time.sleep(0.01)

### RUN ###
if args.with_interact:
    save_camera_parameters(vis)
    vis.run()

    vis.destroy_window()
else:
    vis.destroy_window()
