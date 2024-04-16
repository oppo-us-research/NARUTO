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
from typing import Dict

import os, sys
import time


sys.path.append(os.getcwd())
from src.visualization.o3d_utils import (
    create_camera_frustum, 
    save_camera_parameters, 
    load_camera_parameters_from_json, 
    create_dashed_line,
    LineMesh
    )


class NarutoO3DVisualizer():
    def __init__(self, 
                 vis_dir : str,
                 cam_json: str,
                 cfg     : Dict = None,
                 save_vis: bool = False,
                 with_interact: bool = False
                 ):
        """
    
        Args:
            vis_dir: visualization data directory
            cam_json: viewing camera parameters 
            cfg: configurations
            save_vis: save rendering result
            with_interact: no interactive actions at the end
        """
        self.cfg = cfg
        self.window_hw = cfg['window_hw']
        self.cam_json = cam_json
        self.vis_dir = vis_dir
        self.save_vis = save_vis
        self.with_interact = with_interact


        self.mesh_type = self.cfg['mesh_type']
        self.mesh_dir = os.path.join(self.vis_dir, self.mesh_type)

        self.vis_cam_param = load_camera_parameters_from_json(cam_json)

        self.init_window()
        self.load_poses()
        self.view_json_name, _ = os.path.splitext(os.path.basename(self.cam_json))
        os.makedirs(os.path.join(self.vis_dir, f"rendered_{self.mesh_type}_at_{self.view_json_name}"), exist_ok=True)
        return
    
    def init_window(self):
        """initialize windows
    
        Attributes:
            vis (o3d.visualization.VisualizerWithKeyCallback): Open3D Visualizer
            view_control (open3d.visualization.ViewControl)  : view control
        """
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(
            window_name = self.mesh_type, 
            width=self.window_hw[1], height=self.window_hw[0]
            )
        self.view_control = self.vis.get_view_control()
        # self.vis.get_render_option().background_color = [0.5, 0.5, 0.5]
    
    def draw_planning_path(self, step: int):
        """draw planning paths for the given step
    
        Args:
            step: step
            
        """
        filepath = os.path.join(self.vis_dir, 'planning_path', f"{step:04}.npy")
        path = np.load(filepath, allow_pickle=True)
        if step != 0:
            points = [point for point in path]
            if len(points) > 1:
                # line_set = create_dashed_line(points, color=[0, 0, 0])
                # self.vis.add_geometry(line_set)

                ### Use thicker lines ###
                line_mesh1 = LineMesh(points, colors=[0,0,0], radius=0.02)
                line_mesh1.add_line(self.vis)
            
    
    def draw_lookat_tgts(self, step: int, pose: np.ndarray):
        """ draw lookat targets (i.e. uncertain target observations)
    
        Args:
            step: step
        """
        filepath = os.path.join(self.vis_dir, 'lookat_tgts', f"{step:04}.npy")
        lookat_tgts = np.load(filepath, allow_pickle=True)
        if step != 0:
            for lookat_tgt in lookat_tgts:
                points = [pose[:3, 3], lookat_tgt]
                # line_set = create_dashed_line(points, color=[1, 1, 1])
                # self.vis.add_geometry(line_set)

                ### Use thicker lines ###
                line_mesh = LineMesh(points, colors=[1,1,1], radius=0.02)
                line_mesh.add_line(self.vis)
            
    
    def load_poses(self):
        """ load camera poses
    
        Attributes:
            cam_poses (np.ndarray, [N,4,4]): camera poses, camera-to-world. RUB
        """
        pose_files = sorted(glob(os.path.join(self.vis_dir, "pose", "*.npy")))
        self.cam_poses = np.stack([np.load(filepath) for filepath in pose_files])
        self.cam_poses = self.cam_poses[:2000]

    def main(self):
        """ Run visualization
        """
        mesh = None
        for step, pose in enumerate(self.cam_poses):
            pose = self.cam_poses[step]
            if step < len(self.cam_poses)-1:
                last_mesh = mesh
                self.vis.clear_geometries()

            if step % self.cfg.get("skip_step", 5) != 0 and step != len(self.cam_poses)-1:
                continue

            ##################################################
            ### Add mesh
            ##################################################
            mesh_file = os.path.join(self.mesh_dir, f"mesh_{step:04}.ply")
            if os.path.isfile(mesh_file):
                ### Add new mesh ###
                mesh = o3d.io.read_triangle_mesh(mesh_file)
                self.vis.add_geometry(mesh)
            else:
                self.vis.add_geometry(last_mesh)

            ##################################################
            ### Add Camera ###
            ##################################################
            intrinsic = np.array([[300, 0, 300],
                                [0, 300, 300],
                                [0, 0, 1]])
            ### Create camera frustum ###
            if step == 0:
                color = [1, 0, 0]
            elif step == len(self.cam_poses) - 1:
                color = [0, 0, 1]
            else:
                color = [0, 0, 0]
            camera_frustum = create_camera_frustum(color=color, extrinsic=pose, intrinsic=intrinsic, scale=1)
            self.vis.add_geometry(camera_frustum)

            ##################################################
            ### Add line
            ##################################################
            if step > 0:
                points = [pose[:3, 3] for pose in self.cam_poses[step-1:step+1]]
                line_set = create_dashed_line(points, color=[0, 0, 0])
                self.vis.add_geometry(line_set)

            ##################################################
            ### Draw planning paths and targets
            ##################################################
            if self.cfg['draw_planning_path']:
                self.draw_planning_path(step)
            if self.cfg['draw_lookat_tgts']:
                self.draw_lookat_tgts(step, pose)

            ##################################################
            ### set camera view
            ##################################################
            self.view_control.convert_from_pinhole_camera_parameters(self.vis_cam_param, allow_arbitrary=True)

            ##################################################
            ### update visualizer
            ##################################################
            self.vis.poll_events()
            self.vis.update_renderer()

            ##################################################
            ### save visualization
            ##################################################
            if self.save_vis:
                render_filepath = os.path.join(self.vis_dir, f"rendered_{self.mesh_type}_at_{self.view_json_name}", f"{step:04}.png")
                self.vis.capture_screen_image(render_filepath)
            time.sleep(0.1)

        if self.with_interact:
            ### RUN ###
            save_camera_parameters(self.vis)
            self.vis.run()
            self.vis.destroy_window()
        else:
            self.vis.destroy_window()


if __name__ == "__main__":
    def argument_parsing() -> argparse.Namespace:
        """parse arguments

        Returns:
            args: arguments
            
        """
        parser = argparse.ArgumentParser(
                description="Arguments to visualize NARUTO Planning and Mapping."
            )
        parser.add_argument("--vis_dir", type=str, default="", 
                            help="visualization data directory")
        parser.add_argument("--cam_json", type=str, default=None, 
                            help="trajectory pose file")
        parser.add_argument("--save_vis", type=int, default=0, 
                    help="1: enable save visualization")
        parser.add_argument("--mesh_type", type=str, default="color_mesh", 
                            help="mesh type")
        parser.add_argument("--with_interact", type=int, default=0, 
                    help="disable interactive visualization")
        args = parser.parse_args()
        return args

    args = argument_parsing()
    cfg = dict(
        draw_planning_path = True,
        draw_lookat_tgts = True,
        mesh_type = args.mesh_type,
        window_hw = (1024, 1024),
        skip_step = 5
    )
    visualizer = NarutoO3DVisualizer(
        args.vis_dir, 
        args.cam_json, 
        cfg, 
        args.save_vis==1, 
        args.with_interact==1
        )
    visualizer.main()