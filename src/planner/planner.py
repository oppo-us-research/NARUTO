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


import mmengine
import numpy as np

from src.utils.general_utils import InfoPrinter
from src.utils.timer import Timer


class Planner():
    def __init__(self, 
                 main_cfg: mmengine.Config,
                 info_printer: InfoPrinter
                 ) -> None:
        """
        Args:
            main_cfg (mmengine.Config): Configuration
            info_printer (InfoPrinter): information printer
    
        Attributes:
            main_cfg (mmengine.Config)   : configurations
            planner_cfg (mmengine.Config): planner configurations
            info_printer (InfoPrinter)   : information printer
            
        """
        self.main_cfg = main_cfg
        self.planner_cfg = main_cfg.planner
        self.info_printer = info_printer
        self.step = 0

        self.init_timer()

    def update_step(self, step):
        """ update step information
    
        Args:
            step (int): step size
    
        """
        self.step = step

    def init_timer(self):
        """ initialize timer if requested
        Attributes:
            timer (Timer): timer object
            
        """
        self.timer = Timer()
        if self.planner_cfg.get("enable_timing", False):
            self.enable_timing = True
        else:
            self.enable_timing = False
    
    def update_sim(self, sim):
        """ initialize/update a Simulator if requested
        Attributes:
            sim (Simulator): Simulator object
            
        """
        self.sim = sim

    def vox2loc(self, vox, bbox=None, voxel_size=None):
        """ convert voxel coordinates to metric coordinates
    
        Args:
            vox (np.ndarray, [3])   : voxel coordinates
            bbox (np.ndarray, [3,2]): bounding box corner coordinates. Use self.bbox if not provided
            voxel_size (float)      : voxel size. Unit: meter. Use self.bbox if not provided
    
        Returns:
            loc (np.ndarray, [3]): metric coordinates
        """
        bbox = bbox if bbox is not None else self.bbox
        voxel_size = voxel_size if voxel_size is not None else self.voxel_size

        loc = vox * voxel_size + bbox[:, 0]
        return loc
    
    def loc2vox(self, loc, bbox=None, voxel_size=None):
        """ convert metric coordinates to voxel coordinates.
    
        Args:
            loc (np.ndarray, [3])   : metric coordinates
            bbox (np.ndarray, [3,2]): bounding box corner coordinates. Use self.bbox if not provided
            voxel_size (float)      : voxel size. Unit: meter. Use self.bbox if not provided
    
        Returns:
            vox (np.ndarray, [3]): voxel coordinates
        """
        bbox = bbox if bbox is not None else self.bbox
        voxel_size = voxel_size if voxel_size is not None else self.voxel_size

        vox = (loc - bbox[:, 0]) / voxel_size
        return vox

def compute_camera_pose(A: np.ndarray, B: np.ndarray, up_dir: np.ndarray = np.array([0, 0, 1])) -> np.ndarray:
    """ compute camera pose given current location A and look-at location B.
    Using OpenGL (RUB) coordinate system. 
    up_dir is the up direction w.r.t world coorindate origin pose.

    Args:
        A (np.ndarray, [3])     : current location
        B (np.ndarray, [3])     : look-at location
        up_dir (np.ndarray, [3]): up direction in world coordinate
    
    Returns:
        M (np.ndarray, [3, 3]): rotation matrix
    """
    # viewing direction (backward)
    V = A - B

    ### FIXME: for edge case that target points in the same x,y position ###
    if V[0] == 0 and V[1] == 0:
        V[0] = 1e-6

    # right viewing direction
    R = np.cross(up_dir, V)

    # up viewing direction
    U = np.cross(V, R)

    # normalize
    V = V / np.linalg.norm(V)
    R = R / np.linalg.norm(R)
    U = U / np.linalg.norm(U)

    # construct pose matrix
    M = np.column_stack((R, U, V))  

    return M
    
