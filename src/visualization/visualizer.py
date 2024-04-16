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


import cv2
import mmengine
import numpy as np
import torch
from typing import Union


from src.utils.general_utils import InfoPrinter

from third_parties.coslam.utils import colormap_image


class Visualizer():
    def __init__(self, 
                 main_cfg    : mmengine.Config,
                 info_printer: InfoPrinter
                 ) -> None:
        """
        Args:
            main_cfg (mmengine.Config): Configuration
            info_printer (InfoPrinter): information printer
    
        Attributes:
            main_cfg (mmengine.Config): configurations
            vis_cfg (mmengine.Config) : visualizer model configurations
            info_printer (InfoPrinter): information printer
            
        """
        self.main_cfg = main_cfg
        self.vis_cfg = main_cfg.visualizer
        self.info_printer = info_printer

    def update_step(self, step):
        """ update step information
    
        Args:
            step (int): step size
    
        """
        self.step = step

    def visualize_rgbd(self,
                       rgb       : torch.Tensor,
                       depth     : torch.Tensor,
                       max_depth : float = 100.,
                       vis_size  : int = 320,
                       return_vis: bool = False
                       ) -> Union[None, np.ndarray]:
        """ visualiz RGB-D 
        Args:
            rgb (torch.Tensor, [H,W,3]): color map. Range: 0-1
            depth (torch.Tensor, [H,W]): depth map.
            max_depth (float)          : maximum depth value
            vis_size (int)             : image size used for visualization
            return_vis (bool)          : return visualization (OpenCV format) if True

        Returns:
            Union: 
                - image (np.ndarray, [H,W,3]): RGB-D visualization if return_vis
        """
        ## process RGB ##
        rgb = cv2.cvtColor(rgb.cpu().numpy(), cv2.COLOR_RGB2BGR)
        rgb = cv2.resize(rgb, (vis_size, vis_size))

        ### process Depth map ###
        depth = depth.unsqueeze(0)
        mask = (depth < max_depth) * 1.0
        depth_colormap = colormap_image(depth, mask)
        depth_colormap = depth_colormap.permute(1, 2, 0).cpu().numpy()
        depth_colormap = cv2.resize(depth_colormap, (vis_size, vis_size))

        ### display RGB-D ###
        image = np.hstack((rgb, depth_colormap))

        ### return visualization ###
        if return_vis:
            return image
        else:
            cv2.namedWindow('RGB-D', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RGB-D', image)
            key = cv2.waitKey(1)
