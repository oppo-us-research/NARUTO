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
from glob import glob
import json
import numpy as np
import open3d as o3d
import os
from typing import List, Tuple


def create_camera_frustum(
        color    : List,
        extrinsic: np.ndarray,
        intrinsic: np.ndarray,
        scale    : float=1.0
        ) -> o3d.geometry.LineSet:
    """Create a frustum for visualization.

    Args:
        color (List)                 : Color of the frustum.
        extrinsic (np.ndarray, [4,4]): Extrinsic matrix of the camera (4x4).
        intrinsic (np.ndarray, [3,3]): Intrinsic matrix of the camera (3x3).
        scale (float)                : Scale of the frustum.

    Returns: 
        line_set (open3d.geometry.LineSet) representing the frustum.
    """
    # Create frustum points (in camera coordinates)
    fovy = 2 * np.arctan( intrinsic[1, 2] / intrinsic[1, 1]) # Assuming square pixels
    aspect_ratio = intrinsic[0, 0] / intrinsic[1, 1]
    near_plane = 0.00
    far_plane = 0.1 * scale

    frustum_points = []
    for z in [near_plane, far_plane]:
        y = np.tan(fovy / 2) * z
        x = aspect_ratio * y
        frustum_points.extend([[-x, -y, -z], [x, -y, -z], [x, y, -z], [-x, y, -z]])

    # Transform frustum points to world coordinates
    frustum_points = np.dot(extrinsic, np.hstack((frustum_points, np.ones((8, 1)))).T).T[:, :3]
    frustum_points = frustum_points[:, :3]

    # Scale down the frustum size
    # frustum_points *= scale

    # Create lines
    lines = [[0, 4], [1, 5], [2, 6], [3, 7], # lines from origin to far plane
             [4, 5], [5, 6], [6, 7], [7, 4]] # lines in the far plane

    # Create line set
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(frustum_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])

    return line_set


def create_dashed_line(
        points     : np.ndarray,
        dash_length: float = 0.05,
        gap_length : float = 0.05,
        color      : List = [1, 0, 0]
        ) -> o3d.geometry.LineSet: 
    """
    Create a dashed line from a list of points.

    Args:
        points     : List of points (Nx3 array).
        dash_length: Length of each dash.
        gap_length : Length of the gap between dashes.
        color      : Color of the dashes (RGB list).

    Returns:
        open3d.geometry.LineSet representing the dashed line.
    """
    assert len(points) > 1, "There should be at least two points to create a line."

    line_points = []
    line_indices = []
    line_colors = []

    for i in range(len(points) - 1):
        start_point = points[i]
        end_point = points[i + 1]
        segment_vector = end_point - start_point
        segment_length = np.linalg.norm(segment_vector)
        if segment_length != 0:
            segment_vector /= segment_length # normalize

        current_pos = 0
        while current_pos + dash_length < segment_length:
            dash_start = start_point + current_pos * segment_vector
            dash_end = start_point + (current_pos + dash_length) * segment_vector
            line_points.append(dash_start)
            line_points.append(dash_end)
            line_indices.append([len(line_points) - 2, len(line_points) - 1])
            line_colors.append(color)
            current_pos += dash_length + gap_length

    # Create line set
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(line_points)
    line_set.lines = o3d.utility.Vector2iVector(line_indices)
    line_set.colors = o3d.utility.Vector3dVector(line_colors)

    return line_set


def save_camera_parameters(vis: o3d.visualization.VisualizerWithKeyCallback) -> None:
    """ Callback function to save camera parameters when the 'S' key is pressed.
    
    Args:
        vis: The Open3D visualizer instance.
    """
    # Define the key callback function
    def key_callback(vis, action, mods):
        # Get current camera parameters
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()
        # Define the filename to save the parameters
        filename = "saved_camera_params.json"
        # Save the camera parameters to a file
        o3d.io.write_pinhole_camera_parameters(filename, param)
        print(f"Camera parameters saved to {filename}")

    # Register the key callback function to the visualizer
    vis.register_key_action_callback(ord("S"), key_callback)


def load_camera_parameters_from_json(json_file: str) -> o3d.camera.PinholeCameraParameters:
    """ load camera parameters from json

    Args:
        json_file (str): camera parameter json file
    
    Returns:
        cam_param (o3d.camera.PinholeCameraParameters): camera parameters
    """
    with open(json_file, 'r') as file:
        data = json.load(file)

    # Load intrinsic parameters
    intrinsic_params = data["intrinsic"]
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        intrinsic_params["width"],
        intrinsic_params["height"],
        intrinsic_params["intrinsic_matrix"][0], # fx
        intrinsic_params["intrinsic_matrix"][4], # fy
        intrinsic_params["intrinsic_matrix"][2], # cx
        intrinsic_params["intrinsic_matrix"][5] # cy
    )

    # Load extrinsic parameters
    extrinsic_params = data["extrinsic"]
    extrinsic_matrix = np.array(extrinsic_params).reshape((4, 4))
    extrinsic_matrix = extrinsic_matrix.transpose()

    # Create PinholeCameraParameters
    cam_param = o3d.camera.PinholeCameraParameters()
    cam_param.extrinsic = extrinsic_matrix
    cam_param.intrinsic = intrinsic

    return cam_param


def load_rgbd_images(img_dir: str, scale: float = 1.) -> Tuple[List, List]:
    """load rgbd images from img_direcotry

    Args:
        img_dir: image directory to load RGB-D
        scale  : resize factor

    Returns:
        Tuple:
            - rgb_images: List of (o3d.geometry.Image)
            - depth_images: List of (o3d.geometry.Image)

    Attributes:
        
    """
    rgbd_paths = sorted(glob(os.path.join(img_dir, "*.png")))
    rgb_images = []
    depth_images = []
    for img_path in rgbd_paths:
        ### load image ###
        rgbd_img = o3d.io.read_image(img_path)
        rgbd_img = np.asarray(rgbd_img)


        ### resize image ###
        h, w, _ = rgbd_img.shape
        h, w = int(h * scale), int(w * scale) 
        rgbd_img = cv2.resize(rgbd_img, (w, h))
        
        ### split RGB-D ###
        rgb_img = np.ascontiguousarray(rgbd_img[:, :w//2])
        depth_img = np.ascontiguousarray(rgbd_img[:, w//2:])

        ### convert to o3d format ###
        rgb_img = o3d.geometry.Image(rgb_img)
        depth_img = o3d.geometry.Image(depth_img)

        rgb_images.append(rgb_img)
        depth_images.append(depth_img)

    return rgb_images, depth_images


def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle


def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2


class LineMesh(object):
    def __init__(self, points, lines=None, colors=[0, 1, 0], radius=0.15):
        """Creates a line represented as sequence of cylinder triangular meshes

        Arguments:
            points {ndarray} -- Numpy array of ponts Nx3.

        Keyword Arguments:
            lines {list[list] or None} -- List of point index pairs denoting line segments. If None, implicit lines from ordered pairwise points. (default: {None})
            colors {list} -- list of colors, or single color of the line (default: {[0, 1, 0]})
            radius {float} -- radius of cylinder (default: {0.15})
        """
        self.points = np.array(points)
        self.lines = np.array(
            lines) if lines is not None else self.lines_from_ordered_points(self.points)
        self.colors = np.array(colors)
        self.radius = radius
        self.cylinder_segments = []

        self.create_line_mesh()

    @staticmethod
    def lines_from_ordered_points(points):
        lines = [[i, i + 1] for i in range(0, points.shape[0] - 1, 1)]
        return np.array(lines)

    def create_line_mesh(self):
        first_points = self.points[self.lines[:, 0], :]
        second_points = self.points[self.lines[:, 1], :]
        line_segments = second_points - first_points
        ### filter invalid segments ###
        valid_mask = [np.all(line_seg!=0) for line_seg in line_segments]
        line_segments = line_segments[valid_mask]

        line_segments_unit, line_lengths = normalized(line_segments)

        z_axis = np.array([0, 0, 1])
        # Create triangular mesh cylinder segments of line
        for i in range(line_segments_unit.shape[0]):
            line_segment = line_segments_unit[i, :]
            line_length = line_lengths[i]
            # get axis angle rotation to allign cylinder with line segment
            axis, angle = align_vector_to_another(z_axis, line_segment)
            # Get translation vector
            translation = first_points[i, :] + line_segment * line_length * 0.5
            # create cylinder and apply transformations
            cylinder_segment = o3d.geometry.TriangleMesh.create_cylinder(
                self.radius, line_length)
            cylinder_segment = cylinder_segment.translate(
                translation, relative=False)
            if axis is not None:
                axis_a = axis * angle
                cylinder_segment = cylinder_segment.rotate(
                    R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a), center=cylinder_segment.get_center())
                # cylinder_segment = cylinder_segment.rotate(
                #   axis_a, center=True, type=o3d.geometry.RotationType.AxisAngle)
            # color cylinder
            color = self.colors if self.colors.ndim == 1 else self.colors[i, :]
            cylinder_segment.paint_uniform_color(color)

            self.cylinder_segments.append(cylinder_segment)

    def add_line(self, vis):
        """Adds this line to the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.add_geometry(cylinder)

    def remove_line(self, vis):
        """Removes this line from the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.remove_geometry(cylinder)
