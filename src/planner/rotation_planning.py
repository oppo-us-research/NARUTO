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


import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from typing import List


def Mat2Rotation(R_mat: np.ndarray) -> Rotation:
    """ convert Rotation from matrix to scipy.Rotation

    Args:
        R_mat (np.ndarray, [3,3]): rotation matrix

    Returns:
        R (Rotation): Rotation object
    """
    return Rotation.from_matrix(R_mat)


def Rotation2Mat(R: Rotation) -> np.ndarray:
    """ convert Rotation from matrix to scipy.Rotation

    Args:
        R (Rotation): Rotation object

    Returns:
        R_mat (np.ndarray, [3,3]): rotation matrix
    """
    return Rotation.as_matrix(R)


def angular_difference(R1: Rotation, R2: Rotation) -> float:
    """ compute angular difference between two rotations

    Args:
        R1 (Rotation): Rotation object
        R2 (Rotation): Rotation object
    
    Returns:
        angle_diff (float): angular difference in radians
    """
    ### Calculate the angular difference between two rotations ###
    angular_diff = R1.inv() * R2
    
    ### Convert the angular difference to an angle ###
    angle_diff = angular_diff.magnitude()
    
    return angle_diff


def minimize_movement(rotations: List, reference_rotation: Rotation) -> List:
    """ Given a set of rotations and a reference rotation. 
    Find the minimum movement that moves from reference_rotation to all other rotations

    Args:
        rotations (List)             : rotation objects
        reference_rotation (Rotation): Rotation object
    
    Returns:
        sorted_rotations (list): sorted rotation ojbects
    """
    ### Create a list to store the sorted rotations ###
    sorted_rotations = []
    
    ### Create a copy of the input rotations to avoid modifying the original list ###
    remaining_rotations = list(rotations)
    
    ### Start with the reference rotation ###
    current_rotation = reference_rotation
    sorted_rotations.append(current_rotation)
    
    while remaining_rotations:
        ### Find the rotation that minimizes the movement from the current rotation ###
        min_movement_rotation = min(remaining_rotations, key=lambda r: angular_difference(current_rotation, r))
        
        ### Add the selected rotation to the sorted list and remove it from remaining rotations ###
        sorted_rotations.append(min_movement_rotation)
        remaining_rotations.remove(min_movement_rotation)
        
        ### Update the current rotation for the next iteration ###
        current_rotation = min_movement_rotation
    
    return sorted_rotations


def create_rotation_interpolation(R1: Rotation, R2: Rotation) -> Slerp:
    """ Create rotation interpolation function for two given rotations

    Args:
        R1 (Rotation): Rotation object
        R2 (Rotation): Rotation object
    
    Returns:
        rot_interp (Slerp): Spherical Linear Interpolation
    """
    rots = Rotation.concatenate([R1, R2])
    rot_interp = Slerp (np.array([0, 1]), rots)
    return rot_interp


def interpolate_rotation(R1: Rotation, R2: Rotation, step_deg: float) -> List:
    """ Interpolate rotations between R1 and R2 with maximum step (deg)

    Args:
        R1 (Rotation)   : Rotation object
        R2 (Rotation)   : Rotation object
        step_deg (float): maximum step degree
    
    Returns:
        interpolated_rotations (List): Rotation objects
    """
    ### Calculate the total angle (in degrees) between R1 and R2 ###
    total_angle_degrees = (R1.inv() * R2).magnitude() / np.pi * 180
    
    ### Calculate the number of steps needed ###
    num_steps = int(total_angle_degrees / step_deg)
    
    ### Initialize a list to store interpolated rotations ###
    interpolated_rotations = [R1]

    ### create rotation interpolation function ###
    rot_interp = create_rotation_interpolation(R1, R2)
    
    ### Interpolate between R1 and R2 ###
    for i in range(1, num_steps):
        t = i / num_steps
        
        interpolated_rotation = rot_interp(t)
        interpolated_rotations.append(interpolated_rotation)
    
    ### Add R2 to ensure it's included ###
    interpolated_rotations.append(R2)
    
    return interpolated_rotations


def rotation_planning(R_mat: np.ndarray, target_Rs_mat: np.ndarray, max_rot_deg: float) -> List:
    """plan rotations such that R moves to target_Rs with max_rot_deg for each movement.
    The overall motion has to be minimised

    Args:
        R_mat (np.ndarray, [3,3])          : Current rotation
        target_Rs_mat (np.ndarray, [N,3,3]): target rotations
        max_rot_deg (float)                : maximum rotation degree

    Returns:
        planned_Rs (List): planned rotations. each element is (np.ndarray, [3,3])
    """
    ### convert rotations from Matrix to Rotation objects ###
    R = Mat2Rotation(R_mat)
    target_Rs = [Mat2Rotation(i) for i in target_Rs_mat]

    ### plan rotations that minimize motions ###
    sorted_rotations = minimize_movement(target_Rs, R)

    ### interpolate rotation movements ###
    for i in range(len(sorted_rotations) - 1):
        ### interpolate rotations ###
        tmp_rotations = interpolate_rotation(sorted_rotations[i], sorted_rotations[i+1], step_deg=max_rot_deg)
        if i == 0:
            ### initiali planned rotations ###
            planned_Rs = tmp_rotations
        else:
            ### exclude the first rotation as it is included in the previous timestep###
            planned_Rs += tmp_rotations[1:]

    ### Convert planned rotations back to matrix ###
    planned_Rs_mat = [Rotation2Mat(i) for i in planned_Rs]
    return planned_Rs_mat


if __name__ == "__main__":
    ### Create rotation examples in quaternion ###
    reference_rotation = Rotation.as_matrix(Rotation.from_rotvec([0, 0, 0]))
    
    rotations = [
        Rotation.as_matrix(Rotation.from_rotvec([3.14, 0, 0])),
        Rotation.as_matrix(Rotation.from_rotvec([1.3, 0, 0])),
        Rotation.as_matrix(Rotation.from_rotvec([2.5, 0, 0])),
    ]

    ### rotation planning ###
    planned_rotations = rotation_planning(reference_rotation, rotations, 10)
    planned_rotations = [Rotation.as_rotvec(Rotation.from_matrix(i)) for i in planned_rotations]

    ### print result ###
    for i in planned_rotations:
        print(i)
