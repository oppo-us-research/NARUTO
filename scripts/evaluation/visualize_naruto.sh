#!/bin/bash
##################################################
### This script is to visualize a trajectory on
### a given mesh. Trajectory poses are extracted
### from the given checkpoint
##################################################

### Input arguments ###
vis_dir=$1
cam_view=${2:-src/visualization/default_camera_view.json}
save_vis=${3:-1}
mesh_type=${4:-color_mesh}
with_interact=${5:-0}

### visualize trajectory ###
python src/visualization/naruto_o3d_visualizer.py \
--vis_dir $vis_dir \
--cam_json $cam_view \
--save_vis $save_vis \
--mesh_type $mesh_type \
--with_interact $with_interact
