#!/bin/bash
##################################################
### This script is to visualize a trajectory on
### a given mesh. Trajectory poses are extracted
### from the given checkpoint
##################################################

### Input arguments ###
mesh_file=$1
traj_dir=$2
cam_view=${3:-src/visualization/default_camera_view.json}
out_dir=$4
with_interact=${5:-0}

### visualize trajectory ###
python src/visualization/vis_traj.py \
--mesh_file $mesh_file \
--traj_dir $traj_dir \
--cam_json $cam_view \
--out_dir $out_dir \
--with_interact $with_interact
