#!/bin/bash
##################################################
### This script is to visualize a trajectory on
### a given mesh. Trajectory poses are extracted
### from the given checkpoint
##################################################

### Input arguments ###
mesh_dir=$1
ckpt_file=$2
cam_view=$3

### get trajectory from checkpoint ###
python src/slam/export_pose.py \
--ckpt $ckpt_file

### visualize trajectory ###
traj_file = ${ckpt_file}.pose.npy
python src/visualization/vis_mesh_evo.py \
--mesh_dir $mesh_dir \
--traj_file $traj_file \
--cam_json src/visualization/default_camera_view.json 
