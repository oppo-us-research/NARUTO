#!/bin/bash
##################################################
### This script is to run the full NARUTO system 
### (active planning and active ray sampling) 
###  on the NARUTO dataset.
##################################################

# Input arguments
scene=${1:-konoha_village}
num_run=${2:-1}
EXP=${3:-NARUTO} # config in configs/{DATASET}/{scene}/{EXP}.py will be loaded
ENABLE_VIS=${4:-0}

export CUDA_VISIBLE_DEVICES=6
PROJ_DIR=${PWD}
DATASET=NARUTO
RESULT_DIR=${PROJ_DIR}/results

##################################################
### Random Seed
###     also used to initialize agent pose 
###     from indexing a set of predefined pose/traj
##################################################
seeds=(0)
seeds=("${seeds[@]:0:$num_run}")

##################################################
### Scenes
###     choose one or all of the scenes
##################################################
scenes=(nine_tailed_fox running_naruto jiraiya naruto hokage_room)
# Check if the input argument is 'all'
if [ "$scene" == "all" ]; then
    selected_scenes=${scenes[@]} # Copy all scenes
else
    selected_scenes=($scene) # Assign the matching scene
fi

##################################################
### Main
###     Run for selected scenes for N trials
##################################################
for scene in $selected_scenes
do
    for i in "${!seeds[@]}"; do
        seed=${seeds[$i]}

        ### create result folder ###
        result_dir=${RESULT_DIR}/${DATASET}/$scene/${EXP}/run_${i}
        mkdir -p ${result_dir}

        ### run experiment ###
        CFG=configs/${DATASET}/${scene}/${EXP}.py
        python src/naruto/main.py --cfg ${CFG} --seed ${seed} --result_dir ${result_dir} --enable_vis ${ENABLE_VIS}
        
        ### get file paths ###
        CKPT=$(find ${result_dir}/coslam/checkpoint/ -type f -name "ckpt_*_final.pt")
        INPUT_MESH=$(find ${result_dir}/coslam/mesh/ -type f -name "mesh_*_final.ply")

        ### Cull mesh as Co-SLAM ###
        python third_parties/neural_slam_eval/cull_mesh.py \
        --config configs/${DATASET}/${scene}/coslam.yaml \
        --input_mesh $INPUT_MESH \
        --ckpt_path $CKPT \
        --remove_occlusion # GO-Surf strategy
    done
done
