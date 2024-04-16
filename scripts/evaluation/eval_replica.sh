#!/bin/bash
##################################################
### This script is to evaluate the full NARUTO system 
### (active planning and active ray sampling) 
###  on the Replica dataset.
##################################################

# Input arguments
scene=${1:-office0}
trial=${2:-0}
iter_num=${3:-2000}
EXP=${4:-NARUTO}

export CUDA_VISIBLE_DEVICES=0
PROJ_DIR=${PWD}
DATASET=Replica
RESULT_DIR=${PROJ_DIR}/results

##################################################
### Select trial indexs
##################################################
trials=(0 1 2 3 4)
# Check if the input argument is 'all'
if [ "$trial" == "all" ]; then
    selected_trials=${trials[@]} # Copy all trials
else
    selected_trials=($trial) # Assign the matching scene
fi

##################################################
### Select scenes
##################################################
scenes=(room0 room1 room2 office0 office1 office2 office3 office4)
# Check if the input argument is 'all'
if [ "$scene" == "all" ]; then
    selected_scenes=${scenes[@]} # Copy all scenes
else
    selected_scenes=($scene) # Assign the matching scene
fi

##################################################
### Main
###     Evaluate selected experiment 
##################################################
for scene in $selected_scenes
do
    for i in $trials; do
        ### get result folder ###
        result_dir=${RESULT_DIR}/${DATASET}/$scene/${EXP}/run_${i}
        result_txt=${result_dir}/eval_result.txt
        echo "==> Evaluating [${result_dir}]"

        ### get file paths ###
        CKPT=$result_dir/coslam/checkpoint/ckpt_${iter_num}_final.pt
        INPUT_MESH=$result_dir/coslam/mesh/mesh_${iter_num}_final.ply
        REC_MESH=$result_dir/coslam/mesh/mesh_${iter_num}_final_cull_occlusion.ply
        DASHSCENE=${scene: 0: 0-1}_${scene: 0-1}
        GT_MESH=$PROJ_DIR/data/replica_v1/${DASHSCENE}/mesh.ply

        ### Cull mesh as Co-SLAM ###
        python third_parties/neural_slam_eval/cull_mesh.py \
        --config configs/${DATASET}/${scene}/coslam.yaml \
        --input_mesh $INPUT_MESH \
        --ckpt_path $CKPT \
        --remove_occlusion # GO-Surf strategy

        ### Evaluate ###
        echo "==> Evaluating reconstruction result [accuracy, completeness, and completion ratio]"
        python src/evaluation/eval_recon.py \
        --rec_mesh $REC_MESH \
        --gt_mesh $GT_MESH \
        --result_txt $result_txt

        echo "==> Evaluating reconstruction result [MAD]"
        python src/evaluation/eval_mad.py \
        --cfg configs/${DATASET}/${scene}/NARUTO.py \
        --ckpt $CKPT --gt_mesh $GT_MESH \
        --result_txt $result_txt

        echo "==> Evaluating trajectory length (m)"
        python src/evaluation/eval_traj_length.py \
        --ckpt ${CKPT} \
        --result_txt $result_txt
    done
done
