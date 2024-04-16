dataset=${1:-MP3D}
scene=${2:-gZ6f7yhEvPG}
iter=${3:-5000}

vis_dir=tmp/${dataset}_vis/${scene}

### Download data
mkdir -p tmp/${dataset}_vis/${scene}
rsync -avz us000245@172.24.209.46:/home/us000245/projects/naruto/results/${dataset}/${scene}/NARUTO/run_0/visualization/ tmp/${dataset}_vis/${scene}
rsync -avz us000245@172.24.209.46:/home/us000245/projects/naruto/results/${dataset}/${scene}/NARUTO/run_0/coslam/mesh/mesh_${iter}_final_cull_occlusion.ply tmp/${dataset}_vis/${scene}/

### prepare suitable camera viewpoints
### visualize trajectory ###
# python src/visualization/vis_traj.py \
# --mesh_file $vis_dir/mesh_${iter}_final_cull_occlusion.ply \
# --traj_dir $vis_dir/pose \
# --cam_json src/visualization/default_camera_view.json \
# --with_interact 1

# ### prepare data
# bash scripts/evaluation/visualize_naruto.sh $vis_dir data/visualization/${dataset}/${scene}_view1.json 1 uncert_mesh 0
# bash scripts/evaluation/visualize_naruto.sh $vis_dir data/visualization/${dataset}/${scene}_view2.json 1 uncert_mesh 0
# bash scripts/evaluation/visualize_naruto.sh $vis_dir data/visualization/${dataset}/${scene}_view1.json 1 color_mesh 0
# bash scripts/evaluation/visualize_naruto.sh $vis_dir data/visualization/${dataset}/${scene}_view2.json 1 color_mesh 0
# bash scripts/evaluation/visualize_traj.sh $vis_dir/mesh_${iter}_final_cull_occlusion.ply $vis_dir/pose data/visualization/${dataset}/${scene}_view1.json $vis_dir/traj_vis_at_${scene}_view1 0
# bash scripts/evaluation/visualize_traj.sh $vis_dir/mesh_${iter}_final_cull_occlusion.ply $vis_dir/pose data/visualization/${dataset}/${scene}_view2.json $vis_dir/traj_vis_at_${scene}_view2 0

# make video
# python src/visualization/naruto_video_maker.py --scene $scene --base_dir ${vis_dir} --out_video ${dataset}_${scene}.mp4 --pb_speed 1
