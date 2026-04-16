# !/bin/bash

CUDA_VISIBLE_DEVICES=0 nohup python scripts/train.py --config configs/highlevel_mix.json --name mimicplay_robot_5_human_0 --dataset /project_data/held/haotian/MimicPlay/mimicplay/datasets/playdata_setting_v2/pick_place_red_cup_robot_5_human_0.hdf5 > /dev/null 2>&1 &
echo "Job 1 started"; sleep 150

CUDA_VISIBLE_DEVICES=1 nohup python scripts/train.py --config configs/highlevel_mix.json --name mimicplay_robot_5_human_40 --dataset /project_data/held/haotian/MimicPlay/mimicplay/datasets/playdata_setting_v2/pick_place_red_cup_robot_5_human_40.hdf5 > /dev/null 2>&1 &
echo "Job 2 started"; sleep 150

CUDA_VISIBLE_DEVICES=2 nohup python scripts/train.py --config configs/highlevel_mix.json --name mimicplay_robot_5_human_70 --dataset /project_data/held/haotian/MimicPlay/mimicplay/datasets/playdata_setting_v2/pick_place_red_cup_robot_5_human_70.hdf5 > /dev/null 2>&1 &
echo "Job 3 started"; sleep 150

CUDA_VISIBLE_DEVICES=3 nohup python scripts/train.py --config configs/highlevel_mix.json --name mimicplay_robot_5_human_100 --dataset /project_data/held/haotian/MimicPlay/mimicplay/datasets/playdata_setting_v2/pick_place_red_cup_robot_5_human_100.hdf5 > /dev/null 2>&1 &
echo "Job 4 started"; sleep 150

CUDA_VISIBLE_DEVICES=2 nohup python scripts/train.py --config configs/highlevel_mix.json --name mimicplay_robot_40_human_0 --dataset /project_data/held/haotian/MimicPlay/mimicplay/datasets/playdata_setting_v2/pick_place_red_cup_robot_40_human_0.hdf5 > /dev/null 2>&1 &
echo "Job 5 started"; sleep 150

CUDA_VISIBLE_DEVICES=5 nohup python scripts/train.py --config configs/highlevel_mix.json --name mimicplay_robot_40_human_40 --dataset /project_data/held/haotian/MimicPlay/mimicplay/datasets/playdata_setting_v2/pick_place_red_cup_robot_40_human_40.hdf5 > /dev/null 2>&1 &
echo "Job 6 started"; sleep 150

CUDA_VISIBLE_DEVICES=6 nohup python scripts/train.py --config configs/highlevel_mix.json --name mimicplay_robot_40_human_70 --dataset /project_data/held/haotian/MimicPlay/mimicplay/datasets/playdata_setting_v2/pick_place_red_cup_robot_40_human_70.hdf5 > /dev/null 2>&1 &
echo "Job 7 started"

CUDA_VISIBLE_DEVICES=7 nohup python scripts/train.py --config configs/highlevel_mix.json --name mimicplay_robot_40_human_100 --dataset /project_data/held/haotian/MimicPlay/mimicplay/datasets/playdata_setting_v2/pick_place_red_cup_robot_40_human_100.hdf5 > /dev/null 2>&1 &
echo "Job 8 started"; sleep 150

echo "All jobs started"