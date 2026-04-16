#!/bin/bash

DELAY=150
DATA_ROOT="/project_data/held/haotian/MimicPlay/mimicplay/datasets"
HL_MODEL_ROOT="/project_data/held/haotian/MimicPlay/trained_models_highlevel"

# Checkpoints
declare -A CHECKPOINTS=(
    ["mimicplay_robot_5_human_0"]="$HL_MODEL_ROOT/mimicplay_robot_5_human_0/20260416113250/models/model_epoch_3000.pth"
    ["mimicplay_robot_5_human_40"]="$HL_MODEL_ROOT/mimicplay_robot_5_human_40/20260416113520/models/model_epoch_3000.pth"
    ["mimicplay_robot_5_human_70"]="$HL_MODEL_ROOT/mimicplay_robot_5_human_70/20260416020021/models/model_epoch_3000.pth"
    ["mimicplay_robot_5_human_100"]="$HL_MODEL_ROOT/mimicplay_robot_5_human_100/20260416020147/models/model_epoch_3000.pth"
    ["mimicplay_robot_40_human_0"]="$HL_MODEL_ROOT/mimicplay_robot_40_human_0/20260416113750/models/model_epoch_3000.pth"
    ["mimicplay_robot_40_human_40"]="$HL_MODEL_ROOT/mimicplay_robot_40_human_40/20260416020417/models/model_epoch_3000.pth"
    ["mimicplay_robot_40_human_70"]="$HL_MODEL_ROOT/mimicplay_robot_40_human_70/20260416020649/models/model_epoch_3000.pth"
    ["mimicplay_robot_40_human_100"]="$HL_MODEL_ROOT/mimicplay_robot_40_human_100/20260416020649/models/model_epoch_3000.pth"
)

# Datasets
declare -A DATASETS=(
    ["mimicplay_robot_5_human_0"]="$DATA_ROOT/playdata_setting_v2/pick_place_red_cup_robot_train_5.hdf5"
    ["mimicplay_robot_5_human_40"]="$DATA_ROOT/playdata_setting_v2/pick_place_red_cup_robot_train_5.hdf5"
    ["mimicplay_robot_5_human_70"]="$DATA_ROOT/playdata_setting_v2/pick_place_red_cup_robot_train_5.hdf5"
    ["mimicplay_robot_5_human_100"]="$DATA_ROOT/playdata_setting_v2/pick_place_red_cup_robot_train_5.hdf5"
    ["mimicplay_robot_40_human_0"]="$DATA_ROOT/playdata_setting_v2/pick_place_red_cup_robot_train_40.hdf5"
    ["mimicplay_robot_40_human_40"]="$DATA_ROOT/playdata_setting_v2/pick_place_red_cup_robot_train_40.hdf5"
    ["mimicplay_robot_40_human_70"]="$DATA_ROOT/playdata_setting_v2/pick_place_red_cup_robot_train_40.hdf5"
    ["mimicplay_robot_40_human_100"]="$DATA_ROOT/playdata_setting_v2/pick_place_red_cup_robot_train_40.hdf5"
)

# Jobs: GPU NAME
declare -a JOBS=(
    "1 mimicplay_robot_5_human_0"
    "2 mimicplay_robot_5_human_40"
    "3 mimicplay_robot_5_human_70"
    "4 mimicplay_robot_5_human_100"
    "5 mimicplay_robot_40_human_0"
    "6 mimicplay_robot_40_human_40"
    "7 mimicplay_robot_40_human_70"
)

for i in "${!JOBS[@]}"; do
    read -r gpu name <<< "${JOBS[$i]}"
    
    dataset="${DATASETS[$name]}"
    hl_path="${CHECKPOINTS[$name]}"
    
    CUDA_VISIBLE_DEVICES=$gpu nohup python scripts/train.py \
        --config configs/lowlevel.json \
        --name "$name" \
        --dataset "$dataset" \
        --hl_path "$hl_path" > /dev/null 2>&1 &
    
    echo "Job $gpu ($name) started"
    
    if [ $i -lt $((${#JOBS[@]} - 1)) ]; then
        sleep $DELAY
    fi
done

echo "All jobs started"