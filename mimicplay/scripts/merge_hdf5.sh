HUMAN_DATA="/home/haotian/MimicPlay/mimicplay/datasets/playdata/pick_place_red_cup_80_human.hdf5"
ROBOT_DATA="/home/haotian/MimicPlay/mimicplay/datasets/playdata/pick_place_red_cup_robot_578.hdf5"

OUTPUT_DIR="/home/haotian/MimicPlay/mimicplay/datasets/playdata/pick_place_red_cup_robot_578_train.hdf5"

ROBOT_JSON="/home/haotian/MimicPlay/mimicplay/datasets/playdata/pick_place_red_cup_robot_578_train_summary.json"

python merge_hdf5.py \
    merge \
    --robot ${ROBOT_DATA}\
    --human ${HUMAN_DATA}:0 \
    --output ${OUTPUT_DIR} \
    --robot_val 100-104 \