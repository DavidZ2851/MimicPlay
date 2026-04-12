HUMAN_DATA="/home/haotian/MimicPlay/mimicplay/datasets/playdata/pick_place_red_cup_40_human.hdf5"
ROBOT_DATA="/home/haotian/MimicPlay/mimicplay/datasets/playdata/pick_place_red_cup_30_robot.hdf5"

OUTPUT_DIR="/home/haotian/MimicPlay/mimicplay/datasets/playdata/pick_place_red_cup_robot_30_human_20.hdf5"


# python merge_hdf5.py --human human.hdf5:50 --robot robot.hdf5:30 --output merged.hdf5 --val_ratio 0.1

python merge_hdf5.py \
    --robot ${ROBOT_DATA}:30 \
    --human ${HUMAN_DATA}:20 \
    --output ${OUTPUT_DIR} \
    --n_val 5