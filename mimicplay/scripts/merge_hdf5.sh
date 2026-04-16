HUMAN_DATA="/project_data/held/haotian/MimicPlay/mimicplay/datasets/playdata/pick_place_red_cup_human_120.hdf5"
ROBOT_DATA="/project_data/held/haotian/MimicPlay/mimicplay/datasets/playdata/pick_place_red_cup_robot_105.hdf5"

OUTPUT_DIR="/project_data/held/haotian/MimicPlay/mimicplay/datasets/playdata_setting_v2/pick_place_red_cup_robot_40_human_100.hdf5"

ROBOT_JSON="/project_data/held/haotian/MimicPlay/mimicplay/datasets/playdata_setting_v2/pick_place_red_cup_robot_40_human_0_summary.json"

python merge_hdf5.py \
    --robot ${ROBOT_DATA}\
    --human ${HUMAN_DATA}:100 \
    --output ${OUTPUT_DIR} \
    --robot_val 100-104 \
    --robot_json ${ROBOT_JSON}\