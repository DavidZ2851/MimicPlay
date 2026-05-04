# # ── High-level dataset (human + robot, minimal obs) ──────────────────────────
# python create_highlevel_dataset.py \
#     --human  /home/haotian/MimicPlay/mimicplay/datasets/playdata/pick_place_red_cup_80_human.hdf5 \
#     --robot  /home/haotian/MimicPlay/mimicplay/datasets/playdata/pick_place_red_cup_robot_578.hdf5 \
#     --num_human       50 \
#     --num_robot_train 25 \
#     --num_robot_val   10 \
#     --output /home/haotian/MimicPlay/mimicplay/datasets/playdata/pick_place_red_cup_highlevel_test.hdf5 \
#     --seed 42

# ── Low-level dataset (robot only, ALL obs keys) ──────────────────────────────
python create_lowlevel_dataset.py \
    --robot     /home/haotian/MimicPlay/mimicplay/datasets/playdata/pick_place_red_cup_robot_1000.hdf5 \
    --num_val   10 \
    --output /home/haotian/MimicPlay/mimicplay/datasets/playdata/pick_place_red_cup_robot_train_1000.hdf5 \
    --seed 42