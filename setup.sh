pip install mujoco

cd robosuite
git checkout v1.4.1_libero
pip install -r requirements.txt
pip install -r requirements-extra.txt
pip install -e .

cd ..
cd bddl
pip install -e .

# Install LIBERO
cd ..
cd LIBERO
pip install -r requirements.txt
pip install -e .

cd ..
cd robomimic
git checkout mimicplay-libero
pip install -e .


cd ..
cd mimicplay/scripts/human_playdata_process
git clone https://github.com/DavidZ2851/hand_object_detector.git
cd hand_object_detector
git checkout feat/polaris
pip install -r requirements.txt
cd lib
python setup.py build develop
cd ..
mkdir -p models/res101_handobj_100K/pascal_voc
mv faster_rcnn_1_8_132028.pth models/res101_handobj_100K/pascal_voc/.
