import h5py
import numpy as np

path = '/home/haotian/MimicPlay/mimicplay/datasets/playdata/pick_place_red_cup_robot_30_human_20.hdf5'

with h5py.File(path, 'r') as f:
    data = f['data']
    
    print("Checking all demos for shape consistency...\n")
    
    shapes = {
        'actions': [],
        'robot0_eef_pos': [],
        'robot0_eef_pos_future_traj': [],
        'agentview_image': [],
    }
    
    for demo_name in sorted(data.keys(), key=lambda x: int(x.split('_')[1])):
        demo = data[demo_name]
        obs = demo['obs']
        
        shapes['actions'].append((demo_name, demo['actions'].shape, demo['actions'].dtype))
        shapes['robot0_eef_pos'].append((demo_name, obs['robot0_eef_pos'].shape, obs['robot0_eef_pos'].dtype))
        shapes['robot0_eef_pos_future_traj'].append((demo_name, obs['robot0_eef_pos_future_traj'].shape, obs['robot0_eef_pos_future_traj'].dtype))
        shapes['agentview_image'].append((demo_name, obs['agentview_image'].shape, obs['agentview_image'].dtype))
    
    # Check for inconsistencies
    for key, values in shapes.items():
        # Get unique (shape[1:], dtype) combinations (ignore T dimension)
        unique = set((v[1][1:], str(v[2])) for v in values)
        if len(unique) > 1:
            print(f"⚠️  MISMATCH in {key}:")
            for demo_name, shape, dtype in values:
                print(f"    {demo_name}: {shape}, {dtype}")
        else:
            sample = values[0]
            print(f"✅ {key}: shape[1:]={sample[1][1:]}, dtype={sample[2]}")