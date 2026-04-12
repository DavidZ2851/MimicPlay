#!/usr/bin/env python3
"""
Merge human and robot HDF5 datasets for MimicPlay.

Human demos: used for training only (playdata)
Robot demos: used for training + validation (robot data)

Usage:
    python merge_hdf5.py --human human.hdf5:50 --robot robot.hdf5:30 --output merged.hdf5 --n_val 5
    python merge_hdf5.py --human human.hdf5 --robot robot.hdf5:20 --output merged.hdf5
    python merge_hdf5.py --human human.hdf5:all --robot robot.hdf5:10 --output merged.hdf5
"""

import h5py
import numpy as np
import argparse
import json
import random


def parse_input_spec(spec: str) -> tuple:
    """
    Parse input specification like 'path.hdf5:10' or 'path.hdf5'.
    Returns (path, num_demos) where num_demos is None for 'all'.
    """
    if ":" in spec:
        path_str, count_str = spec.rsplit(":", 1)
        if count_str.lower() == "all":
            return path_str, None
        else:
            return path_str, int(count_str)
    else:
        return spec, None  # None means all


def copy_demo(src_file, dst_file, src_demo_name, dst_demo_name):
    """Copy only the required obs fields with consistent dtypes."""
    src_demo = src_file[f'data/{src_demo_name}']
    dst_demo = dst_file.create_group(f'data/{dst_demo_name}')
    
    # Copy attributes
    for key, val in src_demo.attrs.items():
        dst_demo.attrs[key] = val
    
    src_obs = src_demo['obs']
    
    # Copy actions - normalize to (T, 7)
    actions = src_demo['actions'][:]
    if actions.shape[1] != 7:
        # Human actions are (T, 30) - replace with zeros (T, 7)
        T = actions.shape[0]
        actions = np.zeros((T, 7), dtype=np.float32)
    else:
        actions = actions.astype(np.float32)
    dst_demo.create_dataset('actions', data=actions)
    
    # Copy dones, rewards, states
    dst_demo.create_dataset('dones', data=src_demo['dones'][:])
    dst_demo.create_dataset('rewards', data=src_demo['rewards'][:])
    dst_demo.create_dataset('states', data=src_demo['states'][:])
    
    # Copy ONLY required obs with consistent dtypes
    dst_obs = dst_demo.create_group('obs')
    
    # float32 for all low_dim
    dst_obs.create_dataset('robot0_eef_pos', 
                           data=src_obs['robot0_eef_pos'][:].astype(np.float32))
    dst_obs.create_dataset('robot0_eef_pos_future_traj', 
                           data=src_obs['robot0_eef_pos_future_traj'][:].astype(np.float32))
    
    # uint8 for images
    dst_obs.create_dataset('agentview_image', 
                           data=src_obs['agentview_image'][:].astype(np.uint8))
    
    return dst_demo.attrs.get('num_samples', src_demo['actions'].shape[0])


def merge_datasets(human_spec, robot_spec, output_path, n_val=5, seed=42):
    """
    Merge human and robot datasets.
    
    - All human demos go to training
    - Robot demos split into training + validation based on n_val
    """
    
    random.seed(seed)
    np.random.seed(seed)
    
    # Parse input specs
    human_path, num_human = parse_input_spec(human_spec)
    robot_path, num_robot = parse_input_spec(robot_spec)
    
    with h5py.File(human_path, 'r') as f_human, \
         h5py.File(robot_path, 'r') as f_robot, \
         h5py.File(output_path, 'w') as f_out:
        
        f_out.create_group('data')
        
        demo_counter = 0
        total_samples = 0
        train_demo_names = []
        val_demo_names = []
        
        # Get all demo names
        all_human_demos = sorted(f_human['data'].keys(), key=lambda x: int(x.split('_')[1]))
        all_robot_demos = sorted(f_robot['data'].keys(), key=lambda x: int(x.split('_')[1]))
        
        print(f"Available human demos: {len(all_human_demos)}")
        print(f"Available robot demos: {len(all_robot_demos)}")
        
        # Select human demos
        if num_human is not None and num_human < len(all_human_demos):
            human_demos = random.sample(all_human_demos, num_human)
            human_demos = sorted(human_demos, key=lambda x: int(x.split('_')[1]))
        else:
            human_demos = all_human_demos
        
        # Select robot demos
        if num_robot is not None and num_robot < len(all_robot_demos):
            robot_demos = random.sample(all_robot_demos, num_robot)
            robot_demos = sorted(robot_demos, key=lambda x: int(x.split('_')[1]))
        else:
            robot_demos = all_robot_demos
        
        print(f"\nUsing human demos: {len(human_demos)}/{len(all_human_demos)}")
        print(f"Using robot demos: {len(robot_demos)}/{len(all_robot_demos)}")
        
        # Copy human demos (training only)
        print("\nCopying human demos (all to training)...")
        for src_name in human_demos:
            dst_name = f'demo_{demo_counter}'
            n_samples = copy_demo(f_human, f_out, src_name, dst_name)
            train_demo_names.append(dst_name)
            total_samples += n_samples
            demo_counter += 1
            print(f"  {src_name} -> {dst_name} ({n_samples} samples)")
        
        # Split robot demos into train/val
        n_robot = len(robot_demos)
        n_val = max(0, n_val)
        n_train = n_robot - n_val
        
        # Shuffle before splitting
        robot_demos_shuffled = robot_demos.copy()
        random.shuffle(robot_demos_shuffled)
        
        robot_train_demos = robot_demos_shuffled[:n_train]
        robot_val_demos = robot_demos_shuffled[n_train:]
        
        print(f"\nCopying robot demos (train: {n_train}, val: {n_val})...")
        
        for src_name in robot_train_demos:
            dst_name = f'demo_{demo_counter}'
            n_samples = copy_demo(f_robot, f_out, src_name, dst_name)
            train_demo_names.append(dst_name)
            total_samples += n_samples
            demo_counter += 1
            print(f"  [TRAIN] {src_name} -> {dst_name} ({n_samples} samples)")
        
        for src_name in robot_val_demos:
            dst_name = f'demo_{demo_counter}'
            n_samples = copy_demo(f_robot, f_out, src_name, dst_name)
            val_demo_names.append(dst_name)
            demo_counter += 1
            print(f"  [VAL] {src_name} -> {dst_name} ({n_samples} samples)")
        
        # Create masks
        mask_grp = f_out.create_group('mask')
        mask_grp.create_dataset('train', data=[s.encode('utf-8') for s in train_demo_names])
        mask_grp.create_dataset('valid', data=[s.encode('utf-8') for s in val_demo_names])
        
        # Set attributes
        f_out['data'].attrs['total'] = total_samples
        
        if 'env_args' in f_robot['data'].attrs:
            f_out['data'].attrs['env_args'] = f_robot['data'].attrs['env_args']
        else:
            env_meta = {
                "env_name": "Franka_Pick_Place",
                "env_version": "1.0.0",
                "type": 1,
                "env_kwargs": {
                    "robots": ["Panda"],
                    "controller_configs": {"type": "OSC_POSE"},
                    "camera_names": ["agentview"],
                    "camera_heights": 84,
                    "camera_widths": 84,
                }
            }
            f_out['data'].attrs['env_args'] = json.dumps(env_meta, indent=4)
        
        print("\n" + "=" * 60)
        print(f"Done! Saved to {output_path}")
        print(f"  Seed: {seed}")
        print(f"  Total demos: {demo_counter}")
        print(f"  Training demos: {len(train_demo_names)} (human: {len(human_demos)}, robot: {n_train})")
        print(f"  Validation demos: {len(val_demo_names)} (robot only)")
        print(f"  Total training samples: {total_samples}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge human and robot HDF5 datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use 50 human demos and 30 robot demos
  python merge_hdf5.py --human human.hdf5:50 --robot robot.hdf5:30 --output merged.hdf5

  # Use all human demos and 20 robot demos
  python merge_hdf5.py --human human.hdf5 --robot robot.hdf5:20 --output merged.hdf5

  # Explicitly specify 'all'
  python merge_hdf5.py --human human.hdf5:all --robot robot.hdf5:10 --output merged.hdf5

  # With custom n_val and seed
  python merge_hdf5.py --human human.hdf5:50 --robot robot.hdf5:30 --output merged.hdf5 --n_val 0.2 --seed 123
        """
    )
    parser.add_argument("--human", type=str, required=True,
                        help="Human dataset path with optional count (e.g., human.hdf5:50)")
    parser.add_argument("--robot", type=str, required=True,
                        help="Robot dataset path with optional count (e.g., robot.hdf5:30)")
    parser.add_argument("--output", type=str, required=True, help="Output path")
    parser.add_argument("--n_val", type=int, default=5,
                        help="Number of robot demos for validation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    
    args = parser.parse_args()
    
    merge_datasets(args.human, args.robot, args.output, args.n_val, args.seed)


if __name__ == "__main__":
    main()