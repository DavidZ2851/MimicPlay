#!/usr/bin/env python3
"""
Merge human and robot HDF5 datasets for MimicPlay.

Usage:
    python merge_hdf5.py --human human.hdf5:50 --robot robot.hdf5:50 --output merged.hdf5
    python merge_hdf5.py --human human.hdf5:50 --robot robot.hdf5:50 --robot_val 6,7,8,9 --output merged.hdf5
"""

import h5py
import numpy as np
import argparse
import json
import random


def parse_input_spec(spec: str) -> tuple:
    if ":" in spec:
        path_str, count_str = spec.rsplit(":", 1)
        if count_str.lower() == "all":
            return path_str, None
        else:
            return path_str, int(count_str)
    else:
        return spec, None


def parse_demo_indices(spec: str) -> list:
    if not spec:
        return []
    indices = []
    for part in spec.replace(' ', '').split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            start, end = part.split('-')
            indices.extend(range(int(start), int(end) + 1))
        else:
            indices.append(int(part))
    return sorted(set(indices))


def copy_demo(src_file, dst_file, src_demo_name, dst_demo_name):
    src_demo = src_file[f'data/{src_demo_name}']
    dst_demo = dst_file.create_group(f'data/{dst_demo_name}')
    
    for key, val in src_demo.attrs.items():
        dst_demo.attrs[key] = val
    
    src_obs = src_demo['obs']
    
    actions = src_demo['actions'][:]
    if actions.shape[1] != 7:
        T = actions.shape[0]
        actions = np.zeros((T, 7), dtype=np.float32)
    else:
        actions = actions.astype(np.float32)
    dst_demo.create_dataset('actions', data=actions)
    
    dst_demo.create_dataset('dones', data=src_demo['dones'][:])
    dst_demo.create_dataset('rewards', data=src_demo['rewards'][:])
    dst_demo.create_dataset('states', data=src_demo['states'][:])
    
    dst_obs = dst_demo.create_group('obs')
    
    dst_obs.create_dataset('robot0_eef_pos', 
                           data=src_obs['robot0_eef_pos'][:].astype(np.float32))
    dst_obs.create_dataset('robot0_eef_pos_future_traj', 
                           data=src_obs['robot0_eef_pos_future_traj'][:].astype(np.float32))
    dst_obs.create_dataset('agentview_image', 
                           data=src_obs['agentview_image'][:].astype(np.uint8))
    
    return dst_demo.attrs.get('num_samples', src_demo['actions'].shape[0])


def merge_datasets(human_spec, robot_spec, robot_val_indices, output_path, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
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
        
        all_human_demos = sorted(f_human['data'].keys(), key=lambda x: int(x.split('_')[1]))
        all_robot_demos = sorted(f_robot['data'].keys(), key=lambda x: int(x.split('_')[1]))
        
        print(f"Available human demos: {len(all_human_demos)}")
        print(f"Available robot demos: {len(all_robot_demos)}")
        
        if num_human is not None:
            human_demos = all_human_demos[:num_human]
        else:
            human_demos = all_human_demos
        
        if num_robot is not None:
            robot_demos = all_robot_demos[:num_robot]
        else:
            robot_demos = all_robot_demos
        
        robot_indices = [int(d.split('_')[1]) for d in robot_demos]
        
        # Validate val indices (if any)
        for idx in robot_val_indices:
            if idx not in robot_indices:
                raise ValueError(f"robot_val index {idx} not in selected robot demos (0-{len(robot_demos)-1})")
        
        robot_train_indices = [i for i in robot_indices if i not in robot_val_indices]
        
        robot_train_demos = [f'demo_{i}' for i in robot_train_indices]
        robot_val_demos = [f'demo_{i}' for i in robot_val_indices]
        
        print(f"\nUsing human demos: {len(human_demos)}")
        print(f"Using robot demos: {len(robot_demos)} (train: {len(robot_train_demos)}, val: {len(robot_val_demos)})")
        if robot_val_indices:
            print(f"  Val indices: {robot_val_indices}")
        
        # Copy human demos
        print("\nCopying human demos...")
        for src_name in human_demos:
            dst_name = f'demo_{demo_counter}'
            n_samples = copy_demo(f_human, f_out, src_name, dst_name)
            train_demo_names.append(dst_name)
            total_samples += n_samples
            demo_counter += 1
        print(f"  Copied {len(human_demos)} human demos")
        
        # Copy robot train demos
        print("Copying robot train demos...")
        for src_name in robot_train_demos:
            dst_name = f'demo_{demo_counter}'
            n_samples = copy_demo(f_robot, f_out, src_name, dst_name)
            train_demo_names.append(dst_name)
            total_samples += n_samples
            demo_counter += 1
        print(f"  Copied {len(robot_train_demos)} robot train demos")
        
        # Copy robot val demos (if any)
        if robot_val_demos:
            print("Copying robot val demos...")
            for src_name in robot_val_demos:
                dst_name = f'demo_{demo_counter}'
                n_samples = copy_demo(f_robot, f_out, src_name, dst_name)
                val_demo_names.append(dst_name)
                demo_counter += 1
            print(f"  Copied {len(robot_val_demos)} robot val demos")
        
        # Create masks
        mask_grp = f_out.create_group('mask')
        mask_grp.create_dataset('train', data=[s.encode('utf-8') for s in train_demo_names])
        mask_grp.create_dataset('valid', data=[s.encode('utf-8') for s in val_demo_names])
        
        f_out['data'].attrs['total'] = total_samples
        
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
        print(f"  Total demos: {demo_counter}")
        print(f"  Training: {len(train_demo_names)} (human: {len(human_demos)}, robot: {len(robot_train_demos)})")
        print(f"  Validation: {len(val_demo_names)}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge human and robot HDF5 datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # No validation set
  python merge_hdf5.py --human human.hdf5:50 --robot robot.hdf5:50 --output merged.hdf5

  # With validation set
  python merge_hdf5.py --human human.hdf5:50 --robot robot.hdf5:50 --robot_val 6,7,8,9 --output merged.hdf5
        """
    )
    parser.add_argument("--human", type=str, required=True,
                        help="Human dataset (e.g., human.hdf5:50)")
    parser.add_argument("--robot", type=str, required=True,
                        help="Robot dataset (e.g., robot.hdf5:50)")
    parser.add_argument("--robot_val", type=str, default="",
                        help="Robot demo indices for validation (e.g., '6,7,8,9' or '6-9'). Optional.")
    parser.add_argument("--output", type=str, required=True, help="Output path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    robot_val_indices = parse_demo_indices(args.robot_val)
    
    merge_datasets(args.human, args.robot, robot_val_indices, args.output, args.seed)


if __name__ == "__main__":
    main()