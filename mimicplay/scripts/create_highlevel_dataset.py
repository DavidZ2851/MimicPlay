#!/usr/bin/env python3
"""
Create high-level dataset for MimicPlay by merging human and robot demos.
Randomly samples from both sources. Keeps only minimal obs:
  agentview_image, robot0_eef_pos, robot0_eef_pos_future_traj

Usage:
    python create_highlevel_dataset.py \
        --human human.hdf5 \
        --robot robot.hdf5 \
        --num_human 50 \
        --num_robot_train 25 \
        --num_robot_val 10 \
        --output highlevel.hdf5
"""

import h5py
import numpy as np
import argparse
import json
import random
import os


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def copy_demo(src_file, dst_file, src_demo_name, dst_demo_name):
    """
    Copy demo keeping only high-level obs:
      agentview_image, robot0_eef_pos, robot0_eef_pos_future_traj
    Actions are zeroed out if not 7-DOF.
    """
    src_demo = src_file[f'data/{src_demo_name}']
    dst_demo = dst_file.create_group(f'data/{dst_demo_name}')

    for key, val in src_demo.attrs.items():
        dst_demo.attrs[key] = val

    actions = src_demo['actions'][:]
    if actions.shape[1] != 7:
        T = actions.shape[0]
        actions = np.zeros((T, 7), dtype=np.float32)
    else:
        actions = actions.astype(np.float32)
    dst_demo.create_dataset('actions', data=actions)

    dst_demo.create_dataset('dones',   data=src_demo['dones'][:])
    dst_demo.create_dataset('rewards', data=src_demo['rewards'][:])
    dst_demo.create_dataset('states',  data=src_demo['states'][:])

    src_obs = src_demo['obs']
    dst_obs = dst_demo.create_group('obs')
    dst_obs.create_dataset('robot0_eef_pos',
                           data=src_obs['robot0_eef_pos'][:].astype(np.float32))
    dst_obs.create_dataset('robot0_eef_pos_future_traj',
                           data=src_obs['robot0_eef_pos_future_traj'][:].astype(np.float32))
    dst_obs.create_dataset('agentview_image',
                           data=src_obs['agentview_image'][:].astype(np.uint8))

    return dst_demo.attrs.get('num_samples', src_demo['actions'].shape[0])


def convert_to_native(obj):
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray):  return obj.tolist()
    if isinstance(obj, dict):        return {k: convert_to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):        return [convert_to_native(i) for i in obj]
    return obj


def save_summary(output_path, human_path, robot_path, demo_mapping, seed):
    summary_path = os.path.splitext(output_path)[0] + "_summary.json"
    summary = {
        "seed": seed,
        "sources": {"human": human_path, "robot": robot_path},
        "statistics": {
            "total_demos":       len(demo_mapping),
            "human_demos":       sum(1 for d in demo_mapping if d["source_type"] == "human"),
            "robot_train_demos": sum(1 for d in demo_mapping if d["source_type"] == "robot_train"),
            "robot_val_demos":   sum(1 for d in demo_mapping if d["source_type"] == "robot_val"),
        },
        "demo_mapping": demo_mapping,
    }
    with open(summary_path, 'w') as f:
        json.dump(convert_to_native(summary), f, indent=2)
    print(f"  Summary saved to: {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def create_highlevel_dataset(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    with h5py.File(args.human, 'r') as f:
        all_human_demos = sorted(f['data'].keys(), key=lambda x: int(x.split('_')[1]))
    with h5py.File(args.robot, 'r') as f:
        all_robot_demos = sorted(f['data'].keys(), key=lambda x: int(x.split('_')[1]))

    print(f"Available human demos : {len(all_human_demos)}")
    print(f"Available robot demos : {len(all_robot_demos)}")

    # Resolve defaults
    num_human       = args.num_human       if args.num_human       is not None else len(all_human_demos)
    num_robot_val   = args.num_robot_val   if args.num_robot_val   is not None else 0
    num_robot_train = args.num_robot_train if args.num_robot_train is not None else len(all_robot_demos) - num_robot_val

    # Validate counts
    if num_human > len(all_human_demos):
        raise ValueError(f"Requested {num_human} human demos but only {len(all_human_demos)} available.")

    robot_total = num_robot_train + num_robot_val
    if robot_total > len(all_robot_demos):
        raise ValueError(f"Requested {robot_total} robot demos but only {len(all_robot_demos)} available.")

    # Random sample
    human_demos       = random.sample(all_human_demos, num_human)
    robot_sampled     = random.sample(all_robot_demos, robot_total)
    robot_train_demos = robot_sampled[:num_robot_train]
    robot_val_demos   = robot_sampled[num_robot_train:]

    print(f"\nSampled human demos      : {human_demos}")
    print(f"Sampled robot train demos: {robot_train_demos}")
    print(f"Sampled robot val demos  : {robot_val_demos}")

    demo_mapping  = []
    train_names   = []
    val_names     = []
    total_samples = 0
    demo_counter  = 0

    with h5py.File(args.human, 'r') as f_human, \
         h5py.File(args.robot, 'r') as f_robot, \
         h5py.File(args.output, 'w') as f_out:

        f_out.create_group('data')

        # Preserve env_args from robot dataset
        if 'env_args' in f_robot['data'].attrs:
            f_out['data'].attrs['env_args'] = f_robot['data'].attrs['env_args']
        else:
            env_meta = {
                "env_name": "Franka_Pick_Place", "env_version": "1.0.0", "type": 1,
                "env_kwargs": {
                    "robots": ["Panda"],
                    "controller_configs": {"type": "OSC_POSE"},
                    "camera_names": ["agentview"],
                    "camera_heights": 84, "camera_widths": 84,
                }
            }
            f_out['data'].attrs['env_args'] = json.dumps(env_meta, indent=4)

        print("\nCopying human demos...")
        for src_name in human_demos:
            dst_name  = f'demo_{demo_counter}'
            n_samples = copy_demo(f_human, f_out, src_name, dst_name)
            train_names.append(dst_name)
            total_samples += n_samples
            demo_mapping.append({
                "output_demo": dst_name, "source_dataset": args.human,
                "source_demo": src_name, "source_type": "human",
                "split": "train", "num_samples": int(n_samples),
            })
            print(f"  {src_name} -> {dst_name}  ({n_samples} samples)")
            demo_counter += 1

        print("\nCopying robot train demos...")
        for src_name in robot_train_demos:
            dst_name  = f'demo_{demo_counter}'
            n_samples = copy_demo(f_robot, f_out, src_name, dst_name)
            train_names.append(dst_name)
            total_samples += n_samples
            demo_mapping.append({
                "output_demo": dst_name, "source_dataset": args.robot,
                "source_demo": src_name, "source_type": "robot_train",
                "split": "train", "num_samples": int(n_samples),
            })
            print(f"  {src_name} -> {dst_name}  ({n_samples} samples)")
            demo_counter += 1

        print("\nCopying robot val demos...")
        for src_name in robot_val_demos:
            dst_name  = f'demo_{demo_counter}'
            n_samples = copy_demo(f_robot, f_out, src_name, dst_name)
            val_names.append(dst_name)
            demo_mapping.append({
                "output_demo": dst_name, "source_dataset": args.robot,
                "source_demo": src_name, "source_type": "robot_val",
                "split": "valid", "num_samples": int(n_samples),
            })
            print(f"  {src_name} -> {dst_name}  ({n_samples} samples)")
            demo_counter += 1

        mask_grp = f_out.create_group('mask')
        mask_grp.create_dataset('train', data=[s.encode('utf-8') for s in train_names])
        mask_grp.create_dataset('valid', data=[s.encode('utf-8') for s in val_names])
        f_out['data'].attrs['total'] = total_samples

    print(f"\n{'='*60}")
    print(f"Done! Saved to {args.output}")
    print(f"  Human train  : {len(human_demos)}")
    print(f"  Robot train  : {len(robot_train_demos)}")
    print(f"  Robot val    : {len(robot_val_demos)}")
    print(f"  Total demos  : {demo_counter}")
    print(f"  Total steps  : {total_samples}")

    save_summary(args.output, args.human, args.robot, demo_mapping, args.seed)


def main():
    parser = argparse.ArgumentParser(description="Create high-level MimicPlay dataset (human + robot, minimal obs)")
    parser.add_argument("--human",           type=str, required=True, help="Human HDF5 path")
    parser.add_argument("--robot",           type=str, required=True, help="Robot HDF5 path")
    parser.add_argument("--num_human",       type=int, default=None, help="Number of human demos to randomly sample (default: all)")
    parser.add_argument("--num_robot_train", type=int, default=None, help="Number of robot train demos to randomly sample (default: all minus val)")
    parser.add_argument("--num_robot_val",   type=int, default=None, help="Number of robot val demos to randomly sample (default: 0)")
    parser.add_argument("--output",          type=str, required=True, help="Output HDF5 path")
    parser.add_argument("--seed",            type=int, default=42,    help="Random seed (default: 42)")
    args = parser.parse_args()
    create_highlevel_dataset(args)


if __name__ == "__main__":
    main()