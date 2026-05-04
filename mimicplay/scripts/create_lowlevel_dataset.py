#!/usr/bin/env python3
"""
Create low-level dataset for MimicPlay using robot demos only.
Preserves ALL obs keys. Randomly samples train/val splits.

Usage:
    python create_lowlevel_dataset.py \
        --robot robot.hdf5 \
        --num_train 80 \
        --num_val 20 \
        --output lowlevel.hdf5
"""

import h5py
import numpy as np
import argparse
import json
import random
import os



def copy_demo_full(src_file, dst_file, src_demo_name, dst_demo_name):
    """Copy a demo preserving ALL datasets, groups, and attributes recursively."""
    src_demo = src_file[f'data/{src_demo_name}']
    dst_demo = dst_file.create_group(f'data/{dst_demo_name}')

    for k, v in src_demo.attrs.items():
        dst_demo.attrs[k] = v

    def _copy_group(src_grp, dst_grp):
        for key in src_grp.keys():
            item = src_grp[key]
            if isinstance(item, h5py.Dataset):
                ds = dst_grp.create_dataset(key, data=item[:])
                for ak, av in item.attrs.items():
                    ds.attrs[ak] = av
            elif isinstance(item, h5py.Group):
                grp = dst_grp.create_group(key)
                for ak, av in item.attrs.items():
                    grp.attrs[ak] = av
                _copy_group(item, grp)

    _copy_group(src_demo, dst_demo)
    return dst_demo.attrs.get('num_samples', src_demo['actions'].shape[0])


def convert_to_native(obj):
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray):  return obj.tolist()
    if isinstance(obj, dict):        return {k: convert_to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):        return [convert_to_native(i) for i in obj]
    return obj


def create_lowlevel_dataset(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    with h5py.File(args.robot, 'r') as f_src:
        all_demos = sorted(f_src['data'].keys(), key=lambda x: int(x.split('_')[1]))

    total_available = len(all_demos)
    num_val   = args.num_val   if args.num_val   is not None else 0
    num_train = args.num_train if args.num_train is not None else total_available - num_val

    total_requested = num_train + num_val
    print(f"Available demos : {total_available}")
    print(f"Requested       : {num_train} train + {num_val} val = {total_requested}")

    if total_requested > total_available:
        raise ValueError(
            f"Requested {total_requested} demos but only {total_available} available."
        )

    sampled     = random.sample(all_demos, total_requested)
    train_demos = sampled[:num_train]
    val_demos   = sampled[num_train:]

    print(f"\nSampled train: {train_demos}")
    print(f"Sampled val  : {val_demos}")

    demo_mapping  = []
    train_names   = []
    val_names     = []
    total_samples = 0
    demo_counter  = 0

    with h5py.File(args.robot, 'r') as f_src, \
         h5py.File(args.output, 'w') as f_out:

        f_out.create_group('data')

        for k, v in f_src['data'].attrs.items():
            f_out['data'].attrs[k] = v

        if 'env_args' not in f_src['data'].attrs:
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

        print("\nCopying train demos...")
        for src_name in train_demos:
            dst_name  = f'demo_{demo_counter}'
            n_samples = copy_demo_full(f_src, f_out, src_name, dst_name)
            train_names.append(dst_name)
            total_samples += n_samples
            demo_mapping.append({
                "output_demo": dst_name, "source_dataset": args.robot,
                "source_demo": src_name, "source_type": "robot_train",
                "split": "train", "num_samples": int(n_samples),
            })
            print(f"  {src_name} -> {dst_name}  ({n_samples} samples)")
            demo_counter += 1

        print("\nCopying val demos...")
        for src_name in val_demos:
            dst_name  = f'demo_{demo_counter}'
            n_samples = copy_demo_full(f_src, f_out, src_name, dst_name)
            val_names.append(dst_name)
            total_samples += n_samples
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

    summary_path = os.path.splitext(args.output)[0] + "_summary.json"
    summary = {
        "seed": args.seed,
        "sources": {"robot": args.robot},
        "statistics": {
            "total_demos":       demo_counter,
            "robot_train_demos": len(train_names),
            "robot_val_demos":   len(val_names),
        },
        "demo_mapping": demo_mapping,
    }
    with open(summary_path, 'w') as f:
        json.dump(convert_to_native(summary), f, indent=2)

    print(f"\n{'='*60}")
    print(f"Done! Saved to {args.output}")
    print(f"  Train        : {len(train_names)} demos")
    print(f"  Val          : {len(val_names)} demos")
    print(f"  Total steps  : {total_samples}")
    print(f"  Summary JSON : {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Create low-level MimicPlay dataset (robot only, all obs keys)")
    parser.add_argument("--robot",     type=str, required=True, help="Input robot HDF5 path")
    parser.add_argument("--num_train", type=int, default=None, help="Number of train demos to randomly sample (default: all minus val)")
    parser.add_argument("--num_val",   type=int, default=None, help="Number of val demos to randomly sample (default: 0)")
    parser.add_argument("--output",    type=str, required=True, help="Output HDF5 path")
    parser.add_argument("--seed",      type=int, default=42,    help="Random seed (default: 42)")
    args = parser.parse_args()
    create_lowlevel_dataset(args)


if __name__ == "__main__":
    main()