#!/usr/bin/env python3
"""
Merge human and robot HDF5 datasets for MimicPlay.

Usage:
    # Merge human + robot datasets
    python merge_hdf5.py merge --human human.hdf5:50 --robot robot.hdf5:25 --robot_val 100-104 --output merged.hdf5

    # Replicate from existing summary JSON:
    python merge_hdf5.py merge --human human.hdf5 --robot robot.hdf5 --human_json summary.json --robot_json summary.json --output merged.hdf5

    # Extract only robot_train demos from a summary JSON:
    python merge_hdf5.py extract_robot_train --json summary.json --output robot_train_only.hdf5
    python merge_hdf5.py extract_robot_train --json summary.json --output robot_train_only.hdf5 --robot /new/path/robot.hdf5
"""

import h5py
import numpy as np
import argparse
import json
import random
import os


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def parse_input_spec(spec: str) -> tuple:
    """
    Parse input spec. Supports:
      - path.hdf5           → (path, None, None)  # use all
      - path.hdf5:50        → (path, 50, None)    # random sample 50
      - path.hdf5:0,1,2,3   → (path, None, [0,1,2,3])  # specific indices
      - path.hdf5:0-4,10-14 → (path, None, [0,1,2,3,4,10,11,12,13,14])  # index ranges
    """
    if ":" in spec:
        path_str, count_str = spec.rsplit(":", 1)
        if count_str.lower() == "all":
            return path_str, None, None
        elif count_str.isdigit():
            return path_str, int(count_str), None
        else:
            indices = parse_demo_indices(count_str)
            return path_str, None, indices
    else:
        return spec, None, None


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


def load_indices_from_json(json_path: str, source_type: str) -> list:
    """
    Load demo indices from a summary JSON file.

    Args:
        json_path: Path to the summary JSON file
        source_type: One of 'human', 'robot_train', 'robot_val'

    Returns:
        List of demo indices (integers)
    """
    with open(json_path, 'r') as f:
        summary = json.load(f)

    indices = []
    for entry in summary.get("demo_mapping", []):
        if entry.get("source_type") == source_type:
            demo_name = entry.get("source_demo", "")
            if demo_name.startswith("demo_"):
                idx = int(demo_name.split("_")[1])
                indices.append(idx)

    return sorted(indices)


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

    dst_demo.create_dataset('dones',   data=src_demo['dones'][:])
    dst_demo.create_dataset('rewards', data=src_demo['rewards'][:])
    dst_demo.create_dataset('states',  data=src_demo['states'][:])

    dst_obs = dst_demo.create_group('obs')
    dst_obs.create_dataset('robot0_eef_pos',
                           data=src_obs['robot0_eef_pos'][:].astype(np.float32))
    dst_obs.create_dataset('robot0_eef_pos_future_traj',
                           data=src_obs['robot0_eef_pos_future_traj'][:].astype(np.float32))
    dst_obs.create_dataset('agentview_image',
                           data=src_obs['agentview_image'][:].astype(np.uint8))

    return dst_demo.attrs.get('num_samples', src_demo['actions'].shape[0])


def convert_to_native(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(i) for i in obj]
    return obj


def save_summary(output_path, human_path, robot_path, demo_mapping, seed):
    """Save a JSON summary of the dataset merge."""
    summary_path = os.path.splitext(output_path)[0] + "_summary.json"

    summary = {
        "seed": seed,
        "sources": {
            "human": human_path,
            "robot": robot_path,
        },
        "statistics": {
            "total_demos": len(demo_mapping),
            "human_demos": sum(1 for d in demo_mapping if d["source_type"] == "human"),
            "robot_train_demos": sum(1 for d in demo_mapping if d["source_type"] == "robot_train"),
            "robot_val_demos": sum(1 for d in demo_mapping if d["source_type"] == "robot_val"),
        },
        "demo_mapping": demo_mapping,
    }

    summary = convert_to_native(summary)

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  Summary saved to: {summary_path}")


# ---------------------------------------------------------------------------
# Command: merge
# ---------------------------------------------------------------------------

def merge_datasets(human_spec, robot_spec, robot_val_indices, output_path,
                   human_json=None, robot_json=None, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    human_path, num_human, human_indices = parse_input_spec(human_spec)
    robot_path, num_robot, robot_indices = parse_input_spec(robot_spec)

    if human_json:
        human_indices = load_indices_from_json(human_json, "human")
        print(f"Loaded {len(human_indices)} human demo indices from {human_json}")

    if robot_json:
        robot_indices = load_indices_from_json(robot_json, "robot_train")
        robot_val_indices = load_indices_from_json(robot_json, "robot_val")
        print(f"Loaded {len(robot_indices)} robot train indices from {robot_json}")
        print(f"Loaded {len(robot_val_indices)} robot val indices from {robot_json}")

    demo_mapping = []

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

        if human_indices is not None:
            human_demos = [f'demo_{i}' for i in human_indices]
        elif num_human is not None:
            human_demos = random.sample(all_human_demos, min(num_human, len(all_human_demos)))
        else:
            human_demos = all_human_demos

        if robot_indices is not None:
            robot_train_demos = [f'demo_{i}' for i in robot_indices]
        elif num_robot is not None:
            robot_train_demos = random.sample(all_robot_demos, min(num_robot, len(all_robot_demos)))
        else:
            robot_train_demos = all_robot_demos

        robot_val_demos = [f'demo_{i}' for i in robot_val_indices]

        for demo_name in human_demos:
            if demo_name not in all_human_demos:
                raise ValueError(f"human demo '{demo_name}' not found in human dataset")

        for demo_name in robot_train_demos + robot_val_demos:
            if demo_name not in all_robot_demos:
                raise ValueError(f"robot demo '{demo_name}' not found in robot dataset")

        print(f"\nUsing human demos: {len(human_demos)}")
        print(f"Using robot train demos: {len(robot_train_demos)}")
        if robot_val_indices:
            print(f"Using robot val demos: {len(robot_val_demos)} (indices: {robot_val_indices})")

        print("\nCopying human demos...")
        for src_name in human_demos:
            dst_name = f'demo_{demo_counter}'
            n_samples = copy_demo(f_human, f_out, src_name, dst_name)
            train_demo_names.append(dst_name)
            total_samples += n_samples
            demo_mapping.append({
                "output_demo": dst_name,
                "source_dataset": human_path,
                "source_demo": src_name,
                "source_type": "human",
                "split": "train",
                "num_samples": n_samples,
            })
            demo_counter += 1
        print(f"  Copied {len(human_demos)} human demos")

        print("Copying robot train demos...")
        for src_name in robot_train_demos:
            dst_name = f'demo_{demo_counter}'
            n_samples = copy_demo(f_robot, f_out, src_name, dst_name)
            train_demo_names.append(dst_name)
            total_samples += n_samples
            demo_mapping.append({
                "output_demo": dst_name,
                "source_dataset": robot_path,
                "source_demo": src_name,
                "source_type": "robot_train",
                "split": "train",
                "num_samples": n_samples,
            })
            demo_counter += 1
        print(f"  Copied {len(robot_train_demos)} robot train demos")

        if robot_val_demos:
            print("Copying robot val demos...")
            for src_name in robot_val_demos:
                dst_name = f'demo_{demo_counter}'
                n_samples = copy_demo(f_robot, f_out, src_name, dst_name)
                val_demo_names.append(dst_name)
                demo_mapping.append({
                    "output_demo": dst_name,
                    "source_dataset": robot_path,
                    "source_demo": src_name,
                    "source_type": "robot_val",
                    "split": "valid",
                    "num_samples": n_samples,
                })
                demo_counter += 1
            print(f"  Copied {len(robot_val_demos)} robot val demos")

        mask_grp = f_out.create_group('mask')
        mask_grp.create_dataset('train', data=[s.encode('utf-8') for s in train_demo_names])
        mask_grp.create_dataset('valid', data=[s.encode('utf-8') for s in val_demo_names])

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
        print(f"  Total demos: {demo_counter}")
        print(f"  Training: {len(train_demo_names)} (human: {len(human_demos)}, robot: {len(robot_train_demos)})")
        print(f"  Validation: {len(val_demo_names)}")

    save_summary(output_path, human_path, robot_path, demo_mapping, seed)


# ---------------------------------------------------------------------------
# Command: extract_robot_train
# ---------------------------------------------------------------------------

def extract_robot_train(json_path: str, output_path: str, robot_path: str = None):
    """
    Read a summary JSON and copy only robot_train demos into a new HDF5.

    Args:
        json_path:   Path to the summary JSON file
        output_path: Path for the output HDF5
        robot_path:  Optional override for the robot HDF5 path.
                     If None, uses the path from the JSON's sources.robot field.
    """
    with open(json_path, 'r') as f:
        summary = json.load(f)

    src_robot_path = robot_path or summary["sources"]["robot"]

    train_entries = [e for e in summary["demo_mapping"] if e["source_type"] == "robot_train"]

    if not train_entries:
        raise ValueError("No robot_train demos found in the JSON.")

    print(f"Found {len(train_entries)} robot_train demos to copy.")
    print(f"Source HDF5 : {src_robot_path}")
    print(f"Output HDF5 : {output_path}")

    demo_mapping  = []
    train_names   = []
    total_samples = 0

    with h5py.File(src_robot_path, 'r') as f_src, \
         h5py.File(output_path, 'w') as f_out:

        f_out.create_group('data')

        for new_idx, entry in enumerate(train_entries):
            src_demo = entry["source_demo"]
            dst_demo = f"demo_{new_idx}"

            n_samples = copy_demo(f_src, f_out, src_demo, dst_demo)
            train_names.append(dst_demo)
            total_samples += n_samples

            demo_mapping.append({
                "output_demo":    dst_demo,
                "source_dataset": src_robot_path,
                "source_demo":    src_demo,
                "source_type":    "robot_train",
                "split":          "train",
                "num_samples":    int(n_samples),
            })

            print(f"  {src_demo} -> {dst_demo}  ({n_samples} samples)")

        mask_grp = f_out.create_group('mask')
        mask_grp.create_dataset('train', data=[s.encode('utf-8') for s in train_names])
        mask_grp.create_dataset('valid', data=np.array([], dtype='S10'))

        f_out['data'].attrs['total'] = total_samples

        if 'env_args' in f_src['data'].attrs:
            f_out['data'].attrs['env_args'] = f_src['data'].attrs['env_args']

    print(f"\nDone! {len(train_names)} demos saved to: {output_path}")
    print(f"Total samples: {total_samples}")

    save_summary(output_path, "", src_robot_path, demo_mapping, seed=summary.get("seed", 0))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="HDF5 dataset tools for MimicPlay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- merge subcommand ---
    p_merge = subparsers.add_parser(
        "merge",
        help="Merge human and robot HDF5 datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python merge_hdf5.py merge --human human.hdf5:25 --robot robot.hdf5:25 --robot_val 100-104 --output merged.hdf5
  python merge_hdf5.py merge --human human.hdf5 --robot robot.hdf5 --human_json prev.json --robot_json prev.json --output merged.hdf5
        """
    )
    p_merge.add_argument("--human",      type=str, required=True, help="Human dataset (e.g. human.hdf5:50)")
    p_merge.add_argument("--robot",      type=str, required=True, help="Robot dataset (e.g. robot.hdf5:25)")
    p_merge.add_argument("--robot_val",  type=str, default="",    help="Robot demo indices for validation")
    p_merge.add_argument("--human_json", type=str, default=None,  help="JSON summary to load human demo indices from")
    p_merge.add_argument("--robot_json", type=str, default=None,  help="JSON summary to load robot train/val demo indices from")
    p_merge.add_argument("--output",     type=str, required=True, help="Output HDF5 path")
    p_merge.add_argument("--seed",       type=int, default=42,    help="Random seed")

    # --- extract_robot_train subcommand ---
    p_extract = subparsers.add_parser(
        "extract_robot_train",
        help="Extract only robot_train demos from a summary JSON into a new HDF5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python merge_hdf5.py extract_robot_train --json summary.json --output robot_train_only.hdf5
  python merge_hdf5.py extract_robot_train --json summary.json --output robot_train_only.hdf5 --robot /new/path/robot.hdf5
        """
    )
    p_extract.add_argument("--json",   type=str, required=True, help="Path to summary JSON file")
    p_extract.add_argument("--output", type=str, required=True, help="Output HDF5 path")
    p_extract.add_argument("--robot",  type=str, default=None,  help="Override robot HDF5 path (optional)")

    args = parser.parse_args()

    if args.command == "merge":
        robot_val_indices = parse_demo_indices(args.robot_val)
        merge_datasets(args.human, args.robot, robot_val_indices, args.output,
                       human_json=args.human_json, robot_json=args.robot_json, seed=args.seed)

    elif args.command == "extract_robot_train":
        extract_robot_train(args.json, args.output, robot_path=args.robot)


if __name__ == "__main__":
    main()