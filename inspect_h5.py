#!/usr/bin/env python3
"""
HDF5 File Inspector
Usage: python inspect_h5.py <path_to_h5_file> [--demo demo_name]
"""

import h5py
import sys
import numpy as np
import argparse


def print_attrs(obj, indent=0):
    """Print attributes of an HDF5 object"""
    prefix = "  " * indent
    for key, val in obj.attrs.items():
        if isinstance(val, bytes):
            val = val.decode('utf-8', errors='replace')
        val_str = str(val)
        if len(val_str) > 200:
            val_str = val_str[:200] + "..."
        print(f"{prefix}  @{key}: {val_str}")


def summarize_array(arr):
    """Summarize array per dimension."""
    print(f"Shape: {arr.shape}, dtype: {arr.dtype}")
    print(f"\n{'Dim':<5} {'Min':>12} {'Max':>12} {'Mean':>12} {'Std':>12}")
    print("-" * 55)
    for i in range(arr.shape[1]):
        col = arr[:, i]
        print(f"{i:<5} {col.min():>12.4f} {col.max():>12.4f} {col.mean():>12.4f} {col.std():>12.4f}")
    
def explore_h5(obj, indent=0):
    """Recursively explore HDF5 structure"""
    prefix = "  " * indent

    if isinstance(obj, h5py.File):
        print(f"File: {obj.filename}")
        print_attrs(obj, indent)
        
        # Print action stats only once at file level
        if 'data' in obj and 'demo_0' in obj['data']:
            actions = obj['data']['demo_0']['actions'][:]
            print("\nSample actions stats (demo_0):")
            summarize_array(actions)
        
        for key in obj.keys():
            explore_h5(obj[key], indent)
    
    elif isinstance(obj, h5py.Group):
        print(f"{prefix}📁 Group: {obj.name}")
        print_attrs(obj, indent)
        for key in obj.keys():
            explore_h5(obj[key], indent + 1)
    
    elif isinstance(obj, h5py.Dataset):
        shape = obj.shape
        dtype = obj.dtype
        size_mb = np.prod(shape) * obj.dtype.itemsize / (1024 * 1024)
        print(f"{prefix}📊 Dataset: {obj.name}")
        print(f"{prefix}   shape: {shape}, dtype: {dtype}, size: {size_mb:.2f} MB")
        print_attrs(obj, indent)
        
        if np.prod(shape) < 10 and np.prod(shape) > 0:
            try:
                print(f"{prefix}   data: {obj[()]}")
            except:
                pass


def inspect_demo(filepath, demo_name):
    """Inspect a single demo"""
    with h5py.File(filepath, 'r') as f:
        demo_path = f"data/{demo_name}"
        if demo_path not in f:
            print(f"Error: Demo '{demo_name}' not found.")
            print(f"Available demos: {list(f['data'].keys())[:10]}...")
            return
        
        print("=" * 60)
        print(f"Demo: {demo_name}")
        print("=" * 60 + "\n")
        
        explore_h5(f[demo_path], indent=0)


def summary_h5(filepath):
    """Print a summary of the HDF5 file"""
    with h5py.File(filepath, 'r') as f:
        print("=" * 60)
        print(f"HDF5 File: {filepath}")
        print("=" * 60)
        
        n_groups = 0
        n_datasets = 0
        total_size = 0
        
        def count(name, obj):
            nonlocal n_groups, n_datasets, total_size
            if isinstance(obj, h5py.Group):
                n_groups += 1
            elif isinstance(obj, h5py.Dataset):
                n_datasets += 1
                total_size += np.prod(obj.shape) * obj.dtype.itemsize
        
        f.visititems(count)
        
        print(f"\nSummary:")
        print(f"  Groups: {n_groups}")
        print(f"  Datasets: {n_datasets}")
        print(f"  Total size: {total_size / (1024*1024):.2f} MB")
        print("\n" + "=" * 60)
        print("Structure:")
        print("=" * 60 + "\n")
        
        explore_h5(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect HDF5 file")
    parser.add_argument("filepath", help="Path to HDF5 file")
    parser.add_argument("--demo", type=str, default=None, help="Inspect single demo (e.g., demo_0)")
    
    args = parser.parse_args()
    
    try:
        if args.demo:
            inspect_demo(args.filepath, args.demo)
        else:
            summary_h5(args.filepath)
    except FileNotFoundError:
        print(f"Error: File not found: {args.filepath}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)