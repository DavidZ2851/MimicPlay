import h5py
import numpy as np
import argparse


def check_actions(hdf5_path, max_demos=5, max_steps=10):
    with h5py.File(hdf5_path, "r") as f:
        # Print top-level keys
        print("=== Top-level keys ===")
        print(list(f.keys()))

        data_grp = f.get("data", f)  # fallback to root if no 'data' group

        demo_keys = sorted(data_grp.keys())
        print(f"\nTotal demos: {len(demo_keys)}")

        for demo_key in demo_keys[:max_demos]:
            demo = data_grp[demo_key]
            print(f"\n{'='*50}")
            print(f"Demo: {demo_key}")

            if "actions" in demo:
                actions = demo["actions"][:]
                print(f"  actions shape : {actions.shape}")
                print(f"  actions dtype : {actions.dtype}")
                print(f"  actions min   : {actions.min(axis=0)}")
                print(f"  actions max   : {actions.max(axis=0)}")
                print(f"  actions mean  : {actions.mean(axis=0)}")
                print(f"\n  First {min(max_steps, len(actions))} steps:")
                for i, a in enumerate(actions[:max_steps]):
                    print(f"    step {i:03d}: {np.array2string(a, precision=4, suppress_small=True)}")
            else:
                print("  No 'actions' key found. Available keys:", list(demo.keys()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect actions in an HDF5 dataset file.")
    parser.add_argument("hdf5_path", type=str, help="Path to the HDF5 file")
    parser.add_argument("--max_demos", type=int, default=5, help="Max number of demos to display (default: 5)")
    parser.add_argument("--max_steps", type=int, default=10, help="Max number of steps to print per demo (default: 10)")
    args = parser.parse_args()

    check_actions(args.hdf5_path, args.max_demos, args.max_steps)