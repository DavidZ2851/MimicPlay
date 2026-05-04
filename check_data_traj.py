import h5py
import numpy as np

dataset_path = "/home/haotian/MimicPlay/mimicplay/datasets/playdata/pick_place_red_cup_30_robot.hdf5"  # Change this

with h5py.File(dataset_path, 'r') as f:
    eef_pos = f['data/demo_0/obs/robot0_eef_pos'][:]
    future_traj = f['data/demo_0/obs/robot0_eef_pos_future_traj'][:]
    
    print(f"eef_pos shape: {eef_pos.shape}")
    print(f"future_traj shape: {future_traj.shape}")
    
    # Check first few frames
    print("\n" + "=" * 60)
    print("First 5 frames:")
    print("=" * 60)
    for t in range(5):
        print(f"\nt={t}:")
        print(f"  Current EEF pos: {eef_pos[t]}")
        print(f"  Future traj (10 points, reshaped):")
        traj = future_traj[t].reshape(10, 3)
        for i, pt in enumerate(traj):
            print(f"    Point {i+1}: {pt}")
    
    # Check consistency: future_traj[t] should match eef_pos at future indices
    print("\n" + "=" * 60)
    print("Consistency check (future_traj vs actual eef_pos):")
    print("=" * 60)
    
    t = 0  # Check frame 0
    traj = future_traj[t].reshape(10, 3)
    
    # Assuming POINT_GAP = 1 (indices: 1, 2, 3, ..., 10)
    # Or POINT_GAP = 2 (indices: 0, 2, 4, ..., 18)
    
    print(f"\nAt t={t}, comparing future_traj points with actual eef_pos:")
    
    # Try POINT_GAP = 1 pattern (1, 2, 3, ..., 10)
    print("\nAssuming POINT_GAP=1 (indices: t+1, t+2, ..., t+10):")
    for i in range(10):
        future_idx = min(t + i + 1, len(eef_pos) - 1)
        predicted = traj[i]
        actual = eef_pos[future_idx]
        diff = np.linalg.norm(predicted - actual)
        match = "✅" if diff < 0.001 else "❌"
        print(f"  Point {i+1} (t+{i+1}={future_idx}): diff={diff:.6f} {match}")
    
    # Try POINT_GAP = 2 pattern (0, 2, 4, ..., 18)
    print("\nAssuming POINT_GAP=2 (indices: t+0, t+2, t+4, ..., t+18):")
    for i in range(10):
        future_idx = min(t + i * 2, len(eef_pos) - 1)
        predicted = traj[i]
        actual = eef_pos[future_idx]
        diff = np.linalg.norm(predicted - actual)
        match = "✅" if diff < 0.001 else "❌"
        print(f"  Point {i+1} (t+{i*2}={future_idx}): diff={diff:.6f} {match}")
    
    # Value range
    print("\n" + "=" * 60)
    print("Value ranges:")
    print("=" * 60)
    print(f"eef_pos:")
    print(f"  min: {eef_pos.min(axis=0)}")
    print(f"  max: {eef_pos.max(axis=0)}")
    print(f"  range: {eef_pos.max(axis=0) - eef_pos.min(axis=0)}")
    
    print(f"\nfuture_traj:")
    print(f"  min: {future_traj.min()}")
    print(f"  max: {future_traj.max()}")