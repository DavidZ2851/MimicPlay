#!/usr/bin/env python3
"""
Convert robot episode data to robomimic/MimicPlay HDF5 format.

Input structure:
    /home/haotian/polaris/data/pick_place_red_mug_200/
    ├── episode_000031/
    │   ├── cam1.mp4
    │   ├── wrist_cam.mp4
    │   └── trajectory.npz
    ├── episode_000032/
    │   ├── cam1.mp4
    │   ├── wrist_cam.mp4
    │   └── trajectory.npz
    ...

Output: HDF5 file with structure matching robomimic format.
"""

import os
import glob
import argparse
import numpy as np
import h5py
import cv2
import json
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

GRIPPER_WIDTH = 0.05 # m, half of gripper width


PIXEL_SHIFT = 100 # TODO: shift 100 pixel to right, adjust this param
POINT_GAP = 1
FUTURE_POINTS_COUNT = 10

def compute_future_trajectory(eef_pos):
    """
    Compute future trajectory for each timestep.
    
    Args:
        eef_pos: (T, 3) end effector positions
    
    Returns:
        future_traj: (T, FUTURE_POINTS_COUNT * 3) = (T, 30)
    """
    future_traj_data = np.array([get_future_points(eef_pos[j:]) for j in range(len(eef_pos))])
    return future_traj_data.astype(np.float64)

def get_future_points(arr):
    future_traj = []

    for i in range(POINT_GAP, (FUTURE_POINTS_COUNT + 1) * POINT_GAP, POINT_GAP):
        # Identify the indices for the current and prior points
        index_current = min(len(arr) - 1, i)

        current_point = arr[index_current]
        future_traj.extend(current_point)

    return future_traj

def center_crop_square(frame, pixel_shift=0):
    """Center-crop a frame to a square based on the smaller side."""
    h, w = frame.shape[:2]
    side = min(h, w)
    y1 = (h - side) // 2
    x1 = (w - side) // 2 + pixel_shift
    return frame[y1:y1 + side, x1:x1 + side]


def extract_frames_from_video(video_path, num_frames, target_size=(84, 84), pixel_shift=0):
    """
    Extract frames from video, crop to square, and resize.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract (should match trajectory length)
        target_size: (width, height) to resize to
        pixel_shift: Horizontal pixel shift for center crop
    
    Returns:
        numpy array of shape (num_frames, H, W, 3) with uint8 dtype
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame indices to sample (evenly spaced)
    if total_video_frames >= num_frames:
        frame_indices = np.linspace(0, total_video_frames - 1, num_frames, dtype=int)
    else:
        # If video has fewer frames than trajectory, repeat last frame
        frame_indices = list(range(total_video_frames)) + [total_video_frames - 1] * (num_frames - total_video_frames)
    
    frames = []
    current_idx = 0
    
    for target_idx in frame_indices:
        # Seek to target frame if needed
        while current_idx < target_idx:
            cap.read()
            current_idx += 1
        
        ret, frame = cap.read()
        if not ret:
            # If we can't read, duplicate last frame
            if frames:
                frames.append(frames[-1].copy())
            else:
                raise ValueError(f"Cannot read frame {target_idx} from {video_path}")
        else:
            current_idx += 1
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Crop to square
            frame = center_crop_square(frame, pixel_shift=pixel_shift)
            # Resize
            frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
            frames.append(frame)
    
    cap.release()
    return np.array(frames, dtype=np.uint8)


def process_episode(episode_dir, target_size=(84, 84)):
    """
    Process a single episode directory.
    
    Args:
        episode_dir: Path to episode directory
        target_size: Image size after resize
    
    Returns:
        dict with all episode data
    """
    # Load trajectory
    traj_path = os.path.join(episode_dir, "trajectory.npz")
    traj_data = np.load(traj_path)
    
    states_ee = traj_data['states_ee'].astype(np.float64)[:-1]  # (T-1, 8)
    gripper = traj_data['gripper_width'].astype(np.float64)[:-1] # (T-1, 1)
    delta_action_ee = traj_data['delta_action'].astype(np.float64)  # (T-1, 7)
    
    T = states_ee.shape[0]
    
    # Extract agentview images (cam1) with pixel shift
    video_path = os.path.join(episode_dir, "cam1.mp4")
    images = extract_frames_from_video(video_path, T, target_size, pixel_shift=PIXEL_SHIFT)
    
    # Extract eye-in-hand images (wrist_cam) with pixel shift
    wrist_video_path = os.path.join(episode_dir, "wrist_cam.mp4")
    if os.path.exists(wrist_video_path):
        wrist_images = extract_frames_from_video(wrist_video_path, T, target_size, pixel_shift=PIXEL_SHIFT)
    else:
        print(f"Warning: wrist_cam.mp4 not found in {episode_dir}, using zeros")
        wrist_images = np.zeros((T, target_size[1], target_size[0], 3), dtype=np.uint8)
    
    # Parse states_ee: position(3) + quaternion(4) + gripper(1)
    eef_pos = states_ee[:, :3]      # (T-1, 3)
    eef_quat_wxyz = states_ee[:, 3:7]    # (T-1, 4)
    eef_quat = eef_quat_wxyz[:, [1, 2, 3, 0]] # wxyz -> xyzw

    eef_pos_future_traj = compute_future_trajectory(eef_pos)

    gripper = np.concatenate([gripper, -gripper], axis=-1) # T-1, 2
    
    actions = delta_action_ee  # (T-1 , 7) - pos + axis_angle + gripper 
    
    # Create dones array
    dones = np.zeros(T, dtype=np.int64)
    dones[-1] = 1  # Last step is done
    
    rewards = np.zeros(T, dtype=np.float64)
    rewards[-1] = 1.0  # Assume success at end
    
    return {
        'actions': actions, # T-1, 
        'dones': dones, # T-1, 
        'rewards': rewards, # T-1, 
        'states': states_ee, # T-1, 8
        'obs': {
            'agentview_image': images, # T-1, H, W, 3
            'robot0_eye_in_hand_image': wrist_images,  # T-1, H, W, 3
            'robot0_eef_pos': eef_pos, # T-1, 3
            'robot0_eef_quat': eef_quat, # T-1, 4
            'robot0_gripper_qpos': gripper, # T-1, 2
            'robot0_eef_pos_future_traj': eef_pos_future_traj,
        }
    }

def print_ep_data(ep_data):
    """Print episode data structure."""
    print(f"Keys: {list(ep_data.keys())}")
    
    for k, v in ep_data.items():
        if k == 'obs':
            print(f"\nObs keys:")
            for obs_k, obs_v in v.items():
                if hasattr(obs_v, 'shape'):
                    print(f"  {obs_k}: {obs_v.shape}, dtype={obs_v.dtype}")
                else:
                    print(f"  {obs_k}: {type(obs_v)}")
        else:
            if hasattr(v, 'shape'):
                print(f"{k}: {v.shape}, dtype={v.dtype}")
            else:
                print(f"{k}: {type(v)}")


def convert_to_hdf5(data_dir, output_path, target_size=(84, 84), val_ratio=0):
    """
    Convert all episodes to a single HDF5 file.
    
    Args:
        data_dir: Directory containing episode_* folders
        output_path: Output HDF5 file path
        target_size: Image size (width, height)
        val_ratio: Fraction of episodes for training
    """
    # Find all episode directories
    episode_dirs = sorted(glob.glob(os.path.join(data_dir, "episode_*")))
    
    if not episode_dirs:
        raise ValueError(f"No episode directories found in {data_dir}")
    
    print(f"Found {len(episode_dirs)} episodes")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    # Create HDF5 file
    with h5py.File(output_path, 'w') as f:
        # Create data group
        data_grp = f.create_group("data")
        
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
        data_grp.attrs['env_args'] = json.dumps(env_meta, indent=4)

        # Process each episode
        demo_names = []
        total_samples = 0
        
        for i, episode_dir in enumerate(tqdm(episode_dirs, desc="Processing episodes")):
            try:
                ep_data = process_episode(episode_dir, target_size)
                print_ep_data(ep_data)
            except Exception as e:
                print(f"\nWarning: Failed to process {episode_dir}: {e}")
                continue
            
            demo_name = f"demo_{i}"
            demo_names.append(demo_name)
            demo_grp = data_grp.create_group(demo_name)
            
            # Store num_samples as attribute
            num_samples = ep_data['actions'].shape[0]
            demo_grp.attrs['num_samples'] = num_samples
            total_samples += num_samples
            
            # Store actions, dones, rewards, states
            demo_grp.create_dataset("actions", data=ep_data['actions'])
            demo_grp.create_dataset("dones", data=ep_data['dones'])
            demo_grp.create_dataset("rewards", data=ep_data['rewards'])
            demo_grp.create_dataset("states", data=ep_data['states'])
            
            # Store observations
            obs_grp = demo_grp.create_group("obs")
            for obs_key, obs_data in ep_data['obs'].items():
                obs_grp.create_dataset(obs_key, data=obs_data)
        
        # Store global attributes
        data_grp.attrs['total'] = total_samples
        
        # Create train/valid masks
        n_demos = len(demo_names)
        n_val = max(0, int(n_demos * val_ratio))
        n_train = n_demos - n_val
        
        mask_grp = f.create_group("mask")
        train_demos = [d.encode('utf-8') for d in demo_names[:n_train]]
        valid_demos = [d.encode('utf-8') for d in demo_names[n_train:]]
        
        mask_grp.create_dataset("train", data=train_demos)
        mask_grp.create_dataset("valid", data=valid_demos)
    

    
    print(f"\nDone! Created {output_path}")
    print(f"  Total episodes: {len(demo_names)}")
    print(f"  Total samples: {total_samples}")
    print(f"  Train: {len(train_demos)}, Valid: {len(valid_demos)}")


def main():
    parser = argparse.ArgumentParser(description="Convert robot data to HDF5 format")
    parser.add_argument("--data_dir", type=str, 
                        default="/home/haotian/polaris/data/pick_place_red_mug_30_mimicplay",
                        help="Directory containing episode_* folders")
    parser.add_argument("--output", type=str,
                        default="/home/haotian/MimicPlay/mimicplay/datasets/playdata/robot_data.hdf5",
                        help="Output HDF5 file path")
    parser.add_argument("--image_size", type=int, default=84,
                        help="Image size after crop and resize")
    parser.add_argument("--val_ratio", type=float, default=0.0,
                        help="Fraction of episodes for training")
    
    args = parser.parse_args()
    
    convert_to_hdf5(
        data_dir=args.data_dir,
        output_path=args.output,
        target_size=(args.image_size, args.image_size),
        val_ratio=args.val_ratio
    )


if __name__ == "__main__":
    main()