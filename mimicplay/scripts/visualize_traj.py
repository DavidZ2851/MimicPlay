#!/usr/bin/env python3
"""
Visualize HDF5 dataset with future trajectory projected onto agentview images,
and wrist camera view side by side.

Usage:
    python scripts/visualize_traj.py --dataset converted_robot_dataset.hdf5 --calib cam_calibration.json --demo demo_0 --output output_video.mp4
"""

import os
import json
import argparse
import numpy as np
import h5py
import cv2
from tqdm import tqdm


def load_calibration(calib_path, camera_name="cam1"):
    """
    Load camera calibration from JSON file.
    
    Args:
        calib_path: Path to calibration JSON file
        camera_name: Camera name (cam0 or cam1)
    
    Returns:
        intrinsic: (3, 3) camera intrinsic matrix
        extrinsic: (4, 4) camera extrinsic matrix (cam to world)
        distortion: distortion coefficients
    """
    with open(calib_path, 'r') as f:
        calib = json.load(f)
    
    cam_calib = calib[camera_name]
    intrinsic = np.array(cam_calib['intrinsic'])
    extrinsic = np.array(cam_calib['extrinsic'])  # cam to world
    distortion = np.array(cam_calib['distortion']) if 'distortion' in cam_calib else None
    
    return intrinsic, extrinsic, distortion


def world_to_camera(points_world, extrinsic_cam_to_world):
    """
    Transform points from world frame to camera frame.
    
    Args:
        points_world: (N, 3) points in world frame
        extrinsic_cam_to_world: (4, 4) transformation matrix (cam to world)
    
    Returns:
        points_cam: (N, 3) points in camera frame
    """
    # Invert extrinsic to get world to camera
    extrinsic_world_to_cam = np.linalg.inv(extrinsic_cam_to_world)
    
    # Convert to homogeneous coordinates
    N = points_world.shape[0]
    points_homo = np.hstack([points_world, np.ones((N, 1))])  # (N, 4)
    
    # Transform
    points_cam_homo = (extrinsic_world_to_cam @ points_homo.T).T  # (N, 4)
    points_cam = points_cam_homo[:, :3]
    
    return points_cam


def project_to_image(points_cam, intrinsic, image_size=None):
    """
    Project 3D points in camera frame to 2D image coordinates.
    
    Args:
        points_cam: (N, 3) points in camera frame
        intrinsic: (3, 3) camera intrinsic matrix
        image_size: (width, height) to clip points, or None
    
    Returns:
        points_2d: (N, 2) pixel coordinates
        valid_mask: (N,) boolean mask for points in front of camera
    """
    # Check if points are in front of camera (z > 0)
    valid_mask = points_cam[:, 2] > 0.01
    
    # Project to image plane
    points_2d_homo = (intrinsic @ points_cam.T).T  # (N, 3)
    
    # Normalize by z
    points_2d = np.zeros((points_cam.shape[0], 2))
    points_2d[valid_mask, 0] = points_2d_homo[valid_mask, 0] / points_2d_homo[valid_mask, 2]
    points_2d[valid_mask, 1] = points_2d_homo[valid_mask, 1] / points_2d_homo[valid_mask, 2]
    
    # Clip to image bounds if provided
    if image_size is not None:
        w, h = image_size
        valid_mask &= (points_2d[:, 0] >= 0) & (points_2d[:, 0] < w)
        valid_mask &= (points_2d[:, 1] >= 0) & (points_2d[:, 1] < h)
    
    return points_2d, valid_mask


def draw_trajectory_on_image(image, points_2d, valid_mask, color_start=(0, 255, 0), color_end=(255, 0, 0)):
    """
    Draw trajectory points and lines on image.
    
    Args:
        image: (H, W, 3) image
        points_2d: (N, 2) pixel coordinates
        valid_mask: (N,) boolean mask
        color_start: BGR color for first point
        color_end: BGR color for last point
    
    Returns:
        image with trajectory drawn
    """
    img = image.copy()
    n_points = len(points_2d)
    
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) == 0:
        return img
    
    # Draw lines between consecutive valid points
    for i in range(len(valid_indices) - 1):
        idx1 = valid_indices[i]
        idx2 = valid_indices[i + 1]
        
        pt1 = tuple(points_2d[idx1].astype(int))
        pt2 = tuple(points_2d[idx2].astype(int))
        
        # Interpolate color
        t = idx1 / max(n_points - 1, 1)
        color = tuple(int(color_start[j] * (1 - t) + color_end[j] * t) for j in range(3))
        
        cv2.line(img, pt1, pt2, color, 2)
    
    # Draw points
    for i, idx in enumerate(valid_indices):
        pt = tuple(points_2d[idx].astype(int))
        
        # Interpolate color
        t = idx / max(n_points - 1, 1)
        color = tuple(int(color_start[j] * (1 - t) + color_end[j] * t) for j in range(3))
        
        cv2.circle(img, pt, 4, color, -1)
        cv2.circle(img, pt, 4, (255, 255, 255), 1)
    
    return img


def visualize_demo(dataset_path, calib_path, demo_name, output_path=None, 
                   camera_name="cam1", fps=20, show=False, display_scale=10):
    """
    Visualize a demo with future trajectory projected onto images.
    """
    # Load calibration
    intrinsic, extrinsic, distortion = load_calibration(calib_path, camera_name)
    
    # Open dataset
    with h5py.File(dataset_path, 'r') as f:
        demo = f[f'data/{demo_name}']
        
        # Load data
        images = demo['obs/agentview_image'][:]  # (T, H, W, 3)
        eef_pos = demo['obs/robot0_eef_pos'][:]  # (T, 3)
        future_traj = demo['obs/robot0_eef_pos_future_traj'][:]  # (T, 30)
        
        # Load wrist camera if available
        has_wrist = 'obs/robot0_eye_in_hand_image' in demo
        if has_wrist:
            wrist_images = demo['obs/robot0_eye_in_hand_image'][:]  # (T, H, W, 3)
        else:
            print("Warning: No wrist camera images found in dataset")
            wrist_images = None
        
        T, H, W, C = images.shape
        image_size = (W, H)
        display_W = W * display_scale
        display_H = H * display_scale
        
        # Combined frame width (agentview + wrist side by side)
        if has_wrist:
            combined_W = display_W * 2
        else:
            combined_W = display_W
        combined_H = display_H
        
        # Scale intrinsic for original image
        ORIGINAL_W, ORIGINAL_H = 1280, 720
        PIXEL_SHIFT = 100
        
        side = min(ORIGINAL_H, ORIGINAL_W)  # 720
        crop_x1 = (ORIGINAL_W - side) // 2 + PIXEL_SHIFT
        crop_y1 = (ORIGINAL_H - side) // 2
        scale = W / side
        
        intrinsic_scaled = intrinsic.copy()
        intrinsic_scaled[0, 0] = intrinsic[0, 0] * scale  # fx
        intrinsic_scaled[1, 1] = intrinsic[1, 1] * scale  # fy
        intrinsic_scaled[0, 2] = (intrinsic[0, 2] - crop_x1) * scale  # cx
        intrinsic_scaled[1, 2] = (intrinsic[1, 2] - crop_y1) * scale  # cy
        
        print(f"Processing {demo_name}: {T} frames")
        print(f"Image size: {W}x{H}, Display: {display_W}x{display_H}")
        print(f"Combined frame: {combined_W}x{combined_H}")
        print(f"Wrist camera: {'Yes' if has_wrist else 'No'}")
        print(f"Scaled intrinsic:\n{intrinsic_scaled}")
        
        # Setup video writer with combined size
        video_writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (combined_W, combined_H))
        
        for t in tqdm(range(T), desc="Rendering frames"):
            # Get current frame (RGB -> BGR for OpenCV)
            frame = cv2.cvtColor(images[t], cv2.COLOR_RGB2BGR)
            
            # Upscale frame
            frame = cv2.resize(frame, (display_W, display_H), interpolation=cv2.INTER_NEAREST)
            
            # Get future trajectory points (reshape from 30 to 10x3)
            future_points = future_traj[t].reshape(-1, 3)  # (10, 3)
            
            # Add current position as first point
            all_points = np.vstack([eef_pos[t].reshape(1, 3), future_points])  # (11, 3)
            
            # Transform to camera frame
            points_cam = world_to_camera(all_points, extrinsic)
            
            # Project to image (in original 84x84 space)
            points_2d, valid_mask = project_to_image(points_cam, intrinsic_scaled, image_size)
            
            # Scale points to display size
            points_2d_scaled = points_2d * display_scale
            
            # Draw trajectory on upscaled frame
            frame = draw_trajectory_on_image(frame, points_2d_scaled, valid_mask,
                                            color_start=(0, 255, 0),   # Green (current)
                                            color_end=(0, 0, 255))     # Red (future)
            
            # Draw current EEF position
            eef_cam = world_to_camera(eef_pos[t].reshape(1, 3), extrinsic)
            eef_2d, eef_valid = project_to_image(eef_cam, intrinsic_scaled, image_size)
            if eef_valid[0]:
                pt = tuple((eef_2d[0] * display_scale).astype(int))
                cv2.circle(frame, pt, 8, (0, 255, 255), -1)  # Yellow
                cv2.circle(frame, pt, 8, (0, 0, 0), 2)
            
            # Add text overlay on agentview
            cv2.putText(frame, f"Agentview - Frame: {t}/{T}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"EEF: [{eef_pos[t, 0]:.3f}, {eef_pos[t, 1]:.3f}, {eef_pos[t, 2]:.3f}]",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Process wrist camera
            if has_wrist:
                wrist_frame = cv2.cvtColor(wrist_images[t], cv2.COLOR_RGB2BGR)
                wrist_frame = cv2.resize(wrist_frame, (display_W, display_H), interpolation=cv2.INTER_NEAREST)
                
                # Add text overlay on wrist
                cv2.putText(wrist_frame, "Wrist Camera", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Combine frames side by side
                combined_frame = np.hstack([frame, wrist_frame])
            else:
                combined_frame = frame
            
            # Write frame
            if video_writer:
                video_writer.write(combined_frame)
            
            # Show frame
            if show:
                cv2.imshow('Trajectory Visualization', combined_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        if video_writer:
            video_writer.release()
            print(f"Saved video to {output_path}")
        
        if show:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Visualize HDF5 dataset with future trajectory")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to HDF5 dataset")
    parser.add_argument("--calib", type=str, required=True,
                        help="Path to camera calibration JSON")
    parser.add_argument("--demo", type=str, default="demo_0",
                        help="Demo name to visualize")
    parser.add_argument("--output", type=str, default="traj_visualization.mp4",
                        help="Output video path")
    parser.add_argument("--camera", type=str, default="cam1",
                        help="Camera name in calibration file")
    parser.add_argument("--fps", type=int, default=20,
                        help="Output video FPS")
    parser.add_argument("--show", action="store_true",
                        help="Show frames in window")
    parser.add_argument("--display_scale", type=int, default=10,
                        help="Scale factor for display")
    parser.add_argument("--original_width", type=int, default=1280,
                        help="Original image width before resize")
    parser.add_argument("--original_height", type=int, default=720,
                        help="Original image height before resize")
    
    args = parser.parse_args()
    
    visualize_demo(
        dataset_path=args.dataset,
        calib_path=args.calib,
        demo_name=args.demo,
        output_path=args.output,
        camera_name=args.camera,
        fps=args.fps,
        show=args.show,
        display_scale=args.display_scale
    )


if __name__ == "__main__":
    main()