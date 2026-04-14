#!/usr/bin/env python3
"""
Evaluate high-level policy on validation set of dataset with video output.

Usage:
    python eval_highlevel_on_dataset.py \
        --agent /path/to/checkpoint.pth \
        --dataset /path/to/dataset.hdf5 \
        --calib /path/to/cam_calibration.json \
        --video_prompt /path/to/video_prompt.mp4 \
        --output eval_traj.mp4 \
        --goal
"""

import argparse
import json
import h5py
import numpy as np
from tqdm import tqdm
import cv2

import torch

import mimicplay.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils


def load_calibration(calib_path, camera_name="cam1"):
    """Load camera calibration from JSON file."""
    with open(calib_path, 'r') as f:
        calib = json.load(f)
    
    cam_calib = calib[camera_name]
    intrinsic = np.array(cam_calib['intrinsic'])
    extrinsic = np.array(cam_calib['extrinsic'])
    
    return intrinsic, extrinsic


def world_to_camera(points_world, extrinsic_cam_to_world):
    """Transform points from world frame to camera frame."""
    extrinsic_world_to_cam = np.linalg.inv(extrinsic_cam_to_world)
    N = points_world.shape[0]
    points_homo = np.hstack([points_world, np.ones((N, 1))])
    points_cam_homo = (extrinsic_world_to_cam @ points_homo.T).T
    return points_cam_homo[:, :3]


def project_to_image(points_cam, intrinsic, image_size=None):
    """Project 3D points in camera frame to 2D image coordinates."""
    valid_mask = points_cam[:, 2] > 0.01
    points_2d_homo = (intrinsic @ points_cam.T).T
    
    points_2d = np.zeros((points_cam.shape[0], 2))
    points_2d[valid_mask, 0] = points_2d_homo[valid_mask, 0] / points_2d_homo[valid_mask, 2]
    points_2d[valid_mask, 1] = points_2d_homo[valid_mask, 1] / points_2d_homo[valid_mask, 2]
    
    if image_size is not None:
        w, h = image_size
        valid_mask &= (points_2d[:, 0] >= 0) & (points_2d[:, 0] < w)
        valid_mask &= (points_2d[:, 1] >= 0) & (points_2d[:, 1] < h)
    
    return points_2d, valid_mask


def draw_trajectory_on_image(image, points_2d, valid_mask, color_start, color_end, thickness=2, radius=4):
    """Draw trajectory points and lines on image."""
    img = image.copy()
    n_points = len(points_2d)
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) == 0:
        return img
    
    for i in range(len(valid_indices) - 1):
        idx1 = valid_indices[i]
        idx2 = valid_indices[i + 1]
        pt1 = tuple(points_2d[idx1].astype(int))
        pt2 = tuple(points_2d[idx2].astype(int))
        t = idx1 / max(n_points - 1, 1)
        color = tuple(int(color_start[j] * (1 - t) + color_end[j] * t) for j in range(3))
        cv2.line(img, pt1, pt2, color, thickness)
    
    for idx in valid_indices:
        pt = tuple(points_2d[idx].astype(int))
        t = idx / max(n_points - 1, 1)
        color = tuple(int(color_start[j] * (1 - t) + color_end[j] * t) for j in range(3))
        cv2.circle(img, pt, radius, color, -1)
        cv2.circle(img, pt, radius, (255, 255, 255), 1)
    
    return img


def get_valid_demos(f):
    """Get list of validation demo names from dataset."""
    if 'mask/valid' not in f:
        print("Warning: No 'mask/valid' found in dataset. Using all demos.")
        return sorted(f['data'].keys(), key=lambda x: int(x.split('_')[1]))
    
    valid_demos = [d.decode('utf-8') if isinstance(d, bytes) else d for d in f['mask/valid'][:]]
    
    if len(valid_demos) == 0:
        print("Warning: 'mask/valid' is empty. Using all demos.")
        return sorted(f['data'].keys(), key=lambda x: int(x.split('_')[1]))
    
    return valid_demos


def eval_highlevel_on_dataset(args):
    # Device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    
    # Load policy
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=args.agent, device=device, verbose=True
    )
    
    if args.video_prompt is not None:
        policy.policy.load_eval_video_prompt(args.video_prompt)
    
    policy.start_episode()
    
    # Load calibration (cam1 = left camera)
    intrinsic, extrinsic = load_calibration(args.calib, camera_name="cam1")
    
    results = {'l2_errors': [], 'l2_errors_per_point': []}
    
    with h5py.File(args.dataset, 'r') as f:
        # Get validation demos only
        demos = get_valid_demos(f)
        
        if args.num_demos > 0:
            demos = demos[:args.num_demos]
        
        print(f"Evaluating on {len(demos)} validation demos: {demos}")
        
        video_writer = None
        
        for demo_idx, demo_name in enumerate(demos):
            if demo_name not in f['data']:
                print(f"Warning: {demo_name} not found in dataset, skipping.")
                continue
            
            demo = f[f'data/{demo_name}']
            
            images = demo['obs/agentview_image'][:]
            eef_pos = demo['obs/robot0_eef_pos'][:]
            gt_future_traj = demo['obs/robot0_eef_pos_future_traj'][:]
            
            T, H, W, C = images.shape
            image_size = (W, H)
            display_scale = args.display_scale
            display_W = W * display_scale
            display_H = H * display_scale
            
            
            if video_writer is None and args.output is not None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(args.output, fourcc, args.fps, (display_W, display_H))
            
            print(f"Processing {demo_name}: {T} frames")
            
            for t in tqdm(range(0, T, args.skip_frames), desc=demo_name):
                img = images[t]
                # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img = cv2.resize(img, (args.img_size, args.img_size), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)
                img = img.astype(np.float32) / 255.0

                obs = {
                    'agentview_image': torch.from_numpy(img).unsqueeze(0).to(device),
                    'robot0_eef_pos': torch.from_numpy(eef_pos[t].astype(np.float32)).unsqueeze(0).to(device),
                }


                goal = None
                if args.goal:
                    goal_idx = min(t + 100, T - 1)  # 150 frames ahead, or last frame
                    goal_img = images[goal_idx]
                    # goal_img = cv2.cvtColor(goal_img, cv2.COLOR_RGB2BGR)
                    goal_img_resized = cv2.resize(goal_img, (args.img_size, args.img_size), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)
                    goal_img_resized = goal_img_resized.astype(np.float32) / 255.0

                    goal = {
                        'agentview_image': torch.from_numpy(goal_img_resized).unsqueeze(0).to(device),  # add batch dim
                    }
                else:
                    goal = None

                with torch.no_grad():
                    pred_traj = policy(ob=obs, goal=goal)
                
                pred_traj = np.array(pred_traj).flatten()
                gt_traj = gt_future_traj[t].flatten()
                
                l2_error = np.linalg.norm(pred_traj - gt_traj)
                results['l2_errors'].append(l2_error)
                
                if len(pred_traj) == 30 and len(gt_traj) == 30:
                    pred_points = pred_traj.reshape(10, 3)
                    gt_points = gt_traj.reshape(10, 3)
                    per_point_errors = np.linalg.norm(pred_points - gt_points, axis=1)
                    results['l2_errors_per_point'].append(per_point_errors)
                
                if video_writer is not None:
                    frame = cv2.cvtColor(images[t], cv2.COLOR_RGB2BGR)
                    frame = cv2.resize(frame, (display_W, display_H), interpolation=cv2.INTER_NEAREST)
                    
                    # GT trajectory (green -> blue)
                    gt_points_3d = np.vstack([eef_pos[t].reshape(1, 3), gt_traj.reshape(-1, 3)])
                    gt_cam = world_to_camera(gt_points_3d, extrinsic)
                    gt_2d, gt_valid = project_to_image(gt_cam, intrinsic, image_size)
                    gt_2d_scaled = gt_2d * display_scale
                    frame = draw_trajectory_on_image(frame, gt_2d_scaled, gt_valid,
                                                     color_start=(0, 255, 0),
                                                     color_end=(255, 0, 0),
                                                     thickness=2, radius=4)
                    
                    # Predicted trajectory (yellow -> red)
                    pred_points_3d = np.vstack([eef_pos[t].reshape(1, 3), pred_traj.reshape(-1, 3)])
                    pred_cam = world_to_camera(pred_points_3d, extrinsic)
                    pred_2d, pred_valid = project_to_image(pred_cam, intrinsic, image_size)
                    pred_2d_scaled = pred_2d * display_scale
                    frame = draw_trajectory_on_image(frame, pred_2d_scaled, pred_valid,
                                                     color_start=(0, 255, 255),
                                                     color_end=(0, 0, 255),
                                                     thickness=2, radius=4)
                    
                    # Current EEF position (cyan)
                    eef_cam = world_to_camera(eef_pos[t].reshape(1, 3), extrinsic)
                    eef_2d, eef_valid = project_to_image(eef_cam, intrinsic, image_size)
                    if eef_valid[0]:
                        pt = tuple((eef_2d[0] * display_scale).astype(int))
                        cv2.circle(frame, pt, 8, (255, 255, 0), -1)
                        cv2.circle(frame, pt, 8, (0, 0, 0), 2)
                    
                    # Text overlay
                    cv2.putText(frame, f"{demo_name} Frame: {t}/{T}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"L2 Error: {l2_error:.4f}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Legend
                    cv2.putText(frame, "GT (green)", (10, display_H - 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame, "Pred (yellow)", (10, display_H - 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    video_writer.write(frame)
        
        if video_writer is not None:
            video_writer.release()
            print(f"Saved video to {args.output}")
    
    # Compute statistics
    l2_errors = np.array(results['l2_errors'])
    
    stats = {
        'mean_l2_error': float(np.mean(l2_errors)),
        'std_l2_error': float(np.std(l2_errors)),
        'median_l2_error': float(np.median(l2_errors)),
        'min_l2_error': float(np.min(l2_errors)),
        'max_l2_error': float(np.max(l2_errors)),
        'num_samples': len(l2_errors),
        'num_demos': len(demos),
    }
    
    if results['l2_errors_per_point']:
        per_point = np.array(results['l2_errors_per_point'])
        stats['per_point_mean'] = per_point.mean(axis=0).tolist()
        stats['per_point_std'] = per_point.std(axis=0).tolist()
    
    # Print results
    print("\n" + "=" * 60)
    print("High-Level Policy Evaluation Results (Validation Set)")
    print("=" * 60)
    print(f"  Validation demos:  {len(demos)}")
    print(f"  Samples evaluated: {stats['num_samples']}")
    print(f"  Mean L2 Error:     {stats['mean_l2_error']:.4f}")
    print(f"  Std L2 Error:      {stats['std_l2_error']:.4f}")
    print(f"  Median L2 Error:   {stats['median_l2_error']:.4f}")
    
    if 'per_point_mean' in stats:
        print("\n  Per-point mean errors (future points 1-10):")
        for i, (mean, std) in enumerate(zip(stats['per_point_mean'], stats['per_point_std'])):
            print(f"    Point {i+1}: {mean:.4f} ± {std:.4f}")
    
    if args.results_output is not None:
        with open(args.results_output, 'w') as f:
            json.dump(stats, f, indent=4)
        print(f"\nSaved results to {args.results_output}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Evaluate high-level policy on validation set")
    parser.add_argument("--agent", type=str, required=True,
                        help="Path to checkpoint pth file")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to HDF5 dataset")
    parser.add_argument("--calib", type=str, required=True,
                        help="Path to camera calibration JSON")
    parser.add_argument("--video_prompt", type=str, default=None,
                        help="Path to video prompt")
    parser.add_argument("--output", type=str, default="eval_traj.mp4",
                        help="Output video path")
    parser.add_argument("--results_output", type=str, default=None,
                        help="Output JSON file for results")
    parser.add_argument("--num_demos", type=int, default=-1,
                        help="Number of validation demos to evaluate (-1 for all)")
    parser.add_argument("--skip_frames", type=int, default=1,
                        help="Evaluate every N frames")
    parser.add_argument("--display_scale", type=int, default=4,
                        help="Scale factor for display")
    parser.add_argument("--fps", type=int, default=20,
                        help="Output video FPS")
    parser.add_argument("--goal", action='store_true',
                        help="Use goal image from future frame")
    parser.add_argument("--img_size", type=int, default=84,
                        help="Resize images to this size before inference")
    
    args = parser.parse_args()
    eval_highlevel_on_dataset(args)


if __name__ == "__main__":
    main()