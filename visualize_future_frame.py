#!/usr/bin/env python3
"""
Visualize video with future frame (150 frames ahead) concatenated on the right.

Usage:
    python visualize_future_frame.py --input video.mp4 --output output.mp4 --offset 150
"""

import argparse
import cv2
import numpy as np
from tqdm import tqdm


def visualize_future_frame(input_path, output_path, offset=150, fps=None):
    """
    Create video with current frame on left, future frame on right.
    
    Args:
        input_path: Input video path
        output_path: Output video path
        offset: Number of frames ahead for future frame
        fps: Output FPS (default: same as input)
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if fps is None:
        fps = input_fps
    
    print(f"Input video: {total_frames} frames, {width}x{height}, {input_fps:.2f} FPS")
    print(f"Future offset: {offset} frames")
    
    # Read all frames into memory
    print("Loading frames...")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    print(f"Loaded {len(frames)} frames")
    
    # Create output video (2x width for side-by-side)
    output_width = width * 2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, height))
    
    print("Creating output video...")
    for i in tqdm(range(len(frames))):
        current_frame = frames[i]
        
        # Get future frame (clamp to last frame if beyond end)
        future_idx = min(i + offset, len(frames) - 1)
        future_frame = frames[future_idx]
        
        # Concatenate side by side
        combined = np.hstack([current_frame, future_frame])
        
        # Add labels
        cv2.putText(combined, f"Current (t={i})", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(combined, f"Future (t={future_idx}, +{future_idx - i})", (width + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Draw dividing line
        cv2.line(combined, (width, 0), (width, height), (255, 255, 255), 2)
        
        out.write(combined)
    
    out.release()
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize video with future frame")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Input video path")
    parser.add_argument("--output", "-o", type=str, default="future_vis.mp4",
                        help="Output video path")
    parser.add_argument("--offset", type=int, default=150,
                        help="Number of frames ahead for future frame (default: 150)")
    parser.add_argument("--fps", type=float, default=None,
                        help="Output FPS (default: same as input)")
    
    args = parser.parse_args()
    
    visualize_future_frame(
        input_path=args.input,
        output_path=args.output,
        offset=args.offset,
        fps=args.fps
    )


if __name__ == "__main__":
    main()