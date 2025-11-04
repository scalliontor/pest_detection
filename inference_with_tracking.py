"""
YOLO Inference with Object Tracking on Video
This script performs object detection and tracking on video files using YOLO model.
Objects are tracked across frames and removed after they disappear for a specified period.
"""

from ultralytics import YOLO
import cv2
from collections import defaultdict
import time
import numpy as np

# Configuration
MODEL_PATH = "best (1).pt"
VIDEO_PATH = "6673965231956.mp4"
OUTPUT_PATH = "output_tracked.mp4"
CONFIDENCE_THRESHOLD = 0.1
TRACK_HISTORY_LENGTH = 30  # Number of frames to keep in track history
DISAPPEAR_FRAMES = 20  # Number of frames an object can be missing before removal
OPTICAL_FLOW_THRESHOLD = 0.7  # Motion magnitude threshold for optical flow (alive detection)
TRACKER_CONFIG = "botsort.yaml"
INFERENCE_SIZE = 640  # Match training image size

# Optical Flow Parameters (Farneback)
FLOW_PYR_SCALE = 0.5
FLOW_LEVELS = 3
FLOW_WINSIZE = 15
FLOW_ITERATIONS = 3
FLOW_POLY_N = 5
FLOW_POLY_SIGMA = 1.2
FLOW_FLAGS = 0

def main():
    # Load YOLO model
    print(f"Loading YOLO model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    
    # Open video
    print(f"Opening video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {VIDEO_PATH}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height} @ {fps} FPS, Total frames: {total_frames}")
    print(f"Inference size: {INFERENCE_SIZE}x{INFERENCE_SIZE}")
    
    # Create video writer with reduced FPS (2x slower)
    output_fps = fps # Half the FPS = 2x slower playback
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, output_fps, (width, height))
    
    print(f"Output FPS: {output_fps} (2x slower than original)")
    
    # Track history and disappearance tracking
    track_history = defaultdict(lambda: [])
    track_last_seen = defaultdict(int)
    active_tracks = set()
    prev_positions = {}
    track_status = {}  # Store alive/dead status and optical flow data
    
    # Optical flow state
    prev_gray = None
    
    frame_count = 0
    start_time = time.time()
    
    print("Starting inference with tracking and optical flow...")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"Track disappear threshold: {DISAPPEAR_FRAMES} frames")
    print(f"Optical flow threshold: {OPTICAL_FLOW_THRESHOLD} (alive detection)")
    print(f"Tracker: {TRACKER_CONFIG}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_count += 1
        
        # Convert to grayscale for optical flow
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Compute optical flow if we have a previous frame
        flow = None
        flow_mag = None
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, current_gray, None,
                FLOW_PYR_SCALE, FLOW_LEVELS, FLOW_WINSIZE, 
                FLOW_ITERATIONS, FLOW_POLY_N, FLOW_POLY_SIGMA, FLOW_FLAGS
            )
            # Calculate magnitude
            flow_mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Run YOLO tracking on the frame
        # persist=True keeps track IDs consistent across frames
        # imgsz resizes to 640x640 to match training data
        results = model.track(
            frame, 
            persist=True, 
            conf=CONFIDENCE_THRESHOLD, 
            tracker=TRACKER_CONFIG,
            imgsz=INFERENCE_SIZE,
            verbose=False
        )
        
        # Get current frame's track IDs
        current_frame_tracks = set()
        
        # Process results
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            
            # Update track information
            for box, track_id, conf, cls_id in zip(boxes, track_ids, confidences, class_ids):
                current_frame_tracks.add(track_id)
                
                # Update last seen frame for this track
                track_last_seen[track_id] = frame_count
                
                # Add to active tracks
                active_tracks.add(track_id)
                
                # Calculate center point
                x1, y1, x2, y2 = box
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Calculate optical flow within bounding box
                flow_motion = 0
                is_alive = False
                
                if flow_mag is not None:
                    # Use bounding box as mask
                    x1_int, y1_int = max(0, int(x1)), max(0, int(y1))
                    x2_int, y2_int = min(width, int(x2)), min(height, int(y2))
                    
                    # Extract flow magnitude in the bounding box
                    if x2_int > x1_int and y2_int > y1_int:
                        box_flow = flow_mag[y1_int:y2_int, x1_int:x2_int]
                        if box_flow.size > 0:
                            flow_motion = np.mean(box_flow)
                            
                            # Determine if bug is alive based on optical flow
                            if flow_motion > OPTICAL_FLOW_THRESHOLD:
                                is_alive = True
                
                # Update track status
                track_status[track_id] = {
                    'alive': is_alive,
                    'flow_motion': flow_motion,
                    'position': (center_x, center_y)
                }
                
                # Store center point for track history (as int for drawing)
                center_x_int = int(center_x)
                center_y_int = int(center_y)
                track_history[track_id].append((center_x_int, center_y_int))
                
                # Keep only recent history
                if len(track_history[track_id]) > TRACK_HISTORY_LENGTH:
                    track_history[track_id].pop(0)
        
        # Remove tracks that haven't been seen for DISAPPEAR_FRAMES
        tracks_to_remove = set()
        for track_id in active_tracks:
            if track_id not in current_frame_tracks:
                frames_missing = frame_count - track_last_seen[track_id]
                if frames_missing > DISAPPEAR_FRAMES:
                    tracks_to_remove.add(track_id)
        
        for track_id in tracks_to_remove:
            active_tracks.remove(track_id)
            if track_id in track_history:
                del track_history[track_id]
            if track_id in track_last_seen:
                del track_last_seen[track_id]
            if track_id in prev_positions:
                del prev_positions[track_id]
            if track_id in track_status:
                del track_status[track_id]
        
        # Don't use YOLO's plot, draw custom bounding boxes
        annotated_frame = frame.copy()
        
        # Draw custom bounding boxes with color based on alive/dead status
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            
            for box, track_id, conf, cls_id in zip(boxes, track_ids, confidences, class_ids):
                x1, y1, x2, y2 = map(int, box)
                
                # Determine color based on alive/dead status
                is_alive = track_status.get(track_id, {}).get('alive', False)
                box_color = (0, 255, 0) if is_alive else (0, 0, 255)  # Green if alive, Red if dead
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
                
                # Create label with status
                status_text = "ALIVE" if is_alive else "DEAD"
                label = f"ID:{track_id} {status_text} {conf:.2f}"
                
                # Draw label background
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), box_color, -1)
                
                # Draw label text
                cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw track history (trajectory) with color based on alive/dead status
        for track_id in active_tracks:
            if track_id in track_history and len(track_history[track_id]) > 1:
                points = track_history[track_id]
                
                # Color based on alive/dead status
                is_alive = track_status.get(track_id, {}).get('alive', False)
                line_color = (0, 255, 0) if is_alive else (0, 0, 255)  # Green if alive, Red if dead
                
                for i in range(1, len(points)):
                    # Draw line between consecutive points
                    cv2.line(annotated_frame, points[i-1], points[i], line_color, 2)
        
        # Count alive vs dead bugs
        alive_count = sum(1 for tid in current_frame_tracks 
                         if track_status.get(tid, {}).get('alive', False))
        dead_count = len(current_frame_tracks) - alive_count
        total_bugs = len(active_tracks)
        
        # Create compact info panel with white background
        panel_height = 80
        panel_width = 300
        panel_x = 10
        panel_y = 10
        
        # Draw white semi-transparent background
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (255, 255, 255), -1)
        # Blend the overlay with the frame (85% opacity)
        cv2.addWeighted(overlay, 0.85, annotated_frame, 0.15, 0, annotated_frame)
        
        # Draw border around panel
        cv2.rectangle(annotated_frame, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (100, 100, 100), 2)
        
        # Add statistics
        y_pos = panel_y + 25
        cv2.putText(annotated_frame, f"Total Bugs: {total_bugs}", (panel_x + 10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        y_pos += 25
        cv2.putText(annotated_frame, f"Alive: {alive_count}", (panel_x + 10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 0), 2)
        
        y_pos += 25
        cv2.putText(annotated_frame, f"Dead: {dead_count}", (panel_x + 10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 150), 2)
        
        # Update previous frame for optical flow
        prev_gray = current_gray
        
        # Write frame
        out.write(annotated_frame)
        
        # Display progress
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps_processing = frame_count / elapsed if elapsed > 0 else 0
            
            # Count current alive/dead
            curr_alive = sum(1 for tid in current_frame_tracks 
                           if track_status.get(tid, {}).get('alive', False))
            curr_dead = len(current_frame_tracks) - curr_alive
            
            print(f"Frame {frame_count}/{total_frames} | "
                  f"Total: {len(active_tracks)} | "
                  f"Alive: {curr_alive} | Dead: {curr_dead} | "
                  f"FPS: {fps_processing:.2f}")
    
    # Cleanup
    cap.release()
    out.release()
    
    total_time = time.time() - start_time
    print(f"\nProcessing complete!")
    print(f"Total frames processed: {frame_count}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average FPS: {frame_count/total_time:.2f}")
    print(f"Output saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
