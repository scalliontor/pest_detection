import sys
import time
import cv2
import numpy as np
from ultralytics import YOLO
try:
    from sort_tracker import Sort
except ImportError:
    print("Error: Required module 'sort_tracker' not found. Ensure it is installed and available on PYTHONPATH.")
    sys.exit(1)

# Config
MODEL_PATH = "best (4).pt"
VIDEO_PATH = "6673965231956.mp4"
OUTPUT_PATH = "output_sort_tracked.mp4"
CONFIDENCE_THRESHOLD = 0.1  # 10%
MAX_AGE = 10
MIN_HITS = 1
IOU_THRESHOLD = 0.3

# Initialize YOLO model
model = YOLO(MODEL_PATH)

# Initialize SORT tracker
tracker = Sort(max_age=MAX_AGE, min_hits=MIN_HITS, iou_threshold=IOU_THRESHOLD)

# Open input video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Error: Could not open input video.")
    exit()

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else 0

# Output writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

# Define colors for different tracks
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
          (255, 0, 255), (0, 255, 255), (255, 165, 0), (128, 0, 128)]

def get_color(track_id):
    """Get a consistent color for each track ID."""
    return COLORS[track_id % len(COLORS)]

print("Processing video with SORT tracker...")
start_time = time.time()
frame_idx = 0
BAR_LEN = 30

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection (without tracking, silenced)
    results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
    
    # Prepare detections for SORT
    detections = []
    if results.boxes is not None and len(results.boxes) > 0:
        xyxy = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), conf in zip(xyxy, confs):
            detections.append([float(x1), float(y1), float(x2), float(y2), float(conf)])
    detections = np.array(detections, dtype=float) if len(detections) > 0 else np.empty((0, 5))
    
    # Update SORT tracker (always call, even with empty detections)
    tracked_objects = tracker.update(detections)
    
    # Draw tracked objects
    for obj in tracked_objects:
        row = np.asarray(obj).flatten()
        x1, y1, x2, y2 = map(int, row[:4])
        track_id = int(row[4])
        color = get_color(track_id)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw track ID
        label = f"ID: {int(track_id)}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # Write to output file
    out.write(frame)

    # Progress tracking
    frame_idx += 1
    elapsed = max(1e-6, time.time() - start_time)
    cur_fps = frame_idx / elapsed
    if total_frames > 0:
        progress = min(1.0, frame_idx / total_frames)
        filled = int(progress * BAR_LEN)
        bar = "█" * filled + "-" * (BAR_LEN - filled)
        remaining_frames = max(0, total_frames - frame_idx)
        eta = remaining_frames / cur_fps if cur_fps > 0 else 0.0
        sys.stdout.write(f"\rProgress: [{bar}] {progress*100:5.1f}% | {frame_idx}/{total_frames} | {cur_fps:5.2f} FPS | ETA: {eta:5.1f}s")
    else:
        sys.stdout.write(f"\rFrames: {frame_idx} | {cur_fps:5.2f} FPS")
    sys.stdout.flush()

cap.release()
out.release()
cv2.destroyAllWindows()

print()  # newline after progress bar
print("✅ Done! Output saved as:", OUTPUT_PATH)
