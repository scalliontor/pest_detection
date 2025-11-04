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

# ORB-based stabilizer setup
ORB_FEATURES = 1000
MATCH_RATIO = 0.4  # Lowe's ratio test
RANSAC_THRESH = 10.0
SMOOTH_ALPHA = 0.3  # blend previous transform for smoothing
STAB_WINDOW = 16  # frames for global smoothing window
orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
prev_kp, prev_des = None, None
last_M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

# Stabilization trajectory state (cumulative path)
cum_x = 0.0
cum_y = 0.0
cum_a = 0.0
traj_x = []
traj_y = []
traj_a = []
print("Stabilizer: ORB")

def _smooth_tail_poly(vals, window, degree=2):
    """
    Fit a degree-k polynomial to the last `window` samples and return the
    optimized value at the tail (Savitzky–Golay style smoothing).
    Falls back to mean if not enough points or on numerical issues.
    """
    n = len(vals)
    if n == 0:
        return 0.0
    w = min(window, n)
    seg = np.asarray(vals[-w:], dtype=np.float32)
    if w <= degree:
        return float(np.mean(seg))
    x = np.arange(w, dtype=np.float32)
    try:
        coeffs = np.polyfit(x, seg, deg=degree)
        x_last = w - 1
        y_fit = np.polyval(coeffs, x_last)
        return float(y_fit)
    except Exception:
        return float(np.mean(seg))

print("Processing video with SORT tracker...")
start_time = time.time()
frame_idx = 0
BAR_LEN = 30

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ORB stabilization (align current frame to previous)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp, des = orb.detectAndCompute(gray, None)

    # Estimate inter-frame motion and update global smoothed stabilization
    dx = dy = da = 0.0
    if (
        prev_kp is not None and prev_des is not None and
        kp is not None and des is not None and
        len(prev_kp) > 0 and len(kp) > 0
    ):
        try:
            matches = bf.knnMatch(prev_des, des, k=2)
            good = [m for m, n in matches if m.distance < MATCH_RATIO * n.distance]
        except Exception:
            good = []
        if len(good) >= 8:
            src_pts = np.float32([prev_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M_est, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_THRESH)
            if M_est is not None:
                # decompose incremental motion
                dx = float(M_est[0, 2])
                dy = float(M_est[1, 2])
                da = float(np.arctan2(M_est[1, 0], M_est[0, 0]))

    # Update cumulative trajectory
    cum_x += dx
    cum_y += dy
    cum_a += da
    traj_x.append(cum_x)
    traj_y.append(cum_y)
    traj_a.append(cum_a)

    # Global optimization over last STAB_WINDOW frames via poly fit
    x_smooth = _smooth_tail_poly(traj_x, STAB_WINDOW, degree=2)
    y_smooth = _smooth_tail_poly(traj_y, STAB_WINDOW, degree=2)
    a_smooth = _smooth_tail_poly(traj_a, STAB_WINDOW, degree=2)

    # Compute correction to apply this frame
    diff_x = x_smooth - cum_x
    diff_y = y_smooth - cum_y
    diff_a = a_smooth - cum_a
    ca = np.cos(diff_a)
    sa = np.sin(diff_a)
    M_corr = np.array([[ca, -sa, diff_x], [sa, ca, diff_y]], dtype=np.float32)
    frame = cv2.warpAffine(frame, M_corr, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    gray = cv2.warpAffine(gray, M_corr, (width, height), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
    # Update previous features
    prev_kp, prev_des = kp, des

    # Run YOLO detection (without tracking, silenced) on stabilized frame
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
    
    # Draw ORB keypoints on the stabilized frame for visualization
    try:
        frame = cv2.drawKeypoints(
            frame, kp if kp is not None else [], None,
            color=(0, 255, 255),  # yellow keypoints for visibility
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
    except Exception:
        pass
    
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
