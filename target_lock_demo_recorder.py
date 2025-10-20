"""
target_lock_demo_recorder.py
Run the click-to-lock CSRT tracker and automatically record a short demo video.
Usage:
    python target_lock_demo_recorder.py        # runs webcam, click to select target, records until time limit or ESC
    python target_lock_demo_recorder.py --video input.mp4  # run on a video file instead of webcam
Options:
    --out PATH        : output demo file path (default: target_lock_demo.mp4)
    --duration SEC    : maximum recording duration in seconds (default: 25)
    --mock            : run in mock serial mode (no Arduino)
Notes:
- Requires opencv-contrib-python and numpy.
- Select a target by clicking on the video window. Recording will start on the first click and stop after duration secs or when you press ESC.
- The script saves a CSV log and an MP4 demo file.
"""

import cv2
import time
import csv
from datetime import datetime
import argparse
import os

# ---------------- DEFAULT CONFIG ----------------
DEFAULT_OUT = 'target_lock_demo.mp4'
DEFAULT_CSV = 'target_lock_log.csv'
DEFAULT_DURATION = 25  # seconds
ROI_SIZE = 100
FOV_X = 60.0
FOV_Y = 40.0
SMOOTH_ALPHA = 0.2
LOCK_THRESHOLD_PX = 25
LOCK_FRAMES_REQUIRED = 6
# ------------------------------------------------

parser = argparse.ArgumentParser(description='Record a demo of click-to-lock CSRT tracker.')
parser.add_argument('--video', help='Input video file path (optional)', default='')
parser.add_argument('--out', help='Output demo file path', default=DEFAULT_OUT)
parser.add_argument('--duration', help='Max recording duration (sec)', type=int, default=DEFAULT_DURATION)
parser.add_argument('--mock', help='Mock serial mode (no Arduino)', action='store_true')
args = parser.parse_args()

VIDEO_PATH = args.video
OUT_PATH = args.out
DURATION = args.duration
MOCK_MODE = args.mock

print('[INFO] Demo recorder starting')
print(f'       VIDEO_PATH="{VIDEO_PATH}", OUT="{OUT_PATH}", DURATION={DURATION}s, MOCK_MODE={MOCK_MODE}')

# Initialize capture
if VIDEO_PATH:
    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(f"Video file not found: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
else:
    cap = cv2.VideoCapture(0)  # default webcam

if not cap.isOpened():
    raise RuntimeError("Cannot open camera or video. Check path/index.")

ret, frame = cap.read()
if not ret or frame is None:
    raise RuntimeError("Cannot read initial frame from camera/video.")
h, w = frame.shape[:2]
cx, cy = w//2, h//2

# Setup CSRT tracker factory
if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_create'):
    TrackerCSRT = cv2.legacy.TrackerCSRT_create
elif hasattr(cv2, 'TrackerCSRT_create'):
    TrackerCSRT = cv2.TrackerCSRT_create
else:
    raise RuntimeError("CSRT tracker not available in this OpenCV build. Install opencv-contrib-python.")

tracker = None
tracking = False

# CSV logger
csv_file = open(DEFAULT_CSV, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['timestamp','tx','ty','pan','tilt','locked'])

# Video writer (will be initialized when recording begins)
writer = None
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

lock_count = 0
prev_pan, prev_tilt = 90, 90
recording_started = False
record_start_time = None

def map_offset_to_angle(dx, dy, frame_w, frame_h, fov_x=FOV_X, fov_y=FOV_Y):
    angle_x = (dx / (frame_w / 2.0)) * (fov_x / 2.0)
    angle_y = (dy / (frame_h / 2.0)) * (fov_y / 2.0)
    pan = 90 - angle_x
    tilt = 90 + angle_y
    pan = max(0, min(180, int(round(pan))))
    tilt = max(0, min(180, int(round(tilt))))
    return pan, tilt

# Mouse callback to select target (starts tracking and recording)
def on_mouse(event, x, y, flags, param):
    global tracker, tracking, lock_count, prev_pan, prev_tilt, recording_started, record_start_time, writer
    if event == cv2.EVENT_LBUTTONDOWN:
        # create ROI centered on click
        x1 = max(0, x - ROI_SIZE//2)
        y1 = max(0, y - ROI_SIZE//2)
        wbox = min(ROI_SIZE, w - x1)
        hbox = min(ROI_SIZE, h - y1)
        bbox = (x1, y1, wbox, hbox)
        tracker = TrackerCSRT()
        ok = tracker.init(frame, bbox)
        if ok:
            tracking = True
            lock_count = 0
            prev_pan, prev_tilt = 90, 90
            print("[INFO] Tracker initialized at", bbox)
            # start recording on first click
            global OUT_PATH
            if not recording_started:
                start_rec()
        else:
            print("[WARN] Tracker initialization failed. Try clicking on a larger visible area.")

def start_rec():
    global writer, recording_started, record_start_time, OUT_PATH
    writer = cv2.VideoWriter(OUT_PATH, fourcc, 20.0, (w, h))
    recording_started = True
    record_start_time = time.time()
    print(f"[INFO] Recording started -> {OUT_PATH}")

cv2.namedWindow('Click CSRT Demo Recorder')
cv2.setMouseCallback('Click CSRT Demo Recorder', on_mouse)

print("[INFO] Click on a target in the window to start tracking and recording. Press ESC to stop early.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[INFO] End of stream.")
        break
    disp = frame.copy()
    current_time = time.time()

    if tracking and tracker is not None:
        ok, box = tracker.update(frame)
        if ok:
            x, y, wb, hb = [int(v) for v in box]
            tx = x + wb//2
            ty = y + hb//2
            cv2.rectangle(disp, (x,y), (x+wb, y+hb), (0,255,0), 2)
            cv2.circle(disp, (tx,ty), 4, (0,255,0), -1)

            dx = tx - cx
            dy = ty - cy
            pan, tilt = map_offset_to_angle(dx, dy, w, h)
            pan = int(prev_pan * (1 - SMOOTH_ALPHA) + pan * SMOOTH_ALPHA)
            tilt = int(prev_tilt * (1 - SMOOTH_ALPHA) + tilt * SMOOTH_ALPHA)

            # deadband update
            if abs(pan - prev_pan) > 1 or abs(tilt - prev_tilt) > 1:
                prev_pan, prev_tilt = pan, tilt
                msg = f'P{pan}T{tilt}'
                if MOCK_MODE:
                    print("[MOCK SERIAL]", msg)

            # lock logic
            if abs(dx) < LOCK_THRESHOLD_PX and abs(dy) < LOCK_THRESHOLD_PX:
                lock_count += 1
            else:
                lock_count = 0
            locked = lock_count >= LOCK_FRAMES_REQUIRED
            if locked:
                if MOCK_MODE:
                    print("[MOCK] LOCKED -> L1")
                cv2.putText(disp, 'LOCKED', (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
            csv_writer.writerow([datetime.utcnow().isoformat(), tx, ty, pan, tilt, int(locked)])
        else:
            cv2.putText(disp, 'TRACKER LOST - click to reselect', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    else:
        cv2.putText(disp, 'Click target to start tracking and recording', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    # Draw center crosshair
    cv2.line(disp, (cx - 10, cy), (cx + 10, cy), (255, 0, 0), 1)
    cv2.line(disp, (cx, cy - 10), (cx, cy + 10), (255, 0, 0), 1)

    # write to file if recording started
    if recording_started and writer is not None:
        writer.write(disp)
        # stop if duration exceeded
        if current_time - record_start_time >= DURATION:
            print("[INFO] Max duration reached, stopping recording.")
            break

    cv2.imshow('Click CSRT Demo Recorder', disp)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        print("[INFO] ESC pressed, exiting.")
        break
    elif key == ord('r'):
        tracking = False
        tracker = None
        lock_count = 0
        prev_pan, prev_tilt = 90, 90

# cleanup
cap.release()
if writer: writer.release()
csv_file.close()
cv2.destroyAllWindows()
print("[INFO] Finished. Demo saved to", OUT_PATH, "Log saved to", DEFAULT_CSV)
