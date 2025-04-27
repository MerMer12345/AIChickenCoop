from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import time
import psutil
import os
import csv
from deep_sort_realtime.deepsort_tracker import DeepSort

model = YOLO('../Training/trained_models/yolov8x/weights/best.pt')
tracker = DeepSort(max_age=40, n_init=2, max_iou_distance=0.7, nn_budget=100)

video_path = '../ImgLabelling/TestVids/SmallTest1.mp4'
cap = cv2.VideoCapture(video_path)

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter('Output_Vids/output_detected_with_deepsort_filtered.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

object_counts = []
frame_index = 0
process = psutil.Process(os.getpid())

# Memory and recovery setup
active_tracks = {}  # track_id: {'center': (x, y), 'last_frame': int}
lost_tracks = {}    # same structure
recovered_ids = set()
track_log_filename = 'Logs/track_recovery_log.csv'

# Setup CSV log
with open(track_log_filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Event', 'Track ID', 'Frame', 'Time (s)'])

MAX_LOST_FRAMES = int(fps * 2)  # 2 seconds of memory
MAX_MATCH_DIST = 50  # pixels

while cap.isOpened():
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    orig_frame = frame.copy()

    results = model(orig_frame)[0]
    boxes = sorted(results.boxes, key=lambda b: b.conf[0], reverse=True)[:4]

    detections = []
    for box in boxes:
        conf = float(box.conf[0])
        if conf < 0.4:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls = int(box.cls[0])
        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

    tracks = tracker.update_tracks(detections, frame=orig_frame)

    current_ids = set()
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, r, b = map(int, track.to_ltrb())
        cx, cy = (l + r) // 2, (t + b) // 2
        current_ids.add(track_id)

        # Recovery logic
        recovered = False
        for lost_id, lost_info in lost_tracks.items():
            lx, ly = lost_info['center']
            frames_since_lost = frame_index - lost_info['last_frame']
            dist = ((cx - lx)**2 + (cy - ly)**2)**0.5
            if dist < MAX_MATCH_DIST and frames_since_lost <= MAX_LOST_FRAMES:
                recovered = True
                recovered_ids.add(track_id)
                # Log recovery
                with open(track_log_filename, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Recovered', track_id, frame_index, f"{frame_index / fps:.2f}"])
                break

        # Draw tracked box
        box_color = (255, 0, 0) if track_id in recovered_ids else (0, 255, 0)
        cv2.rectangle(orig_frame, (l, t), (r, b), box_color, 2)
        cv2.putText(orig_frame, f'ID {track_id}', (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Update active memory
        active_tracks[track_id] = {'center': (cx, cy), 'last_frame': frame_index}
        lost_tracks.pop(track_id, None)

    # Find lost tracks
    previous_ids = set(active_tracks.keys())
    missing_ids = previous_ids - current_ids
    for track_id in missing_ids:
        last_seen = active_tracks[track_id]['last_frame']
        if frame_index - last_seen <= MAX_LOST_FRAMES:
            lost_tracks[track_id] = active_tracks[track_id]
            # Log lost track
            with open(track_log_filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Lost', track_id, frame_index, f"{frame_index / fps:.2f}"])
        active_tracks.pop(track_id)

    num_objects = len(current_ids)
    object_counts.append(num_objects)

    end_time = time.time()
    processing_time = end_time - start_time
    current_fps = 1 / processing_time if processing_time > 0 else 0
    ram_usage = process.memory_info().rss / 1024 / 1024

    print(f'Frame: {frame_index} | FPS: {current_fps:.2f} | RAM Usage: {ram_usage:.2f} MB')

    cv2.putText(orig_frame, f'Tracked Objects: {num_objects}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow('YOLO + Deep SORT (Recovery)', orig_frame)
    out.write(orig_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_index += 1

cap.release()
out.release()
cv2.destroyAllWindows()


def save_detection_data(counts, fps, file='Output_Vids/object_countsDeepSORT.csv'):
    frames = list(range(len(counts)))
    time_seconds = [f / fps for f in frames]

    with open(file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Frame', 'Time (s)', 'Object Count'])
        for frame, time_sec, count in zip(frames, time_seconds, counts):
            writer.writerow([frame, f"{time_sec:.2f}", count])

save_detection_data(object_counts, fps)

def plot_detection_graph(counts, fps):
    frames = list(range(len(counts)))
    time_seconds = [f / fps for f in frames]

    plt.figure(figsize=(10, 5))
    plt.plot(time_seconds, counts, marker='o')
    plt.title('Object Count Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Number of Detected Objects')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_detection_graph(object_counts, fps)