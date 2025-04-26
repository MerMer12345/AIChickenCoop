from ultralytics import YOLO
import cv2
import time
import psutil
import os
import csv
from bytetracker import BYTETracker
import numpy as np
import matplotlib.pyplot as plt

# Initialize YOLOv8 model
model = YOLO('trained_models/yolov8x/weights/best.pt')

# Initialize ByteTrack
tracker = BYTETracker(track_thresh=0.5, track_buffer=30, match_thresh=0.8, frame_rate=30)

video_path = '../ImgLabelling/TestVids/SmallTest1.mp4'
cap = cv2.VideoCapture(video_path)

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter('output_detected_with_bytetrack.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

object_counts = []
frame_index = 0
process = psutil.Process(os.getpid())

while cap.isOpened():
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 detection
    results = model(frame)[0]

    # Prepare detections for ByteTrack
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        score = box.conf[0].item()
        cls = int(box.cls[0].item())
        detections.append([x1, y1, x2, y2, score, cls])

    detections = np.array(detections)

    # Update tracker
    online_targets = tracker.update(detections, frame)

    # Draw tracking results
    for t in online_targets:
        tlwh = t.tlwh
        track_id = t.track_id
        x1, y1, w, h = tlwh
        x2, y2 = x1 + w, y1 + h
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f'ID {track_id}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    object_counts.append(len(online_targets))

    # Calculate FPS and RAM usage
    end_time = time.time()
    processing_time = end_time - start_time
    current_fps = 1 / processing_time if processing_time > 0 else 0
    ram_usage = process.memory_info().rss / 1024 / 1024

    print(f'Frame: {frame_index} | FPS: {current_fps:.2f} | RAM Usage: {ram_usage:.2f} MB')

    cv2.putText(frame, f'Tracked Objects: {len(online_targets)}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow('YOLOv8 + ByteTrack', frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_index += 1

cap.release()
out.release()
cv2.destroyAllWindows()

def save_detection_data(counts, fps, filename='object_counts.csv'):
    frames = list(range(len(counts)))
    time_seconds = [f / fps for f in frames]

    with open(filename, mode='w', newline='') as file:
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