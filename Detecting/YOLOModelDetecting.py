from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import time
import psutil
import os
import csv

model = YOLO('../Training/trained_models/yolo12x/weights/best.pt')

video_path = '../ImgLabelling/TestVids/SmallTest1.mp4'
cap = cv2.VideoCapture(video_path)

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter('Output_Vids/output_detected_with_count.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

object_counts = []
frame_index = 0

process = psutil.Process(os.getpid())

while cap.isOpened():
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        break
    #frame = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))  # for grayscale contrast enhancement
    #frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # convert back to 3-channel for YOLO

    #results = model(frame, conf=0.4, iou=0.5)
    results = model(frame)
    detections = results[0].boxes
    num_objects = len(detections)

    object_counts.append(num_objects)

    annotated_frame = results[0].plot()
    cv2.putText(
        annotated_frame,
        f'Detected Objects: {num_objects}',
        org=(20, 40),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(0, 255, 0),
        thickness=2
    )

    # Calculate FPS
    end_time = time.time()
    processing_time = end_time - start_time
    current_fps = 1 / processing_time if processing_time > 0 else 0

    # Get RAM usage in MB
    ram_usage = process.memory_info().rss / 1024 / 1024

    print(f'Frame: {frame_index} | FPS: {current_fps:.2f} | RAM Usage: {ram_usage:.2f} MB')

    cv2.imshow('YOLO Detection with Count', annotated_frame)
    out.write(annotated_frame)

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