from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

model = YOLO('trained_models/yolov8x/weights/best.pt')

video_path = '../ImgLabelling/TestVids/SmallTest1.mp4'
cap = cv2.VideoCapture(video_path)

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter('output_detected_with_count.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

object_counts = []

frame_index = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

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

    cv2.imshow('YOLO Detection with Count', annotated_frame)
    out.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_index += 1

cap.release()
out.release()
cv2.destroyAllWindows()

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
