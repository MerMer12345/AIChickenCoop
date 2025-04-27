import os.path

import supervision as sv
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import cv2
from removeBadAnnotations import remove_bad_annotations


# Use interactive backend
matplotlib.use('TkAgg')

mask_annotator = sv.MaskAnnotator()
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

bad_annotations = []
dataset_dir = "../ImgLabelling/ValidationDataset"

dataset = sv.DetectionDataset.from_yolo(
    images_directory_path=os.path.join(dataset_dir, "train/images"),
    annotations_directory_path=os.path.join(dataset_dir, "train/labels"),
    data_yaml_path=os.path.join(dataset_dir, "data.yaml"))


for i, (image_path, image, annotation) in enumerate(dataset):
    print(f"Loading: {image_path}, Image shape: {image.shape}, {i}")


def showimg(images, image_names):
    idx = 0
    while 0 <= idx < len(images):
        fig, ax = plt.subplots()
        try:
            rgb_image = cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Color conversion failed: {e}")
            rgb_image = images[idx]

        ax.imshow(rgb_image)
        ax.set_title(f"{image_names[idx]} ({idx + 1}/{len(images)})")
        ax.axis('off')
        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)

        print("\nOptions:")
        print("n  - Next image")
        print("b  - Previous image")
        print("f  - Flag as bad annotation")
        print("s  - Skip next 10 images")
        print("sf - Flag and skip next 10 images")
        print("q  - Quit review\n")

        key = input("Your choice: ").strip().lower()
        plt.close(fig)

        if key == 'n':
            idx += 1
        elif key == 'b':
            idx = max(idx - 1, 0)
        elif key == 'f':
            bad_annotations.append(image_names[idx])
            print(f"Flagged: {image_names[idx]}")
            idx += 1
        elif key == 's':
            idx += 10
        elif key == 'sf':
            for j in range(10):
                if idx + j < len(images):
                    bad_annotations.append(image_names[idx + j])
                    print(f"Flagged: {image_names[idx + j]}")
            idx += 10
        elif key == 'q':
            break
        else:
            print("Invalid input. Use n, b, f, s, sf, or q.")

    return bad_annotations


# Prepare images
images = []
image_names = []
for i, (image_path, image, annotation) in enumerate(dataset):
    annotated_image = image.copy()
    annotated_image = mask_annotator.annotate(scene=annotated_image, detections=annotation)
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=annotation)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=annotation)

    image_names.append(Path(image_path).name)
    images.append(annotated_image)

flagged = showimg(images, image_names)


# Save or delete flagged results
print("s - Save Results")
print("d - Delete bad annotations")
key = input("Your choice: ").strip().lower()
if key == 's':
    with open("Bad_annotations/bad_annotations_TestData.txt", "w") as f:
        for name in flagged:
            f.write(name + "\n")
elif key == 'd':
    dataset_dir = "../ImgLabelling/ValidationDataset/train"
    bad_annotations_file = "Bad_annotations/bad_annotations_TestData.txt"
    remove_bad_annotations(dataset_dir, bad_annotations_file)

print("\nReview complete.")
print(f"Flagged {len(flagged)} images with bad annotations.")
