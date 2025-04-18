import os

# Set your paths here
dataset_dir = "ImgLabelling/ValidationDataset/train"
images_dir = os.path.join(dataset_dir, "images")
labels_dir = os.path.join(dataset_dir, "labels")
bad_annotations_file = "ImgLabelling/bad_annotations_TestData.txt"

# Read the names of badly annotated images
with open(bad_annotations_file, "r") as f:
    bad_files = [line.strip() for line in f if line.strip()]

# Remove the image and corresponding label file
for filename in bad_files:
    image_path = os.path.join(images_dir, filename)
    label_path = os.path.join(labels_dir, os.path.splitext(filename)[0] + ".txt")

    if os.path.exists(image_path):
        os.remove(image_path)
        print(f"Removed image: {image_path}")
    else:
        print(f"Image not found: {image_path}")

    if os.path.exists(label_path):
        os.remove(label_path)
        print(f"Removed annotation: {label_path}")
    else:
        print(f"Annotation not found: {label_path}")
