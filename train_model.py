# train_model.py
import sys
import torch
import gc
from ultralytics import YOLO

model_type = sys.argv[1]
data_yaml = 'ImgLabelling/TestDataset/data.yaml'
output_dir = 'trained_models'

print(f"Training {model_type}...")
model = YOLO(f'{model_type}.pt')
model.train(
    data=data_yaml,
    epochs=50,
    imgsz=640,
    project=output_dir,
    name=model_type,
    exist_ok=True
)
print(f"{model_type} training complete.")

del model
gc.collect()
torch.cuda.empty_cache()
