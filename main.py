import subprocess

models = ['yolo12x', 'yolov11x']

for model in models:
    subprocess.run([
        'python', 'train_model.py', model
    ])
