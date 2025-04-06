from ultralytics import YOLOE

# Initialize a YOLOE model
model = YOLOE("yoloe-11l-seg.pt")  # or select yoloe-11s/m-seg.pt for different sizes

# Set text prompt
names = ["chicken", "bird"]
model.set_classes(names, model.get_text_pe(names))

# Execute prediction for specified categories on an image
results = model.predict("Training/test_easy.png")

# Show results
results[0].show()