from ultralytics import YOLOE

# Initialize model
model = YOLOE("yoloe-11l-seg.pt")

# Set text prompt
names = ["chicken", "bird"]
model.set_classes(names, model.get_text_pe(names))

# make predictions
results = model.predict("../Training/QuickTestIMG/test_easy.png")

# Show results
results[0].show()