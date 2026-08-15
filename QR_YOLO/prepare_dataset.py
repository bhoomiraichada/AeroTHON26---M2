import os
import shutil
import random

IMAGE_DIR = "images"
LABEL_DIR = "labels"

TRAIN_IMAGE_DIR = "train/images"
TRAIN_LABEL_DIR = "train/labels"

VAL_IMAGE_DIR = "val/images"
VAL_LABEL_DIR = "val/labels"

os.makedirs(TRAIN_IMAGE_DIR, exist_ok=True)
os.makedirs(TRAIN_LABEL_DIR, exist_ok=True)
os.makedirs(VAL_IMAGE_DIR, exist_ok=True)
os.makedirs(VAL_LABEL_DIR, exist_ok=True)

images = [
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

random.seed(42)
random.shuffle(images)

split = int(len(images) * 0.8)

train_images = images[:split]
val_images = images[split:]

for image_name in train_images:

    label_name = os.path.splitext(image_name)[0] + ".txt"

    shutil.copy(
        os.path.join(IMAGE_DIR, image_name),
        os.path.join(TRAIN_IMAGE_DIR, image_name)
    )

    shutil.copy(
        os.path.join(LABEL_DIR, label_name),
        os.path.join(TRAIN_LABEL_DIR, label_name)
    )

for image_name in val_images:

    label_name = os.path.splitext(image_name)[0] + ".txt"

    shutil.copy(
        os.path.join(IMAGE_DIR, image_name),
        os.path.join(VAL_IMAGE_DIR, image_name)
    )

    shutil.copy(
        os.path.join(LABEL_DIR, label_name),
        os.path.join(VAL_LABEL_DIR, label_name)
    )

print("Dataset prepared.")
print("Training images:", len(train_images))
print("Validation images:", len(val_images))