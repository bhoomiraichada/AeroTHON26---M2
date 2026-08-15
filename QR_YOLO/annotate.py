import cv2
import os

IMAGE_DIR = "images"
LABEL_DIR = "labels"

os.makedirs(LABEL_DIR, exist_ok=True)

images = [f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))]

images.sort()

for image_name in images:

    image_path = os.path.join(IMAGE_DIR, image_name)

    image = cv2.imread(image_path)

    if image is None:
        print("Could not open:", image_name)
        continue

    print("\nAnnotating:", image_name)
    print("Drag a box around the QR code.")
    print("Press ENTER when finished.")
    print("Press C to cancel.")

    #rectangle
    box = cv2.selectROI(
        "QR Annotation",
        image,
        fromCenter=False,
        showCrosshair=True
    )

    cv2.destroyWindow("QR Annotation")

    x, y, w, h = box

    #if no box was selected
    if w == 0 or h == 0:
        print("Skipped:", image_name)
        continue

    #image dimensions
    image_height, image_width = image.shape[:2]

    #convert to YOLO format
    center_x = (x + w / 2) / image_width
    center_y = (y + h / 2) / image_height

    box_width = w / image_width
    box_height = h / image_height

    #class 0 = QR
    label = f"0 {center_x} {center_y} {box_width} {box_height}"

    #create corresponding .txt filename
    label_name = os.path.splitext(image_name)[0] + ".txt"
    label_path = os.path.join(LABEL_DIR, label_name)

    with open(label_path, "w") as file:
        file.write(label)

    print("Saved:", label_path)

print("\nFinished annotating all images.")