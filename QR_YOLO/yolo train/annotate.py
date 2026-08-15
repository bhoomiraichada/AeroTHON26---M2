import cv2
import os

IMAGE_DIR = "qr codes"
LABEL_DIR = "labels"

os.makedirs(LABEL_DIR, exist_ok=True)

images = [
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png", ".heic"))
]

images.sort()

drawing = False
start_x = 0
start_y = 0
end_x = 0
end_y = 0


def mouse_callback(event, x, y, flags, param):
    global drawing, start_x, start_y, end_x, end_y

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x = x
        start_y = y
        end_x = x
        end_y = y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end_x = x
            end_y = y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_x = x
        end_y = y


for image_name in images:

    image_path = os.path.join(IMAGE_DIR, image_name)

    image = cv2.imread(image_path)

    if image is None:
        print("Could not open:", image_name)
        continue

    display = image.copy()

    drawing = False
    start_x = 0
    start_y = 0
    end_x = 0
    end_y = 0

    cv2.namedWindow("Annotate QR")
    cv2.setMouseCallback("Annotate QR", mouse_callback)

    print()
    print("IMAGE:", image_name)
    print("Drag around the QR code.")
    print("Press ENTER to save.")
    print("Press S to skip.")
    print("Press Q to quit.")

    while True:

        display = image.copy()

        if drawing or (end_x != start_x and end_y != start_y):

            cv2.rectangle(
                display,
                (start_x, start_y),
                (end_x, end_y),
                (0, 255, 0),
                2
            )

        cv2.imshow("Annotate QR", display)

        key = cv2.waitKey(20) & 0xFF

        # ENTER = save
        if key == 13:

            x1 = min(start_x, end_x)
            y1 = min(start_y, end_y)

            x2 = max(start_x, end_x)
            y2 = max(start_y, end_y)

            width = x2 - x1
            height = y2 - y1

            if width <= 0 or height <= 0:
                print("No valid box. Try again.")
                continue

            image_height, image_width = image.shape[:2]

            # YOLO format
            center_x = ((x1 + x2) / 2) / image_width
            center_y = ((y1 + y2) / 2) / image_height

            box_width = width / image_width
            box_height = height / image_height

            label = (
                f"0 {center_x:.6f} "
                f"{center_y:.6f} "
                f"{box_width:.6f} "
                f"{box_height:.6f}"
            )

            label_name = os.path.splitext(image_name)[0] + ".txt"
            label_path = os.path.join(LABEL_DIR, label_name)

            with open(label_path, "w") as file:
                file.write(label)

            print("SAVED:", label_path)
            print("LABEL:", label)

            break

        # S = skip
        elif key == ord("s"):
            print("SKIPPED:", image_name)
            break

        # Q = quit
        elif key == ord("q"):
            cv2.destroyAllWindows()
            print("Stopped.")
            raise SystemExit

    cv2.destroyAllWindows()

print()
print("DONE.")