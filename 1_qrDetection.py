import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture(0)

model = YOLO("QR_YOLO/runs/detect/train/weights/best.pt")

qr_det = cv2.QRCodeDetector()

target_location = None

initial_qr_scanned = False

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not access camera")
        break


    results = model.predict(frame,conf=0.15,verbose=False)

    display = frame.copy()

    qr_found_by_yolo = False
    decoded_data = None
    qr_points = None


    for result in results:

        boxes = result.boxes

        if boxes is None:
            continue

        for box in boxes:

            #yolo confidence
            confidence = float(box.conf[0])

            #coords
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            #box remain insife 
            x1 = max(0,x1)
            y1 = max(0,y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            #yolo box

            cv2.rectangle(display,(x1, y1),(x2, y2),(255, 0, 0),3)

            cv2.putText(display,f"YOLO QR {confidence:.2f}",(x1, max(25, y1 - 10)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255, 0, 0),2)

            qr_found_by_yolo = True

            padding = 20

            crop_x1 = max(0, x1 - padding)
            crop_y1 = max(0, y1 - padding)
            crop_x2 = min(frame.shape[1], x2 + padding)
            crop_y2 = min(frame.shape[0], y2 + padding)

            qr_crop = frame[
                crop_y1:crop_y2,
                crop_x1:crop_x2
            ]

             #opecv qr decoding
            data, points, _ = qr_det.detectAndDecode(qr_crop)

            if data:

                decoded_data = data

                points = points.astype(int)

                points[:, :, 0] += crop_x1
                points[:, :, 1] += crop_y1

                qr_points = points

            break

        if qr_found_by_yolo:
            break


    #yolo no qr
    if not qr_found_by_yolo:

        if not initial_qr_scanned:

            cv2.putText(display,"SEARCHING FOR INITIAL QR...",(20, 40),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0, 255),2)

        else:

            cv2.putText(display,"SEARCHING FOR MATCHING QR...",(20, 40),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 0, 255),2)


    #yolo finds qr but no data
    elif decoded_data is None:

        cv2.putText(display,"QR DETECTED - DECODING...",(20, 40),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 165, 255),2)


    #data decoded
    else:

        #initial qr
        if not initial_qr_scanned:

            target_location = decoded_data
            initial_qr_scanned = True

            print()
            print("INITIAL QR SCANNED")
            print("Target location:", target_location)

            cv2.putText(display,"INITIAL QR SCANNED",(20, 40),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 0, 255),2)

            cv2.putText(display,"TARGET SAVED: " + target_location,(20, 75),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0, 0, 255),2)

        #matching qr
        else:

            if decoded_data == target_location:

                print("MATCHING QR FOUND:", decoded_data)

                cv2.putText(display,"MATCHING QR FOUND!",(20, 40),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 255, 0),2)

                cv2.putText(display,"TARGET: " + target_location,(20, 75),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0, 255, 0),2)

                #Green QR outline
                if qr_points is not None:

                    for i in range(4):

                        pt1 = tuple(qr_points[0][i])
                        pt2 = tuple(
                            qr_points[0][(i + 1) % 4]
                        )

                        cv2.line(display,pt1,pt2,(0, 255, 0),4)

            #wrong qr
            else:

                print("WRONG QR:", decoded_data)

                cv2.putText(
                    display,
                    "WRONG QR",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    display,
                    "DETECTED: " + decoded_data,
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

                # Red QR outline
                if qr_points is not None:

                    for i in range(4):

                        pt1 = tuple(qr_points[0][i])
                        pt2 = tuple(
                            qr_points[0][(i + 1) % 4]
                        )

                        cv2.line(
                            display,
                            pt1,
                            pt2,
                            (0, 0, 255),
                            4
                        )


    #saved targed
    if target_location is not None:

        cv2.putText(
            display,
            "TARGET: " + target_location,
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )




    cv2.imshow("M2", display)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()