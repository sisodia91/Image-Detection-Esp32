from ultralytics import YOLO
import cv2
import serial
import time

# =================================================
# YOLO MODEL
# =================================================
model = YOLO("models/yolo11n.pt")


# =================================================
# PHONE CAMERA
# =================================================
camera_url = "http://10.189.182.191:8080/video"

cap = cv2.VideoCapture(camera_url)

if not cap.isOpened():
    print(" Could not connect to phone camera")
    exit()


# =================================================
# ESP32 SERIAL
# =================================================
# CHANGE COM3 to the COM port of your ESP32
ser = serial.Serial(
    "COM15",
    115200,
    timeout=1
)

print(" ESP32 serial connected")


# =================================================
# DETECTION SETTINGS
# =================================================
CONFIDENCE_THRESHOLD = 0.50

# Send one message every second
SEND_INTERVAL = 1.0


# =================================================
# DISTANCE CALIBRATION
# =================================================
# TEMPORARY VALUE
#
# Example:
# Person is 2 meters away
# Bounding box height = 300 pixels
#
# K = 2 × 300 = 600
#
# distance = K / bbox_height
#
# We will calibrate this properly later.
# =================================================
CALIBRATION_CONSTANT = 600.0


last_send_time = 0


# =================================================
# MAIN LOOP
# =================================================
while True:

    ret, frame = cap.read()

    if not ret:
        print(" Camera disconnected")
        break


    # =================================================
    # YOLO DETECTION
    # =================================================
    results = model(
        frame,
        verbose=False
    )

    annotated_frame = results[0].plot()


    # =================================================
    # DEFAULT STATE
    # =================================================
    person_detected = False
    estimated_distance = None


    # =================================================
    # CHECK DETECTED OBJECTS
    # =================================================
    for box in results[0].boxes:

        class_id = int(box.cls[0])

        confidence = float(box.conf[0])

        class_name = model.names[class_id]


        # Only interested in PERSON
        if (
            class_name == "person"
            and confidence >= CONFIDENCE_THRESHOLD
        ):

            person_detected = True


            # -----------------------------------------
            # Bounding box coordinates
            # -----------------------------------------
            x1, y1, x2, y2 = box.xyxy[0].tolist()


            # Height of person's bounding box
            bbox_height = y2 - y1


            # -----------------------------------------
            # Estimate distance
            # -----------------------------------------
            if bbox_height > 0:

                estimated_distance = (
                    CALIBRATION_CONSTANT / bbox_height
                )


            break


    # =================================================
    # SEND DATA TO ESP32
    # =================================================
    current_time = time.time()


    if current_time - last_send_time >= SEND_INTERVAL:


        if person_detected:

            if estimated_distance is not None:

                message = (
                    f"PERSON,{estimated_distance:.2f}\n"
                )


                # Show distance on laptop screen
                cv2.putText(
                    annotated_frame,
                    f"Distance: {estimated_distance:.2f} m",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

            else:

                message = "PERSON,UNKNOWN\n"


        else:

            message = "NO_PERSON\n"


        # Send message to ESP32
        ser.write(message.encode())


        print(
            "Sent to ESP32:",
            message.strip()
        )


        last_send_time = current_time


    # =================================================
    # DISPLAY CAMERA
    # =================================================
    cv2.imshow(
        "Aryan YOLO Person Detection",
        annotated_frame
    )


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =================================================
# CLEANUP
# =================================================
cap.release()

ser.close()

cv2.destroyAllWindows()