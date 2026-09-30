import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load trained model

model = tf.keras.models.load_model(
    "models/sho_classifier.keras",
    custom_objects={"preprocess_input": preprocess_input},
    safe_mode=False
) 

# Class labels (same order as training)
train_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    image_size=(128,128),
    batch_size=32
)

class_names = train_ds.class_names

class_name_mapping = {
    "BallMain": "Ball",
    "ConsoleMain": "Console",
    "EarmuffMain": "Earmuff",
    "Full Water level": "Bottle",
    "Full  Water level": "Bottle",
    "GamepadMain": "Gamepad",
    "HandwatchMain": "Handwatch",
    "MicMain": "Microphone",
    "QuadcopterMain": "Quadcopter",
    "SharpenerMain": "Sharpener",
    "StatueMain": "Statue",
    "SunglassMain": "Sunglass",
    "TripodMain": "Tripod",
    "VitaminMain": "Vitamin",
    "WebcamMain": "Webcam"
}

print(class_names)

# Replace with your phone IP Webcam address
camera_url = "http://192.168.1.3:8080/video"

cap = cv2.VideoCapture(camera_url)

while True:

    ret, frame = cap.read()

    if not ret:
        print("❌ Camera not connected.")
        break

    img = cv2.resize(frame, (128, 128))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    rgb = np.expand_dims(rgb, axis=0)
    rgb = preprocess_input(rgb)

    prediction = model.predict(rgb, verbose=0)

    index = np.argmax(prediction)
    confidence = prediction[0][index]

    raw_label = class_names[index]
    label = class_name_mapping.get(raw_label, raw_label)

    # If confidence is low, don't trust the prediction
    if confidence < 0.70:
        label = "Unknown Object"

    # Display prediction
    cv2.putText(
        frame,
        f"{label} : {confidence * 100:.1f}%",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Aryan AI Detector", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Close camera and window AFTER loop ends
cap.release()
cv2.destroyAllWindows()