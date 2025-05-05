import cv2
import pyttsx3
import time
from datetime import datetime

# Load MobileNetSSD model
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt', 'MobileNetSSD_deploy.caffemodel')

# Class labels
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Start webcam
cap = cv2.VideoCapture(0)
detected_objects = set()

def speak(text):
    engine.say(text)
    engine.runAndWait()

def log_detection(obj):
    with open("detection_log.txt", "a") as f:
        f.write(f"[{datetime.now().strftime('%H:%M:%S')}] Detected: {obj}\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]

    # Preprocess frame for object detection
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 
                                 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    detected_this_frame = set()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx]
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (startX, startY, endX, endY) = box.astype("int")

            # Draw bounding box and label
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 255, 0), 2)
            text = f"{label}: {int(confidence * 100)}%"
            cv2.putText(frame, text, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Approximate distance
            box_height = endY - startY
            approx_distance = round(1000 / (box_height + 1), 1)  # Simplified estimate

            message = f"There is a {label} at about {int(approx_distance)} centimeters"

            # Speak only new detections
            if label not in detected_objects:
                speak(message)
                log_detection(label)
                detected_objects.add(label)

            detected_this_frame.add(label)

    # Reset detection after a few seconds
    if not detected_this_frame:
        detected_objects.clear()

    cv2.imshow("Object Detection with Voice", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()