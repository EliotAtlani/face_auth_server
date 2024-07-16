import os
import cv2
import numpy as np

# Define the path to the Haar Cascade XML file
cascade_path = os.path.join(
    os.path.dirname(__file__), "haarcascade_frontalface_default.xml"
)
if not os.path.exists(cascade_path):
    raise FileNotFoundError(f"Haar Cascade file not found at {cascade_path}")

# Load the Haar Cascade classifier
haar_cascade = cv2.CascadeClassifier(cascade_path)
if haar_cascade.empty():
    raise ValueError("Failed to load Haar Cascade classifier")


def detect_faces(image):
    # Convert PIL Image to OpenCV format
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    gray_img = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)

    faces = haar_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)

    return faces, opencv_image
