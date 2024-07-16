from PIL import Image
import cv2


def crop_image(image, face):
    (x, y, w, h) = face
    cropped_image = image[y : y + h, x : x + w]  # Cropping the face from the image
    cropped_image_rgb = cv2.cvtColor(
        cropped_image, cv2.COLOR_BGR2RGB
    )  # Convert to RGB for PIL
    return Image.fromarray(cropped_image_rgb)
