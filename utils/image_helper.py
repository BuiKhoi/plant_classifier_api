import cv2
import numpy as np


async def get_image_file(file):
    # Check file extension
    extension = file.filename.split(".")[-1].lower() in ("jpg", "jpeg", "png")
    if not extension:
        raise NotImplementedError(f"Extension {extension} is not supported")

    # Read file content into numpy array
    try:
        contents = await file.read()
        image = cv2.imdecode(np.asarray(bytearray(contents), dtype=np.uint8), 1)
    except Exception as e:
        raise Exception("Cannot parse image")

    # Check grayscale image
    if len(image.shape) != 3:
        raise Exception("Image is in grayscale")

    return image