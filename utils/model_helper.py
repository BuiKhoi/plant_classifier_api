import cv2
import numpy as np
from PIL import Image

from utils import constants as const
from .trainer import face_learner
from .config import get_config


databank = np.load(const.databank_path, allow_pickle=True)
embeddings = databank["embeddings"]
embeddings = np.squeeze(embeddings, 1)
labels = databank["labels"]

conf = get_config()
inferer = face_learner(conf, const.strange_threshold)
inferer.load_state(const.model_path)


def center_crop_image(im):
    h, w, _ = im.shape
    smaller_size = min(w, h)
    new_size = int(smaller_size*0.8)

    width_start = int((w-new_size)/2)
    width_stop = (width_start + new_size)

    height_start = int((h-new_size)/2)
    height_stop = height_start + new_size

    return im[height_start:height_stop, width_start:width_stop, :]


def predict_image(image):
    if image.shape[:2] != const.predict_size:
        image = cv2.resize(center_crop_image(image), const.predict_size)

    return inferer.infer(conf, Image.fromarray(image[:, :, ::-1]), embeddings, True, labels)