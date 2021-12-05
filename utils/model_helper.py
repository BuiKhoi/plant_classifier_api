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


def predict_image(image):
    if image.shape[:2] != const.predict_size:
        image = cv2.resize(image, const.predict_size)

    return inferer.infer(conf, Image.fromarray(image[:, :, ::-1]), embeddings, True, labels)