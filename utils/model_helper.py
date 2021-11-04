import numpy as np
from PIL import Image

from utils import constants as const
from .trainer import face_learner
from .config import get_config


databank = np.load(const.databank_path, allow_pickle=True)
embeddings = databank["embeddings"]
embeddings = np.squeeze(embeddings, 1)
labels = databank["labels"]

conf = get_config(False)
conf.use_mobilfacenet = True
inferer = face_learner(conf, True)
inferer.load_state(conf, const.model_path, False, True, absolute=True)


def predict_image(image):
    min_idx, _ = inferer.infer(conf, [Image.fromarray(image)], embeddings, True)

    return labels[min_idx[0]]