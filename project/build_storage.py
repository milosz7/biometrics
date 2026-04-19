import pickle
import cv2
import numpy as np

from config import BuildStorageConfig
from tqdm import tqdm
from utils import build_person_img_map
from embedding_model import EmbeddingModel


def build_face_vector_storage(config):
    model = EmbeddingModel(config)
    user_ids, embeddings = [], []

    person_to_imgs_map = build_person_img_map(config.TRAIN_DATA_DIR)
    for person_id, img_paths in tqdm(person_to_imgs_map.items()):
        for path in img_paths:
            img = cv2.imread(path)
            embedding = model(img)
            embeddings.append(embedding)
            user_ids.append(person_id)
    embeddings = np.vstack(embeddings)
    row_to_user_map = {i: user_id for i, user_id in enumerate(user_ids)}

    return row_to_user_map, embeddings


if __name__ == "__main__":
    config = BuildStorageConfig()
    storage = build_face_vector_storage(config)
    with open(BuildStorageConfig.STORAGE_PATH, "wb") as f:
        pickle.dump(storage, f, protocol=pickle.HIGHEST_PROTOCOL)
