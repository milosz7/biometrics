from insightface.app import FaceAnalysis
from collections import defaultdict
import os
import pickle
import numpy as np


def init_face_analysis_model(config):
    providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    model = FaceAnalysis(
        name=config.FACE_ANALYSIS_MODEL,
        providers=providers,
        allowed_modules=config.FACE_ANALYSIS_MODEL_MODULES,
    )
    model.prepare(ctx_id=-1)
    return model


def build_person_img_map(dir_path: str) -> dict[str, list[str]]:
    try:
        person_img_map = defaultdict(list)
        for filename in os.listdir(dir_path):
            if filename.endswith(".jpg"):
                person_id = filename.split("-")[0]
                img_path = os.path.join(dir_path, filename)
                person_img_map[person_id].append(img_path)
        return person_img_map
    except FileNotFoundError:
        print(f"Directory not found: {dir_path}. Maybe you forgot to prepare the data?")
        raise


def load_database(config):
    with open(config.STORAGE_PATH, "rb") as f:
        row_to_user_map, embeddings = pickle.load(f)
    return row_to_user_map, embeddings


def calculate_cosine_similarity(emb, vectors):
    emb_norm = np.linalg.norm(emb)
    vectors_norm = np.linalg.norm(vectors, axis=1)
    dot_product = np.dot(vectors, emb)
    cosine_similarity = dot_product / (vectors_norm * emb_norm + 1e-10)
    return cosine_similarity
