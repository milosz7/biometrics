from insightface.app import FaceAnalysis
import numpy as np


class EmbeddingModel:
    def __init__(self, config):
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        self.model = FaceAnalysis(
            name=config.FACE_ANALYSIS_MODEL,
            providers=providers,
            allowed_modules=config.FACE_ANALYSIS_MODEL_MODULES,
        )
        self.model.prepare(ctx_id=-1)

    def get_pose(self, x: np.ndarray) -> tuple[float, float, float]:
        result = self.model.get(x)[0]
        return result["pose"]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        result = self.model.get(x)[0]
        return result["embedding"]
