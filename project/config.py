import os
from dataclasses import dataclass


@dataclass
class BaseConfig:
    FACE_ANALYSIS_MODEL = "buffalo_l"
    FACE_ANALYSIS_MODEL_MODULES = ["detection", "recognition"]
    DATA_ROOT = "data"
    STORAGE_PATH = "storage.pkl"

    @property
    def TRAIN_DATA_DIR(self) -> str:
        return str(os.path.join(self.DATA_ROOT, "train"))

    @property
    def TEST_DATA_DIR(self) -> str:
        return str(os.path.join(self.DATA_ROOT, "test"))

    @property
    def IMPOSTORS_DATA_DIR(self) -> str:
        return str(os.path.join(self.TEST_DATA_DIR, "impostors"))

    @property
    def USERS_DATA_DIR(self) -> str:
        return str(os.path.join(self.TEST_DATA_DIR, "authorized"))


class PrepareDataConfig(BaseConfig):
    FACE_ANALYSIS_MODEL = "buffalo_s"
    FACE_ANALYSIS_MODEL_MODULES = ["detection", "landmark_3d_68"]

    @property
    def USERS_DATA_RAW_DIR(self) -> str:
        return str(os.path.join(self.DATA_ROOT, "users"))

    @property
    def IMPOSTORS_DATA_RAW_DIR(self) -> str:
        return str(os.path.join(self.DATA_ROOT, "impostors"))

    N_IMAGES_PER_USER = 6
    N_IMAGES_PER_IMPOSTOR = 2


class BuildStorageConfig(BaseConfig):
    pass


class EvaluationConfig(BaseConfig):
    THRESHOLD = 0.7
