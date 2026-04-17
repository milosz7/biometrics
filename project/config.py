class PrepareDataConfig:
    FACE_ANALYSIS_MODEL = "buffalo_s"
    FACE_ANALYSIS_MODEL_MODULES = ["detection", "landmark_3d_68"]
    USERS_DATA_RAW_DIR = "data/users"
    IMPOSTORS_DATA_RAW_DIR = "data/impostors"
    TRAIN_DATA_OUTPUT_DIR = "data/train"
    TEST_DATA_OUTPUT_DIR = "data/test"
    N_IMAGES_PER_USER = 6
    N_IMAGES_PER_IMPOSTOR = 2
