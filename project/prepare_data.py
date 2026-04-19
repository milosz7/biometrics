import os
import random

from project.embedding_model import EmbeddingModel
from utils import build_person_img_map
import cv2 as cv
from tqdm import tqdm

from config import PrepareDataConfig


def calculate_rotation_score(angle: tuple[float, float, float]) -> float:
    yaw, pitch, roll = angle
    return abs(yaw) + abs(pitch) + abs(roll)


def build_path(root_dir: str, img_name: str) -> str:
    return os.path.join(root_dir, img_name)


def get_best_angle_imgs(
    img_map: dict[str, list[str]], n_imgs_per_class: int, split_frac=0.75
) -> tuple[list[str], list[str]]:
    config = PrepareDataConfig()
    model = EmbeddingModel(config)
    best_angle_imgs_train, best_angle_imgs_test = [], []
    for person_id, img_paths in tqdm(img_map.items()):
        images = [cv.imread(img_path) for img_path in img_paths]
        angles = []
        for image in images:
            try:
                result = model(image)
                angles.append(result["pose"])
            except IndexError:  # detection failed
                bad_pose = (90.0, 90.0, 90.0)  # worst possible pose
                angles.append(bad_pose)

        path_angle_pairs = list(zip(img_paths, angles))
        best_pairs = sorted(
            path_angle_pairs, key=lambda pair: calculate_rotation_score(pair[1])
        )[:n_imgs_per_class]
        best_images = [path for path, _angle in best_pairs]

        random.shuffle(best_images)
        split_idx = int(len(best_images) * split_frac)
        best_angle_imgs_train.extend(best_images[:split_idx])
        best_angle_imgs_test.extend(best_images[split_idx:])

    return best_angle_imgs_train, best_angle_imgs_test


def copy_imgs(paths: list[str], source_dir: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for path in paths:
        filename = os.path.basename(path)
        source_path = os.path.join(source_dir, filename)
        output_path = os.path.join(output_dir, filename)
        cv.imwrite(output_path, cv.imread(source_path))


def main():
    config = PrepareDataConfig()
    users_img_map = build_person_img_map(config.USERS_DATA_RAW_DIR)
    users_train_data, users_test_data = get_best_angle_imgs(
        users_img_map, config.N_IMAGES_PER_USER
    )
    copy_imgs(users_train_data, config.USERS_DATA_RAW_DIR, config.TRAIN_DATA_DIR)
    copy_imgs(users_test_data, config.USERS_DATA_RAW_DIR, config.USERS_DATA_DIR)

    impostors_img_map = build_person_img_map(config.IMPOSTORS_DATA_RAW_DIR)
    impostors_test_data, _ = get_best_angle_imgs(
        impostors_img_map, config.N_IMAGES_PER_IMPOSTOR, split_frac=1.0
    )
    copy_imgs(
        impostors_test_data, config.IMPOSTORS_DATA_RAW_DIR, config.IMPOSTORS_DATA_DIR
    )


if __name__ == "__main__":
    main()
