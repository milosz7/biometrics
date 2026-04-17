import os
import random
from collections import defaultdict
from utils import init_face_analysis_model
import cv2 as cv
from tqdm import tqdm

from config import PrepareDataConfig as PDC


def build_person_img_map(dir_path: str) -> dict[str, list[str]]:
    person_img_map = defaultdict(list)
    for filename in os.listdir(dir_path):
        if filename.endswith(".jpg"):
            person_id = filename.split("-")[0]
            img_path = os.path.join(dir_path, filename)
            person_img_map[person_id].append(img_path)
    return person_img_map


def calculate_rotation_score(angle: tuple[float, float, float]) -> float:
    yaw, pitch, roll = angle
    return abs(yaw) + abs(pitch) + abs(roll)


def build_path(root_dir: str, img_name: str) -> str:
    return os.path.join(root_dir, img_name)


def get_best_angle_imgs(
    img_map: dict[str, list[str]], n_imgs_per_class: int, split_frac = 0.75
) ->tuple[list[str], list[str]]:
    model = init_face_analysis_model(PDC)
    best_angle_imgs_train, best_angle_imgs_test = [], []
    for person_id, img_paths in tqdm(img_map.items()):
        images = [cv.imread(img_path) for img_path in img_paths]
        angles = []
        for image in images:
            try:
                result = model.get(image)[0]
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


def copy_imgs(
    paths: list[str], source_dir: str, output_dir: str
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for path in paths:
        filename = os.path.basename(path)
        source_path = os.path.join(source_dir, filename)
        output_path = os.path.join(output_dir, filename)
        cv.imwrite(output_path, cv.imread(source_path))


def main():
    users_img_map = build_person_img_map(PDC.USERS_DATA_RAW_DIR)
    users_train_data, users_test_data = get_best_angle_imgs(
        users_img_map, PDC.N_IMAGES_PER_USER
    )
    copy_imgs(
        users_train_data, PDC.USERS_DATA_RAW_DIR, PDC.TRAIN_DATA_OUTPUT_DIR
    )
    copy_imgs(
        users_test_data, PDC.USERS_DATA_RAW_DIR, PDC.TEST_DATA_OUTPUT_DIR
    )

    impostors_img_map = build_person_img_map(PDC.IMPOSTORS_DATA_RAW_DIR)
    impostors_test_data, _ = get_best_angle_imgs(
        impostors_img_map, PDC.N_IMAGES_PER_IMPOSTOR, split_frac=1.0
    )
    copy_imgs(
        impostors_test_data, PDC.IMPOSTORS_DATA_RAW_DIR, PDC.TEST_DATA_OUTPUT_DIR
    )


if __name__ == "__main__":
    main()
