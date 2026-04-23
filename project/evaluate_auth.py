from collections import defaultdict

from tqdm import tqdm

from embedding_model import EmbeddingModel
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_curve, auc
import os
import cv2 as cv
from config import EvaluationConfig
from utils import load_database, calculate_cosine_similarity
import numpy as np
import matplotlib.pyplot as plt

type ImageDataset = dict[str, list[np.ndarray]]


def extract_user_from_path(img_path: str) -> str:
    filename = os.path.basename(img_path)
    user = filename.split("-")[0]
    return user


def load_images(path: str) -> ImageDataset:
    paths = os.listdir(path)
    paths = [
        os.path.join(path, img_path) for img_path in paths if img_path.endswith(".jpg")
    ]
    images = defaultdict(list)
    for path in paths:
        user = extract_user_from_path(path)
        img = cv.imread(path)
        if img is not None:
            images[user].append(img)
    return images


def evaluate_impostors(
    model: EmbeddingModel,
    impostors_dataset: ImageDataset,
    embeddings: np.ndarray,
    threshold: float,
) -> tuple[list[float], list[float]]:
    results, probs = [], []
    for _user, images in tqdm(impostors_dataset.items()):
        for img in images:
            try:
                emb = model(img)
            except Exception as e:
                print(f"Error processing image: {e}")
                results.append(0.0)
                probs.append(0.0)
                continue
            cosine_similarity = calculate_cosine_similarity(emb, embeddings)
            prediction = np.argmax(cosine_similarity)

            if cosine_similarity[prediction] >= threshold:
                # if any of the impostor's images are classified as authorized, we consider the impostor successful
                results.append(1.0)
            else:
                # user does not have access
                results.append(0.0)
            probs.append(cosine_similarity[prediction])
    return results, probs


def evaluate_authorized(
    model: EmbeddingModel,
    users_dataset: ImageDataset,
    row_to_user_map: dict[int, str],
    embeddings: np.ndarray,
    threshold: float,
) -> tuple[list[float], list[float]]:
    results = []
    probs = []
    for user, images in tqdm(users_dataset.items()):
        for img in images:
            try:
                emb = model(img)
            except Exception as e:
                print(f"Error processing image: {e}")
                results.append(0.0)
                probs.append(0.0)
                continue
            cosine_similarity = calculate_cosine_similarity(emb, embeddings)
            prediction = np.argmax(cosine_similarity)
            predicted_user = row_to_user_map[prediction]

            if cosine_similarity[prediction] >= threshold and predicted_user == user:
                # if any of the user's images are classified as authorized and
                # the predicted user matches the actual user, we consider the user successful
                results.append(1.0)
            else:
                results.append(0.0)
            probs.append(cosine_similarity[prediction])
    return results, probs


def get_num_imgs(dataset: dict[str, list[np.ndarray]]) -> int:
    return sum(len(images) for images in dataset.values())


def evaluate_auth(config: EvaluationConfig) -> None:
    model = EmbeddingModel(config)
    impostors_dataset = load_images(config.IMPOSTORS_DATA_DIR)
    users_dataset = load_images(config.USERS_DATA_DIR)
    row_to_user_map, embeddings = load_database(config)
    y_true = [0.0] * get_num_imgs(impostors_dataset) + [1.0] * get_num_imgs(
        users_dataset
    )

    threshold = config.THRESHOLD
    y_pred_impostors, y_probs_impostors = evaluate_impostors(
        model, impostors_dataset, embeddings, threshold
    )
    y_pred_users, y_probs_users = evaluate_authorized(
        model, users_dataset, row_to_user_map, embeddings, threshold
    )
    y_pred = y_pred_impostors + y_pred_users
    y_pred_proba = y_probs_impostors + y_probs_users

    ba = balanced_accuracy_score(y_true, y_pred)
    plot_roc(y_true, y_pred_proba)

    print(f"Balanced Accuracy: {ba:.4f}")


def plot_roc(y_true: list[float], y_pred_proba: list[float]) -> None:
    y_pred_proba = [(x + 1.0) / 2.0 for x in y_pred_proba]
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], "k--", label="Random guess")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for Face Authentication System")
    plt.legend()
    plt.savefig("roc_curve.png")


if __name__ == "__main__":
    config = EvaluationConfig()
    evaluate_auth(config)
