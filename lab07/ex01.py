import requests
import os
import cv2
import numpy as np
from collections import Counter


def download_dataset():
    emojis = {
        "angry": [
            "1f621",
            "1f620",
            "1f92c",
            "1f624",
            "1f611",
            "1f92f",
            "1f595",
            "1f608",
            "1f47f",
            "1f4a2",
        ],
        "love": [
            "1f60d",
            "1f618",
            "1f617",
            "1f619",
            "1f496",
            "1f495",
            "1f49c",
            "1f49b",
            "1f9e1",
            "2764",
            "1f493",
        ],
        "funny": [
            "1f602",
            "1f923",
            "1f61c",
            "1f61d",
            "1f92a",
            "1f92b",
            "1f607",
            "1f609",
            "1f60f",
            "1f643",
        ],
        "neutral": [
            "1f610",
            "1f611",
            "1f636",
            "1f644",
            "1f914",
            "1f928",
            "1f610",
            "1f612",
            "1f615",
            "1f616",
        ],
    }

    base_url = "https://twemoji.maxcdn.com/v/latest/72x72/"

    for label, codes in emojis.items():
        os.makedirs(f"dataset/{label}", exist_ok=True)
        for i, code in enumerate(codes):
            url = base_url + code + ".png"
            try:
                img = requests.get(url, timeout=10).content
                with open(f"dataset/{label}/{i}.png", "wb") as f:
                    f.write(img)
            except Exception as e:
                print(f"Failed to download {url}: {e}")


def histogram_intersection_distance(I, M):
    intersection = np.sum(np.minimum(I, M))
    sum_M = np.sum(M)
    if sum_M == 0:
        return 1.0
    similarity = intersection / sum_M
    distance = 1.0 - similarity
    return distance


def extract_histogram(img_path, mode="grayscale", bins=64):
    img = cv2.imread(img_path)
    if img is None:
        return None

    if mode == "grayscale":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [bins], [0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist.flatten()
    elif mode == "hog":
        img = cv2.resize(img, (64, 128))
        hog = cv2.HOGDescriptor()
        h = hog.compute(img)
        if h is not None and len(h) > 0:
            h = h.flatten()
            norm = np.linalg.norm(h)
            if norm > 0:
                h = h / norm
            return h
        return None

    return None


def evaluate(dataset_path="dataset", mode="grayscale", k=3):
    features = []
    labels = []

    for label in os.listdir(dataset_path):
        label_dir = os.path.join(dataset_path, label)
        if os.path.isdir(label_dir):
            for filename in os.listdir(label_dir):
                if filename.endswith(".png") or filename.endswith(".jpg"):
                    filepath = os.path.join(label_dir, filename)
                    hist = extract_histogram(filepath, mode)
                    if hist is not None:
                        features.append(hist)
                        labels.append(label)

    correct = 0
    total = len(features)

    class_correct = Counter()
    class_total = Counter()

    for i in range(total):
        test_feature = features[i]
        test_label = labels[i]

        distances = []
        for j in range(total):
            if i != j:
                dist = histogram_intersection_distance(test_feature, features[j])
                distances.append((dist, labels[j]))

        distances.sort(key=lambda x: x[0])

        k_nearest_labels = [d[1] for d in distances[:k]]
        most_common = Counter(k_nearest_labels).most_common(1)[0][0]

        class_total[test_label] += 1
        if most_common == test_label:
            correct += 1
            class_correct[test_label] += 1

    accuracy = correct / total if total > 0 else 0
    print(f"\nAccuracy ({mode} histograms, k={k}): {accuracy:.2%} ({correct}/{total})")
    print("Accuracy per class:")
    for cls in sorted(class_total.keys()):
        cls_acc = class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0
        print(f"  {cls}: {cls_acc:.2%} ({class_correct[cls]}/{class_total[cls]})")
    return accuracy


if __name__ == "__main__":
    if not os.path.exists("dataset"):
        print("Dataset not found, downloading...")
        download_dataset()

    print("Evaluating basic grayscale histogram comparison...")
    evaluate(mode="grayscale", k=1)

    print("\nImprovement: Using HOG (Histogram of Oriented Gradients)")
    evaluate(mode="hog", k=1)
    evaluate(mode="hog", k=3)
