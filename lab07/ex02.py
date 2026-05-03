import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt


def detect_skin_hls(image_path):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")
    img_bgr = cv2.imread(image_path)
    img_hls = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS)
    H, L, S = cv2.split(img_hls)

    cond1 = S >= 0.2
    cond2 = (L > 0.5 * S) & (L < 3.0 * S)
    cond3 = (H <= 14) | (H >= 165)

    mask = (cond1 & cond2 & cond3).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    clean_mask = cv2.erode(mask, kernel, iterations=2)
    clean_mask = cv2.dilate(clean_mask, kernel, iterations=2)
    clean_mask = cv2.GaussianBlur(clean_mask, (3, 3), 0)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean_mask)
    result = cv2.bitwise_and(img_bgr, img_bgr, mask=clean_mask)
    titles = ["Raw image", "Raw HLS Mask", "Cleaned Mask", "Result"]
    images = [img_bgr, mask, clean_mask, result]
    images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]
    plot_images(images, titles)


def plot_images(images, titles):
    fig, axs = plt.subplots(1, len(images), figsize=(len(images) * 3, 4))
    for n, (img, ti) in enumerate(zip(images, titles)):
        axs[n].imshow(img, cmap="gray")
        axs[n].set_title(ti)
        axs[n].axis("off")
    plt.tight_layout()
    plt.savefig("ex01_out.png")


if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else "person.jpg"
    detect_skin_hls(image_path)
