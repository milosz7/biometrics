import sys

import cv2 as cv
import matplotlib.pyplot as plt


def count_grains(masked_img):
    result = cv.connectedComponentsWithStats(masked_img, 8, cv.CV_32S)
    return result


def main(img_path):
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    eroded = cv.erode(img, kernel, iterations=2)
    n_labels, labels, values, centroids = count_grains(eroded)

    fig, axes = plt.subplots(1, 2, figsize=(12, 3 * 4))
    axes[0].imshow(img, cmap="gray")
    axes[1].imshow(eroded, cmap="gray")
    axes[0].set_title("Raw image")
    axes[1].set_title(f"Counted grains (Result: {n_labels - 1})")
    axes[0].set_axis_off()
    axes[1].set_axis_off()

    components_ax = axes[1]
    # 1st is background
    for k, coords in enumerate(centroids[1:]):
        x, y = coords[0], coords[1]
        components_ax.text(x, y, f"{k + 1}", c="red", size=12)

    plt.tight_layout()
    plt.savefig("./ex01.png")
    plt.show()


if __name__ == "__main__":
    path = sys.argv[1]
    main(path)
