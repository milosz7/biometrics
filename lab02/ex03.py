import sys
import matplotlib.pyplot as plt

import cv2 as cv
import numpy as np


def load_image(image_path):
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("file could not be read, check with os.path.exists()")
    return img


def threshold_otsu_binary_with_filters(img):
    # sharpen + blur
    img = cv.filter2D(img, -1, np.array([[0, -1, 0], [-1, 9, -1], [0, -1, 0]]))
    img = cv.blur(img, (11, 11), 0)
    threshold, masked_img = cv.threshold(
        img, 127, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU
    )

    return masked_img


def threshold_otsu_binary(img):
    threshold, masked_img = cv.threshold(
        img, 127, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU
    )
    return masked_img


def threshold_binary(img):
    threshold, masked_img = cv.threshold(img, 25, 255, cv.THRESH_BINARY_INV)
    return masked_img


def apply_connected_components(masked_img):
    result = cv.connectedComponentsWithStats(masked_img, 8, cv.CV_32S)
    return result


if __name__ == "__main__":
    raw_img = load_image(sys.argv[1])
    binarizers = [
        threshold_otsu_binary_with_filters,
        threshold_otsu_binary,
        threshold_binary,
    ]
    outputs = []
    for binarizer in binarizers:
        masked_img = binarizer(raw_img)
        analysis = apply_connected_components(masked_img)
        outputs.append((masked_img, analysis, binarizer.__name__))

    fig, axes = plt.subplots(len(binarizers), 3, figsize=(3 * 6, len(binarizers) * 4))

    for n, (masked_img, analysis, filter_name) in enumerate(outputs):
        n_labels, labels, values, centroids = analysis

        components_found = values[1:]
        areas = [value[cv.CC_STAT_AREA] for value in components_found]
        average_area = int(sum(areas) / len(areas))
        masks = np.clip(labels * 255, 0, 255)

        images = [raw_img, masked_img, masks]
        titles = [
            "Raw Image",
            f"Masked Image {filter_name}",
            f"Connected Components (avg object size: {average_area})",
        ]

        for i, (image, title) in enumerate(zip(images, titles)):
            axes[n, i].imshow(image, cmap="gray", vmin=0, vmax=255)
            axes[n, i].set_title(title)
            axes[n, i].axis("off")

        components_ax = axes[n, 2]
        for k, coords in enumerate(centroids[1:]):
            x, y = coords[0], coords[1]
            components_ax.text(x, y, f"{k + 1}", c="red", size=12)

    plt.tight_layout()
    plt.savefig("ex03_out.png")
