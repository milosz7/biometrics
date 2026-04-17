import sys

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def load_image(image_path):
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    return img


def get_thresholds(img):
    thresh = np.mean(img)
    pupil_thresh = thresh / 4.5
    iris_thresh = thresh / 1.5
    return pupil_thresh, iris_thresh


def binarize_pupil(img):
    pupil_thresh, _ = get_thresholds(img)

    _, masked_img = cv.threshold(img, pupil_thresh, 255, cv.THRESH_BINARY_INV)
    return masked_img


def binarize_iris(img):
    _, iris_thresh = get_thresholds(img)

    _, masked_img = cv.threshold(img, iris_thresh, 255, cv.THRESH_BINARY_INV)
    return masked_img


def find_enclosing_circle(img):
    contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv.contourArea)
        (x, y), radius = cv.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)
        return center, radius
    return None, None


def run_detection(img):
    binarized_pupil = binarize_pupil(img)
    thresh1, thresh2 = 100, 150
    pupil_edges = cv.Canny(binarized_pupil, thresh1, thresh2, apertureSize=3)

    binarized_iris = binarize_iris(img)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    eroded = cv.erode(binarized_iris, kernel, iterations=4)
    iris_edges = cv.Canny(eroded, thresh1, thresh2, apertureSize=3)

    pupil_circle = find_enclosing_circle(pupil_edges)
    iris_circle = find_enclosing_circle(iris_edges)
    return pupil_circle, iris_circle


def draw_detected_circles(img, iris_circle, pupil_circle):
    (i_cx, i_cy), i_radius = iris_circle
    (p_cx, p_cy), p_radius = pupil_circle
    cv.circle(img, (i_cx, i_cy), i_radius, (255, 0, 0), 1)
    cv.circle(img, (p_cx, p_cy), p_radius, (0, 255, 0), 1)
    return img


def plot_images(images, titles):
    fig, axs = plt.subplots(1, len(images), figsize=(len(images) * 3, 4))
    for n, (img, ti) in enumerate(zip(images, titles)):
        axs[n].imshow(img, cmap="gray")
        axs[n].set_title(ti)
        axs[n].axis("off")
    plt.tight_layout()
    plt.savefig("ex01_out.png")


def main(img_path):
    img = load_image(img_path)
    pupil_circle_raw, iris_circle_raw = run_detection(img)
    blur_kernel = np.array(
        [
            [0.037, 0.039, 0.04, 0.039, 0.037],
            [0.039, 0.042, 0.042, 0.042, 0.039],
            [0.04, 0.042, 0.043, 0.042, 0.04],
            [0.039, 0.042, 0.042, 0.042, 0.039],
            [0.037, 0.039, 0.04, 0.039, 0.037],
        ]
    )
    img_blurred = cv.filter2D(img, -1, blur_kernel)
    pupil_circle_blurred, iris_circle_blurred = run_detection(img_blurred)
    img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)

    detected_circles = draw_detected_circles(
        img.copy(), iris_circle_raw, pupil_circle_raw
    )
    detected_circles_blurred = draw_detected_circles(
        img.copy(), iris_circle_blurred, pupil_circle_blurred
    )
    titles = ["Detected Circles on raw image", "Detected Circles on blurred image"]
    images = [detected_circles, detected_circles_blurred]
    plot_images(images, titles)


if __name__ == "__main__":
    img_path = sys.argv[1]
    main(img_path)
