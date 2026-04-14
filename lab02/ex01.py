import sys

import cv2 as cv
from matplotlib import pyplot as plt


def add_weighted(im1, im2, alpha):
    if alpha < 0 or alpha > 1:
        raise ValueError("alpha must be between 0 and 1")

    beta = 1.0 - alpha
    out = cv.addWeighted(im1, alpha, im2, beta, 0)
    return out


def sum_clipped(im1, im2):
    return cv.add(im1, im2)


def sum_modulo(im1, im2):
    return im1 + im2


def plot_images(images, titles):
    fig, axes = plt.subplots(1, len(images), figsize=(len(images) * 4, 4))
    for i, (image, title) in enumerate(zip(images, titles)):
        axes[i].imshow(image, vmin=0, vmax=255)
        axes[i].set_title(title)
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig("ex01_out.png")
    plt.show()


def flip_channels(image):
    return image[:, :, ::-1]


def main():
    if len(sys.argv) != 3:
        print("Usage: python ex01.py <image1> <image2>")
        exit(1)

    im_1 = cv.imread(sys.argv[1])
    im_2 = cv.imread(sys.argv[2])

    img_sum = sum_clipped(im_1, im_2)
    img_modulo = sum_modulo(im_1, im_1)
    im_weighted_sum = add_weighted(img_sum, img_modulo, 0.7)

    titles = [
        "Image 1",
        "Image 2",
        "Image 1 + Image 2",
        "Image 1 + Image 2 mod 256",
        "Image 1 * 0.7 + Image 2 * 0.3",
    ]
    images = [im_1, im_2, img_sum, img_modulo, im_weighted_sum]
    images = [flip_channels(img) for img in images]

    plot_images(images, titles)


if __name__ == "__main__":
    main()
