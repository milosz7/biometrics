import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def salt_and_pepper_noise(image, probability):
    h, w = image.shape[:2]
    image = image.copy()
    salt_mask = np.random.uniform(0, 1, (h, w))
    salt_mask = salt_mask < probability / 2
    pepper_mask = np.random.uniform(0, 1, (h, w))
    pepper_mask = pepper_mask < probability / 2
    if image.ndim == 3:
        salt_mask = np.repeat(salt_mask[:, :, np.newaxis], 3, axis=2)
        pepper_mask = np.repeat(pepper_mask[:, :, np.newaxis], 3, axis=2)
    image[salt_mask] = 255
    image[pepper_mask] = 0
    return image


def plot_images(images, titles):
    fig, axes = plt.subplots(2, len(images) // 2, figsize=(len(images) * 3 / 2, 8))
    axes = axes.ravel()
    for i, (image, title) in enumerate(zip(images, titles)):
        if image.ndim == 2:
            axes[i].imshow(image, cmap="gray", vmin=0, vmax=255)
        else:
            axes[i].imshow(image, vmin=0, vmax=255)
        axes[i].set_title(title)
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig("ex05_out.png")


if __name__ == "__main__":
    grayscale_lenna_img = cv.imread("super_big_Lenna.png", cv.IMREAD_GRAYSCALE)
    grayscale_lenna_salt_pepper_img_01 = salt_and_pepper_noise(grayscale_lenna_img, 0.1)
    grayscale_lenna_salt_pepper_img_05 = salt_and_pepper_noise(grayscale_lenna_img, 0.5)

    color_lenna_img = cv.imread("super_big_Lenna.png", cv.IMREAD_COLOR_RGB)
    color_lenna_salt_pepper_img_01 = salt_and_pepper_noise(color_lenna_img, 0.1)
    color_lenna_salt_pepper_img_05 = salt_and_pepper_noise(color_lenna_img, 0.5)

    titles = ["Basic", "Salt & Pepper 0.1", "Salt & Pepper 0.5"] * 2
    images = [
        grayscale_lenna_img,
        grayscale_lenna_salt_pepper_img_01,
        grayscale_lenna_salt_pepper_img_05,
        color_lenna_img,
        color_lenna_salt_pepper_img_01,
        color_lenna_salt_pepper_img_05,
    ]
    plot_images(images, titles)
