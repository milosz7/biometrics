import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def gaussian_noise(image, level, probability):
    h, w = image.shape[:2]
    mask = np.random.uniform(0, 1, (h, w)) < probability
    if image.ndim == 3:
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    noise = np.random.randn(*image.shape) * level
    image = cv.add((noise * mask).astype(np.uint8), image)
    return np.clip(image, 0, 255).astype(np.uint8)


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
    plt.savefig("ex04_out.png")


if __name__ == "__main__":
    grayscale_lenna_img = cv.imread("super_big_Lenna.png", cv.IMREAD_GRAYSCALE)
    grayscale_lenna_gaussian_noise_img_01 = gaussian_noise(grayscale_lenna_img, 20, 0.1)
    grayscale_lenna_gaussian_noise_img_05 = gaussian_noise(grayscale_lenna_img, 20, 0.5)

    color_lenna_img = cv.imread("super_big_Lenna.png", cv.IMREAD_COLOR_RGB)
    color_lenna_gaussian_noise_img_01 = gaussian_noise(color_lenna_img, 20, 0.1)
    color_lenna_gaussian_noise_img_05 = gaussian_noise(color_lenna_img, 20, 0.5)

    titles = ["Basic", "Gaussian 20, 0.1", "Gaussian 20, 0.5"] * 2
    images = [
        grayscale_lenna_img,
        grayscale_lenna_gaussian_noise_img_01,
        grayscale_lenna_gaussian_noise_img_05,
        color_lenna_img,
        color_lenna_gaussian_noise_img_01,
        color_lenna_gaussian_noise_img_05,
    ]
    plot_images(images, titles)
