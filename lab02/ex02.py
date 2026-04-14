import matplotlib.pyplot as plt
import argparse

import cv2
import numpy as np
from abc import ABC, abstractmethod


class Process(ABC):
    @abstractmethod
    def _apply_filter(self, img, kernel, normalize=True):
        raise NotImplementedError


class Filter(Process):
    def __init__(self):
        super().__init__()

    def _apply_filter(self, img, kernel, normalize=True):
        if normalize:
            kernel_sum = kernel.sum()
            if kernel_sum != 0.0:
                kernel = kernel / kernel_sum
        return cv2.filter2D(img, -1, kernel)

    def blur(self, img):
        kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9
        return self._apply_filter(img, kernel)

    def gaussian_blur(self, img):
        kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
        return self._apply_filter(img, kernel)

    def sharpen(self, img):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return self._apply_filter(img, kernel, normalize=False)

    def edge_detect(self, img):
        kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        return self._apply_filter(img, kernel, normalize=False)


class Image:
    def __init__(self, path, grayscale=False):
        self.grayscale = bool(grayscale)
        self.im = self.load(path, grayscale=grayscale)

    @staticmethod
    def load(arg, grayscale=False):
        flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        return cv2.imread(arg, flag)

    def print(self):
        print(f"Image shape: {self.im.shape}")
        print(self.im)

    def save(self, path):
        cv2.imwrite(path, self.im)


def plot_images(images, titles):
    fig, axes = plt.subplots(1, len(images), figsize=(len(images) * 6, 4))
    for i, (image, title) in enumerate(zip(images, titles)):
        if image.ndim == 2:
            axes[i].imshow(image, cmap="gray", vmin=0, vmax=255)
        else:
            axes[i].imshow(image, vmin=0, vmax=255)
        axes[i].set_title(title)
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig("ex02_out.png")


def main():
    parser = argparse.ArgumentParser(description="Apply simple filters to an image.")
    parser.add_argument("img_path", help="Path to the input image")
    parser.add_argument(
        "-g", "--grayscale", action="store_true", help="Load image as grayscale"
    )
    args = parser.parse_args()

    img = Image(args.img_path, grayscale=args.grayscale)
    filters = Filter()
    img.print()
    img_arr = img.im
    blurred = filters.blur(img_arr)
    sharpened = filters.sharpen(img_arr)
    edges = filters.edge_detect(img_arr)
    gaussian = filters.gaussian_blur(img_arr)

    imgs = [img_arr, blurred, sharpened, edges, gaussian]
    if img_arr.ndim == 3:
        imgs = [im[:, :, ::-1] for im in imgs]
    plot_images(imgs, ["original", "blurred", "sharpened", "edges", "gaussian blur"])


if __name__ == "__main__":
    main()
