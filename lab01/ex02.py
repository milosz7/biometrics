import sys

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path):
    img = cv.imread(image_path)
    if img is None:
        raise FileNotFoundError("file could not be read, check with os.path.exists()")
    return img


def mask_image(image, threshold):
    return image > threshold


def gamma_correction_table(gamma):
    table = np.array([((i / 255.0) ** gamma) for i in np.arange(0, 256)])
    table *= 255
    table = table.astype("uint8")
    return table


def apply_gamma_correction_below_threshold(image, gamma, threshold):
    correction_table = gamma_correction_table(gamma)
    masked_img = np.ma.masked_where(image <= threshold, image)
    gamma_corrected = cv.LUT(masked_img, correction_table)
    return gamma_corrected


def plot_and_save_image(image):
    image = image[:, :, ::-1]
    plt.axis("off")
    plt.imshow(image, vmin=0, vmax=255) # BGR -> RGB
    plt.tight_layout()
    plt.savefig("ex02_out.png")
    plt.show()


if __name__ == '__main__':
    path = sys.argv[1]
    pixel_threshold = 100
    gamma = 0.5
    img = load_image(path)
    gamma_corrected = apply_gamma_correction_below_threshold(img, gamma, pixel_threshold)
    plot_and_save_image(gamma_corrected)
