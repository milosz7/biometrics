import sys

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path):
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("file could not be read, check with os.path.exists()")
    return img

def equalize_image(image):
    img_min = float(image.min())
    img_max = float(image.max())
    if img_max == img_min:
        return image.copy().astype(np.uint8)
    stretched = (image.astype(np.float32) - img_min) / (img_max - img_min) * 255.0
    return np.clip(stretched, 0, 255).astype(np.uint8)

def stretch_histogram(hist):
    idx_min, idx_max = 0, 255
    hist = hist[0]
    lookup_table = np.arange(256)
    while hist[idx_min] <= 0:
        idx_min = idx_min + 1
    while hist[idx_max] <= 0:
        idx_max = idx_max - 1

    for idx in range(idx_min, idx_max):
        lookup_table[idx] = idx_max / (idx_max - idx_min) * (idx - idx_min)
    return lookup_table

def plot_images_and_histograms(image_path):
    img = load_image(image_path)
    cv_eq = cv.equalizeHist(img)
    my_eq = equalize_image(cv_eq)
    img_hist = np.histogram(img.flatten(), bins=256, range=(0, 256))

    stretch_lookup_table = stretch_histogram(img_hist)
    img_stretched = stretch_lookup_table[img]

    images = [img, cv_eq, my_eq, img_stretched]
    titles = ['Original', 'cv.equalizeHist', 'My equalize', 'My stretched']

    hists = [np.histogram(im, bins=256, range=(0, 256))[0] for im in images]
    bins = np.arange(256)

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    for i, (ax_img, ax_hist) in enumerate(zip(axes[0], axes[1])):
        ax_img.imshow(images[i], cmap='gray', vmin=0, vmax=255)
        ax_img.set_title(titles[i])
        ax_img.axis('off')

        ax_hist.bar(bins, hists[i], width=1.0, color='black')

        ax_hist.set_xlim(0, 255)
        ax_hist.set_xlabel('Intensity')
        ax_hist.set_ylabel('Count')

    plt.tight_layout()
    plt.savefig('ex01.png')
    plt.show()

if __name__ == '__main__':
    input_path = sys.argv[1]
    plot_images_and_histograms(sys.argv[1])
