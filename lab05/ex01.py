import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def gray_average(im):
    img_out = (im[:, :, 0] / 3 + im[:, :, 1] / 3 + im[:, :, 2] / 3)
    img_out = np.clip(img_out, 0, 255)
    img_out = img_out.astype(np.uint8)
    return img_out


def gray_human_like(im):
    img_out = (0.299 * im[:, :, 0] + 0.587 * im[:, :, 1] + 0.114 * im[:, :, 2])
    img_out = np.clip(img_out, 0, 255)
    img_out = img_out.astype(np.uint8)
    return img_out


def plot_images(images, titles):
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
    plt.savefig('ex01_out.png')


if __name__ == '__main__':
    img_path = sys.argv[1]
    img_raw = cv.imread(img_path, cv.IMREAD_COLOR_RGB)
    img_gray_avg = gray_average(img_raw)
    img_gray_human_like = gray_human_like(img_raw)

    img_gray_avg = np.stack([img_gray_avg, img_gray_avg, img_gray_avg], axis=-1)
    img_gray_human_like = np.stack([img_gray_human_like, img_gray_human_like, img_gray_human_like], axis=-1)
    red_image = img_raw.copy()
    red_image[:, :, 1] = 0
    red_image[:, :, 2] = 0

    images = [img_raw, img_gray_avg, img_gray_human_like, red_image]
    titles = ['raw', 'gray avg', 'gray human like', 'red channel']
    plot_images(images, titles)