import sys

import cv2 as cv
from matplotlib import pyplot as plt
from skimage.morphology import medial_axis


def erosion_size(radius):
    return 2 * radius + 1, 2 * radius + 1


def erode_and_get_boundary(img, radius):
    size = erosion_size(radius)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, size)
    eroded = cv.erode(img, kernel, iterations=1)
    boundary = cv.subtract(img, eroded)
    return boundary


def plot_images(images, titles):
    fig, axs = plt.subplots(1, len(images), figsize=(len(images) * 3, 4))
    for n, (img, ti) in enumerate(zip(images, titles)):
        axs[n].imshow(img, cmap='gray')
        axs[n].set_title(ti)
        axs[n].axis('off')
    plt.tight_layout()
    plt.savefig('ex01_out.png')


def main(img_path):
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    boundary_r1 = erode_and_get_boundary(img, 1)
    boundary_r4 = erode_and_get_boundary(img, 4)
    skel, distance = medial_axis(img, return_distance=True)
    skeleton = skel * distance
    images = [img, boundary_r1, boundary_r4, skeleton]
    titles = ['Original Image', 'Boundary R1', 'Boundary R4', 'Skeleton']
    plot_images(images, titles)


if __name__ == '__main__':
    main(sys.argv[1])
