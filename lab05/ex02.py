import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


def compute_lut(tile, clip_limit):
    hist, _ = np.histogram(tile.flatten(), 256, (0,256))

    excess = np.maximum(hist - clip_limit, 0)
    hist = np.minimum(hist, clip_limit)
    hist += excess.sum() // 256

    cdf = hist.cumsum()
    cdf = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    return cdf.astype('uint8')


def clahe(img, n_tiles=8, clip_limit=40):
    h, w = img.shape

    tile_h = h // n_tiles
    tile_w = w // n_tiles

    luts = [[None for _ in range(n_tiles)] for _ in range(n_tiles)]

    for i in range(n_tiles):
        for j in range(n_tiles):
            tile = img[i*tile_h:(i+1)*tile_h, j*tile_w:(j+1)*tile_w]
            luts[i][j] = compute_lut(tile, clip_limit)

    output = np.empty_like(img, dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            i = y // tile_h
            j = x // tile_w

            i1 = min(i, n_tiles - 1)
            j1 = min(j, n_tiles - 1)
            i2 = min(i1 + 1, n_tiles - 1)
            j2 = min(j1 + 1, n_tiles - 1)

            dy = (y - i1 * tile_h) / tile_h
            dx = (x - j1 * tile_w) / tile_w

            val = img[y, x]

            TL = luts[i1][j1][val]
            TR = luts[i1][j2][val]
            BL = luts[i2][j1][val]
            BR = luts[i2][j2][val]

            top = TL * (1 - dx) + TR * dx
            bottom = BL * (1 - dx) + BR * dx
            interpolated = top * (1 - dy) + bottom * dy

            output[y, x] = int(interpolated)

    return output


def plot_images(images, titles):
    fig, axs = plt.subplots(1, len(images), figsize=(len(images) * 3, 4))
    for n, (img, ti) in enumerate(zip(images, titles)):
        axs[n].imshow(img, cmap='gray')
        axs[n].set_title(ti)
        axs[n].axis('off')
    plt.tight_layout()
    plt.savefig('ex02_out.png')


if __name__ == '__main__':
    img_path = sys.argv[1]
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    my_clahe_img = clahe(img.copy(), n_tiles=16, clip_limit=15)
    clahe_cv = cv2.createCLAHE(clipLimit=10, tileGridSize=(4, 4))
    img_equ = cv2.equalizeHist(img.copy())
    cv_clahe_img = clahe_cv.apply(img.copy())

    images = [img, my_clahe_img, img_equ, cv_clahe_img]
    titles = ['img', 'My CLAHE', 'Global equalization', 'CV CLAHE']
    plot_images(images, titles)
