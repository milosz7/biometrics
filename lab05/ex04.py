import sys

import cv2 as cv
from scipy.ndimage import center_of_mass


def load_image(path):
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    return img


def apply_clahe(img):
    clahe = cv.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def apply_gaussian_blur(img):
    return cv.GaussianBlur(img, (31, 31), -3)


def find_brightest_pixel(img):
    idx_flat = img.argmax()
    row = idx_flat // img.shape[1]
    col = idx_flat % img.shape[1]
    return row, col


def find_brightest_area(img, frac=0.2):
    row, col = find_brightest_pixel(img)
    H, W = img.shape
    h = max(1, int(round(H * frac)))
    w = max(1, int(round(W * frac)))
    area = img[row - h // 2: row + h // 2, col - w // 2: col + w // 2]
    cx, cy = center_of_mass(area)
    return int(row - h // 2 + cx), int(col - w // 2 + cy), h, w


if __name__ == '__main__':
    img_path = sys.argv[1]
    img_raw = load_image(img_path)
    img = apply_clahe(img_raw)
    img = apply_gaussian_blur(img)
    row, col, h, w = find_brightest_area(img)
    br_x, br_y = find_brightest_pixel(img)

    optic_nerve_rect = img_raw[row-h//2:row + h//2, col-w//2:col + w//2]
    optic_nerve_clahe = apply_clahe(optic_nerve_rect)
    n_row, n_col, n_h, n_w = find_brightest_area(optic_nerve_clahe)
    H, W = img_raw.shape
    x0 = int(col - w // 2)
    y0 = int(row - h // 2)
    global_x = int(x0 + n_col)
    global_y = int(y0 + n_row)
    global_x = max(0, min(W - 1, global_x))
    global_y = max(0, min(H - 1, global_y))
    cv.circle(img_raw, (global_x, global_y), 25, (0, 255, 0), 1)
    cv.imwrite("ex04_out.png", img_raw)
