import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import medial_axis


def load_image(path):
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    return img


def threshold_otsu_binary(img):
    _, otsu_mask = cv.threshold(
        img, 50, 255, cv.THRESH_BINARY | cv.THRESH_OTSU
    )
    adaptive_mask = cv.adaptiveThreshold(
        img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 33, -15
    )
    mask = cv.bitwise_or(otsu_mask, adaptive_mask)
    return mask


def apply_clahe(img):
    clahe = cv.createCLAHE(clipLimit=8, tileGridSize=(4, 4))
    return clahe.apply(img)


def apply_bilateral(img):
    return cv.bilateralFilter(img, -1, 4,  4)


def filter_highpass(img):
    kernel = np.zeros((7, 7))
    for i in range(kernel.shape[0]):
        kernel[i, i] = -2
        kernel[i, kernel.shape[0] - 1 - i] = -2
    kernel[3, 3] = 24
    img_f = img.astype(np.float32)
    resp = cv.filter2D(img_f, cv.CV_32F, kernel, borderType=cv.BORDER_REFLECT)
    out = cv.normalize(resp, None, 0, 255, cv.NORM_MINMAX)
    return out.astype(np.uint8)


def filter_kernel_mf_fdog(L, sigma, t=3):
    dim_y = int(L)
    dim_x = 2 * int(t * sigma)

    ctr_x = dim_x / 2

    x = np.arange(dim_x) - ctr_x
    arr = np.tile(x, (dim_y, 1))

    two_sigma_sq = 2 * sigma * sigma
    def k_fun(x):
        return -1 * np.exp(-x * x / two_sigma_sq)

    kernel = k_fun(arr)
    kernel = kernel - kernel.mean()
    return cv.flip(kernel, -1)


def create_matched_filters(kernel, n=12):
    h, w = kernel.shape
    diag = int(np.ceil(np.sqrt(h*h + w*w)))

    pad_h = (diag - h) // 2
    pad_w = (diag - w) // 2

    padded = np.pad(kernel, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')

    ph, pw = padded.shape
    center = ((pw - 1) / 2, (ph - 1) / 2)

    kernels = []
    rotate_step = 180 / n

    for i in range(n):
        angle = i * rotate_step
        r_mat = cv.getRotationMatrix2D(center, angle, 1)
        k = cv.warpAffine(
            padded,
            r_mat,
            (pw, ph),
            flags=cv.INTER_LINEAR,
            borderMode=cv.BORDER_CONSTANT,
            borderValue=0
        )
        k /= (np.sum(np.abs(k)) + 1e-8)
        kernels.append(k.astype(np.float32))

    return kernels


def apply_matched_filters(im, kernels):
    images = np.array([cv.filter2D(im, cv.CV_32F, k) for k in kernels])
    return np.max(images, axis=0)


def remove_artifacts(img):
    img_cp = img.copy()
    groups = []
    visited = np.zeros(img.shape, dtype=bool)
    H, W = img.shape

    for r in range(H):
        for c in range(W):
            if not visited[r, c] and img_cp[r, c] != 0:
                group_current = []
                queue = [(r, c)]
                visited[r, c] = True
                while queue:
                    cr, cc = queue.pop(0)
                    group_current.append((cr, cc))
                    for nr, nc in (
                            (cr, cc - 1),
                            (cr, cc + 1),
                            (cr - 1, cc),
                            (cr + 1, cc),
                    ):
                        if 0 <= nr < H and 0 <= nc < W:
                            if (not visited[nr, nc]) and (img_cp[nr, nc] != 0):
                                visited[nr, nc] = True
                                queue.append((nr, nc))
                groups.append(group_current)

    lengths = [len(group) for group in groups]
    median = np.median(lengths)
    mean = np.mean(lengths)
    thresh = (mean + median) / 2

    for group in groups:
        if len(group) < thresh:
            for gr, gc in group:
                img_cp[gr, gc] = 0

    return img_cp


def plot_images(images, titles):
    fig, axs = plt.subplots(1, len(images), figsize=(len(images) * 3, 4))
    for n, (img, ti) in enumerate(zip(images, titles)):
        axs[n].imshow(img, cmap='gray')
        axs[n].set_title(ti)
        axs[n].axis('off')
    plt.tight_layout()
    plt.savefig('ex03_out.png')


if __name__ == "__main__":
    img = load_image("./retina.png")
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9))
    img_mask = threshold_otsu_binary(cv.erode(img, kernel, iterations=2))
    img_clahe = apply_clahe(img)
    img_bilateral = apply_bilateral(img_clahe)
    img_highpass = filter_highpass(img_bilateral)

    kernel_narrow = filter_kernel_mf_fdog(15, 1.5)
    rotated_narrow = create_matched_filters(kernel_narrow)
    result_narrow = apply_matched_filters(img_highpass.copy(), rotated_narrow)

    kernel_wide = filter_kernel_mf_fdog(15, 2)
    rotated_wide = create_matched_filters(kernel_wide)
    result_wide = apply_matched_filters(img_highpass.copy(), rotated_wide)

    result_narrow = cv.normalize(result_narrow, None, 0, 255, cv.NORM_MINMAX)
    result_wide = cv.normalize(result_wide, None, 0, 255, cv.NORM_MINMAX)
    result_narrow = result_narrow.astype(np.uint8)
    result_wide = result_wide.astype(np.uint8)

    otsu_narrow = threshold_otsu_binary(result_narrow)
    otsu_wide = threshold_otsu_binary(result_wide)
    otsu_wide = remove_artifacts(otsu_wide)
    otsu_narrow = remove_artifacts(otsu_narrow)

    otsu = cv.bitwise_or(otsu_narrow, otsu_wide)
    masked_otsu = cv.bitwise_and(img_mask, otsu)

    skel, distance = medial_axis(masked_otsu, return_distance=True)
    skeleton = skel * distance

    images = [img, masked_otsu, skeleton]
    titles = ["Raw image", "Retina mask", "Retina skeleton"]
    plot_images(images, titles)
