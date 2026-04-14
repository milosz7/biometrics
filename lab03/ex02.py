import sys

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class TemplateLocData:
    x_start: int
    y_start: int
    x_end: int
    y_end: int


def get_template_coords(image, coords):
    return image[
        coords.y_start : coords.y_end,
        coords.x_start : coords.x_end,
    ]


def get_templates(image):
    star_loc = TemplateLocData(x_start=62, y_start=474, x_end=107, y_end=518)
    star_template = get_template_coords(image, star_loc)
    rhombus_loc = TemplateLocData(x_start=254, y_start=487, x_end=280, y_end=516)
    rhombus_template = get_template_coords(image, rhombus_loc)
    heart_loc = TemplateLocData(x_start=67, y_start=531, x_end=104, y_end=565)
    heart_template = get_template_coords(image, heart_loc)
    smile_loc = TemplateLocData(x_start=249, y_start=540, x_end=285, y_end=575)
    smile_template = get_template_coords(image, smile_loc)
    templates = [heart_template, star_template, rhombus_template, smile_template]
    return templates


def main(img_path):
    img = cv.imread(img_path, cv.COLOR_BGR2RGB)
    kernel = np.ones((3, 3), np.uint8)
    img_detect = cv.dilate(img, kernel, iterations=2)

    templates = get_templates(img)
    object_names = ["heart", "star", "rhombus", "smile"]
    bbox_colors = [(255, 255, 0), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
    image_height = 450
    match_threshold = 0.7

    img_detect = img_detect[:image_height, :].copy()
    template_counts = []
    for template, color in zip(templates, bbox_colors):
        _, w, h = template.shape[::-1]
        matches = cv.matchTemplate(img_detect, template, cv.TM_CCOEFF_NORMED)
        relevant_matches = np.where(matches >= match_threshold)
        relevant_rects = [(x, y, w, h) for x, y in zip(*relevant_matches[::-1])]
        relevant_matches, _ = cv.groupRectangles(relevant_rects, 1, 0.5)
        for pt in relevant_matches:
            [x, y, w, h] = pt
            start = (x, y)
            end = (x + w, y + h)
            cv.rectangle(img, start, end, color, 2)
        template_counts.append(len(relevant_matches))

    title = "Counts: " + ", ".join(
        (f"{name}: {count}" for name, count in zip(object_names, template_counts))
    )
    plt.title(title)
    plt.axis("off")
    plt.imshow(img[:, :, ::-1])
    plt.savefig("ex02_out.png")


if __name__ == "__main__":
    img_path = sys.argv[1]
    main(img_path)
