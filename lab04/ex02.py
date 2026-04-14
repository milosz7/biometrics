import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import cv2 as cv


def main(img_path):
    img = cv.imread(img_path)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_gray = cv.medianBlur(img_gray, 5)
    circles = cv.HoughCircles(
        img_gray,
        cv.HOUGH_GRADIENT,
        dp=1,
        minDist=40,
        param1=20,
        param2=10,
        minRadius=15,
        maxRadius=25
    )
    circles = np.around(circles).astype(np.uint8)

    for [x, y, r] in circles[0]:
        cv.circle(img, (x, y), r, (0, 0, 255), 2)
        cv.line(img, (x - r, y), (x + r, y), (255, 0, 0), 2)
        plt.text(x + r // 2, y - r // 2, f"{2 * r}px", color="red", size=14, label="radius")
    plt.title(f"Found {len(circles[0])} circles.")
    legend_handles = [Line2D([0], [0], color="red", lw=2, label="radius (px)")]
    plt.legend(handles=legend_handles, loc="upper right")
    plt.imshow(img)
    plt.axis("off")
    plt.savefig("ex02_out.png")


if __name__ == "__main__":
    main(sys.argv[1])
    cv.waitKey(0)
    cv.destroyAllWindows()