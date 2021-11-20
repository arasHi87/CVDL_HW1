import cv2
import numpy as np

from .display import DisplayWindow


class Detection:
    def __init__(self, parent=None):
        self.img = cv2.imread("data/House.jpg")
        self.display = DisplayWindow(parent)

    def gaussian_blur(self):
        kernel = np.array(
            [
                [0.045, 0.122, 0.045],
                [0.122, 0.332, 0.122],
                [0.045, 0.122, 0.045],
            ]
        )
        _img = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        k_h, k_w = kernel.shape  # kernel heigh and width
        o_h, o_w = _img.shape  # origin heigh and width
        p_h, p_w = (k_h - 1) // 2, (k_w - 1) // 2  # pad heigh and width
        result = np.zeros(shape=(o_h, o_w), dtype=np.uint8)
        _img = np.pad(
            _img,
            pad_width=[(p_h, p_h), (p_w, p_w)],
            mode="constant",
            constant_values=0,
        )

        for i in range(o_h):
            for j in range(o_w):
                result[i, j] = np.sum(
                    np.multiply(_img[i : i + k_h, j : j + k_w], kernel),
                    dtype=np.uint8,
                )

        self.display.add_img_to_window(cv2.cvtColor(result, cv2.COLOR_GRAY2BGR))
        self.display.show()

    def sobel_x(self):
        kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        img = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        o_h, o_w = img.shape  # origin heigh and width
        k_h, k_w = kernel.shape  # kernel heigh and width
        p_h, p_w = (k_h - 1) // 2, (k_w - 1) // 2  # pad heigh and width
        result = np.zeros(shape=(o_h, o_w), dtype=np.uint8)
        img = np.pad(
            img,
            pad_width=[(p_h, p_h), (p_w, p_w)],
            mode="constant",
            constant_values=0,
        )

        for i in range(o_h):
            for j in range(o_w):
                result[i, j] = int(
                    abs(
                        np.sum(
                            np.multiply(img[i : i + k_h, j : j + k_w], kernel),
                        )
                    )
                )

        self.display.add_img_to_window(cv2.cvtColor(result, cv2.COLOR_GRAY2BGR))
        self.display.show()
