import cv2
import numpy as np
from scipy import signal

from .display import DisplayWindow


class Detection:
    def __init__(self, parent=None):
        self.img = cv2.imread("data/House.jpg", cv2.IMREAD_GRAYSCALE)
        self.display = DisplayWindow(parent)

    def _gaussian_blur(self):
        kernel = np.array(
            [
                [0.045, 0.122, 0.045],
                [0.122, 0.332, 0.122],
                [0.045, 0.122, 0.045],
            ]
        )
        return signal.convolve2d(self.img, kernel, mode="same", boundary="symm").astype(
            np.uint8
        )

    def gaussian_blur(self):
        result = self._gaussian_blur()
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        self.display.add_img_to_window(result)
        self.display.show()

    def sobel_x(self):
        kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        result = signal.convolve2d(
            self._gaussian_blur(), kernel, mode="same", boundary="symm"
        )
        result = abs(result)
        result = (
            (result - np.amin(result)) * 255 / (np.amax(result) - np.amin(result))
        ).astype(np.uint8)
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        self.display.add_img_to_window(result)
        self.display.show()

    def sobel_y(self):
        kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        result = signal.convolve2d(
            self._gaussian_blur(), kernel, mode="same", boundary="symm"
        )
        result = abs(result)
        result = (
            (result - np.amin(result)) * 255 / (np.amax(result) - np.amin(result))
        ).astype(np.uint8)
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        self.display.add_img_to_window(result)
        self.display.show()

    def magnitude(self):
        kernel = [
            np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
            np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]),
        ]
        img = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        o_h, o_w = img.shape  # origin heigh and width
        k_h, k_w = kernel[0].shape  # kernel heigh and width
        result = np.zeros(shape=(o_h, o_w), dtype=np.uint8)

        for i in range(1, o_h - 2):
            for j in range(1, o_w - 2):
                result[i, j] = abs(
                    np.sum(
                        np.multiply(img[i : i + k_h, j : j + k_w], kernel[0]),
                    )
                ) + abs(
                    np.sum(
                        np.multiply(img[i : i + k_h, j : j + k_w], kernel[1]),
                    )
                )

        self.display.add_img_to_window(cv2.cvtColor(result, cv2.COLOR_GRAY2BGR))
        self.display.show()
