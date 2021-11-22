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
        gaussian = self._gaussian_blur()
        kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        grad_x = signal.convolve2d(gaussian, kernel, mode="same", boundary="symm")
        kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        grad_y = signal.convolve2d(gaussian, kernel, mode="same", boundary="symm")
        result = (grad_x ** 2 + grad_y ** 2) ** 0.5
        result = (
            (result - np.amin(result)) * 255 / (np.amax(result) - np.amin(result))
        ).astype(np.uint8)
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        self.display.add_img_to_window(result)
        self.display.show()
