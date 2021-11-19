import cv2

from .display import DisplayWindow


class Smoothing:
    def __init__(self, parent=None):
        self.img = [
            cv2.imread("data/Lenna_whiteNoise.jpg"),
            cv2.imread("data/Lenna_pepperSalt.jpg"),
        ]
        self.display = DisplayWindow(parent)

    def gaussian_blur(self):
        result = cv2.GaussianBlur(self.img[0], (5, 5), 0)
        self.display.add_img_to_window(result)
        self.display.show()

    def bilateral_filter(self):
        result = cv2.bilateralFilter(self.img[0], 9, 90, 90)
        self.display.add_img_to_window(result)
        self.display.show()
