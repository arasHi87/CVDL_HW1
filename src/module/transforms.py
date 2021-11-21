import cv2
import numpy as np

from .display import DisplayWindow


class Transforms:
    def __init__(self, parent=None):
        self.img = cv2.imread("data/SQUARE-01.png")
        self.img = cv2.resize(self.img, (256, 256), interpolation=cv2.INTER_AREA)
        self.display = DisplayWindow(parent)

    def resize(self):
        self.display.add_img_to_window(self.img)
        self.display.show()

    def translation(self):
        pass

    def rotation(self):
        o_h, o_w, _ = self.img.shape
        temp = cv2.getRotationMatrix2D((o_w / 2, o_h / 2), 10, 0.5)
        result = cv2.warpAffine(self.img, temp, (o_w, o_h))
        frame = np.zeros(shape=(300, 400, 3), dtype=np.uint8)
        frame[0 : 0 + result.shape[0], 0 : 0 + result.shape[1]] = result
        self.display.add_img_to_window(frame)
        self.display.show()

    def shearing(self):
        o_h, o_w, _ = self.img.shape
        pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
        pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
        temp = cv2.getRotationMatrix2D((o_w / 2, o_h / 2), 0, 0.5)
        result = cv2.warpAffine(self.img, temp, (o_w, o_h))
        frame = np.zeros(shape=(300, 400, 3), dtype=np.uint8)
        frame[0 : 0 + result.shape[0], 0 : 0 + result.shape[1]] = result
        temp = cv2.getAffineTransform(pts1, pts2)
        result = cv2.warpAffine(frame, temp, (frame.shape[1], frame.shape[0]))
        self.display.add_img_to_window(result)
        self.display.show()
