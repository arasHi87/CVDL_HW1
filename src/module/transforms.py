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

    def _translation(self, img):
        frame = np.zeros(shape=(300, 400, 3), dtype=np.uint8)
        frame[0 : 0 + img.shape[0], 0 : 0 + img.shape[1]] = img
        o_h, o_w, _ = frame.shape
        rotate_mat = cv2.getRotationMatrix2D((o_w / 2, o_h / 2), 0, 1)
        shift_mat = np.float32([[0, 0, 0], [0, 0, 60]])
        result = cv2.warpAffine(frame, np.add(rotate_mat, shift_mat), (o_w, o_h))
        return result

    def _rotation(self, img):
        o_h, o_w, _ = img.shape
        temp = cv2.getRotationMatrix2D((o_w / 2, o_h / 2), 10, 0.5)
        result = cv2.warpAffine(img, temp, (o_w, o_h))
        return result

    def _shearing(self, img):
        o_h, o_w, _ = img.shape
        pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
        pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
        temp = cv2.getAffineTransform(pts1, pts2)
        result = cv2.warpAffine(img, temp, (o_w, o_h))
        return result

    def translation(self):
        result = self._translation(self.img)
        self.display.add_img_to_window(result)
        self.display.show()

    def rotation(self):
        result = self._translation(self.img)
        result = self._rotation(result)
        self.display.add_img_to_window(result)
        self.display.show()

    def shearing(self):
        result = self._translation(self.img)
        result = self._rotation(result)
        result = self._shearing(result)
        self.display.add_img_to_window(result)
        self.display.show()
