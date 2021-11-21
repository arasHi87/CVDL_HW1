import cv2

from .display import DisplayWindow


class Transforms:
    def __init__(self, parent=None):
        self.img = cv2.imread("data/SQUARE-01.png")
        self.display = DisplayWindow(parent)

    def resize(self):
        print(self.img)
        result = cv2.resize(self.img, (128, 128), interpolation=cv2.INTER_AREA)
        self.display.add_img_to_window(result)
        self.display.show()
