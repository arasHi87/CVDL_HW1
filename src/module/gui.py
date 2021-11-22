from functools import partial

from PyQt5.QtWidgets import (QGroupBox, QHBoxLayout, QPushButton, QVBoxLayout,
                             QWidget)

from .detection import Detection
from .process import Process
from .smoothing import Smoothing
from .transforms import Transforms


class Window(QWidget):
    def __init__(self):
        super().__init__()

        # layout setting
        self.padding = 20
        self.row_amount = 4
        self.col_amount = 4
        self._height = 600
        self._width = 900

        # window setting
        self.layout = QHBoxLayout()
        self.setLayout(self.layout)
        self.setWindowTitle("HW1")
        self.setGeometry(0, 0, self._width, self._height)
        self.init_btn()
        self.show()

    def init_btn(self):
        msg = [
            ["Load Image", "Color seoeration", "Color Transformations", "Blending"],
            ["Gaussian Blur", "Bilateral Filter", "Median Filter"],
            ["Gaussian Blur", "Sobel X", "Sobel Y", "Magnitude"],
            ["Resize", "Translation", "Rotation, Scaling", "Shearing"],
        ]
        title = [
            "Image Processing",
            "Image Smoothing",
            "Edge Detection",
            "Transformation",
        ]

        for i in range(len(msg)):
            v_layout = QVBoxLayout()
            group_box = QGroupBox(title[i])
            for j in range(len(msg[i])):
                btn = QPushButton(msg[i][j], self)
                btn.clicked.connect(partial(self._executor, i, j))
                v_layout.addWidget(btn)
            group_box.setLayout(v_layout)
            self.layout.addWidget(group_box)

    def _executor(self, i, j):
        _process = Process(self)
        _smoothing = Smoothing(self)
        _detection = Detection(self)
        _transforms = Transforms(self)
        func = [
            [
                getattr(_process, "load_img"),
                getattr(_process, "color_seperation"),
                getattr(_process, "color_transformations"),
                getattr(_process, "blending"),
            ],
            [
                getattr(_smoothing, "gaussian_blur"),
                getattr(_smoothing, "bilateral_filter"),
                getattr(_smoothing, "median_filter"),
            ],
            [
                getattr(_detection, "gaussian_blur"),
                getattr(_detection, "sobel_x"),
                getattr(_detection, "sobel_y"),
                getattr(_detection, "magnitude"),
            ],
            [
                getattr(_transforms, "resize"),
                getattr(_transforms, "translation"),
                getattr(_transforms, "rotation"),
                getattr(_transforms, "shearing"),
            ],
        ]

        if i < len(func) and j < len(func[i]):
            func[i][j]()
