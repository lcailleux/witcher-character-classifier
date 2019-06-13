import os
import sys
import argparse
from helper.utils import ndarray_to_qimage
from data_handler import DataHandler
from neural_network import NeuralNetwork
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QPushButton, QLabel, QVBoxLayout, QFrame
from PyQt5.QtGui import QPixmap
from helper import constant

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class WitcherClassifier(QWidget):
    def __init__(self, args, parent=None):
        super(WitcherClassifier, self).__init__(parent)

        self.args = args
        self.data_handler = DataHandler()
        self.neural_network = NeuralNetwork()
        layout = QVBoxLayout()

        self.btn = QPushButton("Choose the character image")
        self.btn.clicked.connect(self.predict_example)

        self.le = QLabel()
        self.le.setFixedWidth(constant.IMAGE_SHOW_WIDTH)
        self.le.setFixedHeight(constant.IMAGE_SHOW_HEIGHT)
        #self.le.setStyleSheet("background-color: darkgrey")
        self.le.setFrameShape(QFrame.Panel)
        self.le.setFrameShadow(QFrame.Sunken)
        self.le.setLineWidth(1)

        layout.addWidget(self.le)
        layout.addWidget(self.btn)

        self.setLayout(layout)
        self.setWindowTitle("The witcher character classifier")

    def predict_example(self):
        image_file, _ = QFileDialog.getOpenFileName(
            self,
            "Open file",
            "./dataset",
            "Image files (*.jpg *.jpeg *.png)"
        )

        if image_file:
            output_image = self.neural_network.run(self.args, image_file)
            self.le.setPixmap(QPixmap.fromImage(ndarray_to_qimage(output_image)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", type=str, default=constant.MODEL_PATH, help="path to trained model model")
    ap.add_argument("-l", "--labelbin", default=constant.LB_PATH, help="path to output label binarizer")

    args = vars(ap.parse_args())

    app = QApplication(sys.argv)
    window = WitcherClassifier(args)
    window.show()
    sys.exit(app.exec_())

