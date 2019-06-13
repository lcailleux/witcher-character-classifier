import numpy as np
from PyQt5 import QtGui


def ndarray_to_qimage(image):
    assert (np.max(image) <= 255)
    image8 = image.astype(np.uint8, order='C', casting='unsafe')
    height, width, colors = image8.shape
    bytes_per_line = 3 * width

    image = QtGui.QImage(image8.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
    image = image.rgbSwapped()

    return image
