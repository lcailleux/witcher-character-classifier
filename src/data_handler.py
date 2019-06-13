import os
import cv2
import random
import pickle
import numpy as np
from helper import constant
from imutils import paths
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator, img_to_array


class DataHandler:
    def load_dataset(self, args):
        """
        :param args:
        :return:
        """

        print("[INFO] loading images...")

        data, labels = self.load_data_and_labels(args)

        print("[INFO] data matrix: {:.2f}MB".format(data.nbytes / (1024 * 1000.0)))

        lb = LabelBinarizer()
        labels = lb.fit_transform(labels)

        print("[INFO] serializing label binarizer...")

        self.dump_labels(args, lb)

        print("[INFO] splitting train/test sets...")

        (x_train, x_test, y_train, y_test) = train_test_split(data, labels, test_size=0.2, random_state=42)

        self.show_examples(args, x_train)

        return x_train, x_test, y_train, y_test

    def load_data_and_labels(self, args):
        data = []
        labels = []

        image_paths = sorted(list(paths.list_images(args["dataset"])))
        random.seed(42)
        random.shuffle(image_paths)

        for imagePath in image_paths:
            # load the image, pre-process it, and store it in the data list
            image = cv2.imread(imagePath)
            image = cv2.resize(image, (constant.IMAGE_HEIGHT, constant.IMAGE_WIDTH))
            image = img_to_array(image)
            data.append(image)

            # extract the class label from the image path and update the
            # labels list
            label = imagePath.split(os.path.sep)[-2]
            labels.append(label)

        data = np.array(data, dtype="float") / 255.0
        labels = np.array(labels)

        return data, labels

    def load_labels(self, args):
        lb = pickle.loads(open(args["labelbin"], "rb").read())
        return lb

    def dump_labels(self, args, lb):
        f = open(args["labelbin"], "wb")
        f.write(pickle.dumps(lb))
        f.close()

    def load_example(self, image_path):
        example = cv2.imread(image_path)
        image = example.copy()

        # pre-process the image for classification
        example = cv2.resize(example, (constant.IMAGE_HEIGHT, constant.IMAGE_WIDTH))
        example = example.astype("float") / 255.0
        example = img_to_array(example)
        example = np.expand_dims(example, axis=0)

        return example, image

    def data_augmentation(self):
        return ImageDataGenerator(
            rotation_range=25,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode="nearest"
        )

    def show_examples(self, args, x_train):
        """

        :param args:
        :param x_train:
        :return:
        """
        if args["show_examples"]:
            print("[INFO] showing a batch of training examples...")

            for i in range(9):
                plt.subplot(330 + 1 + i)
                plt.imshow(x_train[i])

            plt.show()

    def display_dataset_shape(self, args):
        """

        :return:
        """
        x_train, x_test, y_train, y_test = self.load_dataset(args)

        print("[INFO] number of training examples = " + str(x_train.shape[0]))
        print("[INFO] number of test examples = " + str(x_test.shape[0]))
        print("[INFO] X_train shape: " + str(x_train.shape))
        print("[INFO] Y_train shape: " + str(y_train.shape))
        print("[INFO] X_test shape: " + str(x_test.shape))
        print("[INFO] Y_test shape: " + str(y_test.shape))
