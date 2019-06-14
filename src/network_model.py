import numpy as np
import keras.applications
from keras.layers import Dropout, Dense, BatchNormalization, Flatten
from keras.models import Model
from keras.optimizers import Adam
from helper import constant
from matplotlib import pyplot as plt


class NetworkModel:
    def pretrained_model(self):
        """
        :return: keras.applications.vgg16.VGG16
        """

        input_tensor = keras.Input(shape=(constant.IMAGE_HEIGHT, constant.IMAGE_WIDTH, constant.CHANNEL_NUMBER))
        return keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

    def create(self, input_model, x):
        """
        :param input_model: keras.applications.vgg16.VGG16
        :param x:
        :return model: keras.models.Model
        """

        for layer in input_model.layers:
            layer.trainable = False

        model = Model(inputs=input_model.input, outputs=x)
        model.compile(
            loss=constant.MODEL_LOSS,
            optimizer=Adam(lr=constant.LEARNING_RATE),
            metrics=['accuracy']
        )

        return model

    def plot_example(self, image, label=None):
        """
        :param image: image data (numpy array)
        :param label: image label (str)
        :return:
        """

        if label:
            plt.title(label.title())
        plt.imshow(image)
        plt.show()

    def plot_loss_and_accuracy(self, history, args):
        """
        :param history:
        :param args: list of arguments (list)
        :return:
        """

        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, constant.EPOCHS), history.history["loss"], label="train_loss")
        plt.plot(np.arange(0, constant.EPOCHS), history.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, constant.EPOCHS), history.history["acc"], label="train_acc")
        plt.plot(np.arange(0, constant.EPOCHS), history.history["val_acc"], label="val_acc")
        plt.title("Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="upper left")
        plt.savefig(args["plot"])
        plt.show()

    def evaluation_metrics(self, model, x_train, x_test, y_train, y_test):
        """
        Evaluate the model on the train/test sets
        :param model: keras.models.Model
        :param x_train: training examples (numpy array)
        :param x_test: test examples (numpy array)
        :param y_train: training labels (numpy array)
        :param y_test: test labels (numpy array)
        """

        score_train = model.evaluate(x_train, y_train)
        score_test = model.evaluate(x_test, y_test)

        print('[INFO] Accuracy on the Train Images: ', score_train[1])
        print('[INFO] Accuracy on the Test Images: ', score_test[1])

    def get_prediction_and_label(self, model, lb, example):
        """
        :param lb: label binarizer class
        :param model: keras.models.Model
        :param example: example data (numpy array)
        :return:
        """

        print("[INFO] classifying example...")
        predictions = model.predict(example)[0]

        index = np.argmax(predictions)
        prediction = predictions[index]
        label = lb.classes_[index]

        return prediction, label

    def add_new_last_layers(self, x):
        """
        :param x:
        :return:
        """

        x = Flatten(name='flatten')(x)
        x = Dense(32, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(constant.CLASSES, activation='softmax')(x)

        return x
