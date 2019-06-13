import imutils
import cv2
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from helper import constant
from data_handler import DataHandler
from network_model import NetworkModel


class NeuralNetwork:
    def __init__(self):
        self.data_handler = DataHandler()
        self.network_model = NetworkModel()

    def train(self, args):
        """
        :return:
        """

        vgg_model = self.network_model.pretrained_model()

        # Loading dataset
        x_train, x_test, y_train, y_test = self.data_handler.load_dataset(args)
        aug = self.data_handler.data_augmentation()

        # Getting output tensor of the last VGG layer that we want to include
        x = vgg_model.output
        x = self.network_model.add_new_last_layers(x)

        model = self.network_model.create(vgg_model, x)

        checkpoint = ModelCheckpoint(
            constant.MODEL_PATH,
            monitor="val_acc",
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            period=1
        )

        history = model.fit_generator(
            aug.flow(x_train, y_train, batch_size=constant.BATCH_SIZE),
            epochs=constant.EPOCHS,
            validation_data=(x_test, y_test),
            steps_per_epoch=len(x_train) // constant.BATCH_SIZE,
            callbacks=[checkpoint]
        )

        self.network_model.plot_loss_and_accuracy(history, args)
        self.network_model.evaluation_metrics(model, x_train, x_test, y_train, y_test)

        return model

    def evaluate(self, args):
        print("[INFO] loading network...")
        model = load_model(args["model"])
        x_train, x_test, y_train, y_test = self.data_handler.load_dataset(args)

        self.network_model.evaluation_metrics(model, x_train, x_test, y_train, y_test)

    def run(self, args, image_file):
        """
        :param args:
        :param image_file:
        :return:
        """

        print("[INFO] loading network...")
        model = load_model(args["model"])

        lb = self.data_handler.load_labels(args)
        example, image = self.data_handler.load_example(image_file)
        prediction, label = self.network_model.get_prediction_and_label(model, lb, example)

        percentage = prediction * 100
        if percentage < 50:
            label = constant.UNKNOWN_LABEL

        label = "{}: {:.2f}%".format(label, percentage)
        output = cv2.resize(image, (constant.IMAGE_SHOW_WIDTH, constant.IMAGE_SHOW_HEIGHT))

        cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return output


