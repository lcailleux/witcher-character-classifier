import os
import argparse
from helper import constant
from neural_network import NeuralNetwork

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", default=constant.MODEL_PATH, help="path to output model")
ap.add_argument("-d", "--dataset", default=constant.DATASET_PATH, help="path to input dataset")
ap.add_argument("-l", "--labelbin", default=constant.LB_PATH, help="path to output label binarizer")
ap.add_argument("-p", "--plot", type=str, default=constant.PLOT_PATH, help="path to output accuracy/loss plot")
ap.add_argument("--show_examples", action='store_true', help="Show some training examples")

args = vars(ap.parse_args())

neural_network = NeuralNetwork()
neural_network.evaluate(args)
