from tensorflow import keras
from model import *
# import numba as nb


def main():

    # Loading MNIST dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # print("Initial shapes: \n", x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    x_train = x_train.reshape(*x_train.shape, 1) / 255
    x_test = x_test.reshape(*x_test.shape, 1) / 255
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    # Initializing class with single image as an input (batch size = 1)
    cnn = CNN(x_train[:1], y_train[:1])

    # Performing training algorithm
    cnn.train(x_train[:625], y_train[:625])

    # Plot history
    cnn.plot_history()

    return 1


if __name__ == "__main__":
    main()
