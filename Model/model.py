from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Conv2D, Flatten, Activation, MaxPooling2D

class Model():
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        # CONV => RELU => BN => POOL
        model.add(Conv2D(32, (5, 5), padding="same",
            input_shape=inputShape))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # first set of (CONV => RELU => CONV => RELU) * 2 => POOL
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation("relu"))
        # model.add(Conv2D(64, (3, 3), padding="same"))
        # model.add(BatchNormalization(axis=chanDim))
        # model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # second set of (CONV => RELU => CONV => RELU) * 2 => POOL
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation("relu"))
        # model.add(Conv2D(128, (3, 3), padding="same"))
        # model.add(BatchNormalization(axis=chanDim))
        # model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # first set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation("relu"))
        model.add(Dropout(0.2))

        # second set of FC => RELU layers
        # model.add(Flatten())
        # model.add(Dense(128))
        # model.add(Activation("relu"))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model

