__author__ = "Baishali Dutta"
__copyright__ = "Copyright (C) 2021-2022 Baishali Dutta"
__license__ = "Apache License 2.0"
__version__ = "0.1"

from tensorflow.keras import backend as K
from tensorflow.keras.layers import BatchNormalization, Conv2D, AveragePooling2D, MaxPooling2D, Activation, Dropout, \
    Dense, Flatten, Input, concatenate
from tensorflow.keras.models import Model


class MiniGoogLeNet:
    @staticmethod
    def conv_module(x, K, kX, kY, stride, chan_dim, padding="same"):
        # define a CONV => BN => RELU pattern
        x = Conv2D(K, (kX, kY), strides=stride, padding=padding)(x)
        x = BatchNormalization(axis=chan_dim)(x)
        x = Activation("relu")(x)

        # return the block
        return x

    @staticmethod
    def inception_module(x, numK1x1, numK3x3, chan_dim):
        # define two CONV modules, then concatenate across the
        # channel dimension
        conv_1x1 = MiniGoogLeNet.conv_module(x, numK1x1, 1, 1,
                                             (1, 1), chan_dim)
        conv_3x3 = MiniGoogLeNet.conv_module(x, numK3x3, 3, 3,
                                             (1, 1), chan_dim)
        x = concatenate([conv_1x1, conv_3x3], axis=chan_dim)

        # return the block
        return x

    @staticmethod
    def downsample_module(x, K, chan_dim):
        # define the CONV module and POOL, then concatenate
        # across the channel dimensions
        conv_3x3 = MiniGoogLeNet.conv_module(x, K, 3, 3, (2, 2),
                                             chan_dim, padding="valid")
        pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = concatenate([conv_3x3, pool], axis=chan_dim)

        # return the block
        return x

    @staticmethod
    def build(width, height, depth, classes):
        # initialize the input shape to be "channels last" and the
        # channels dimension itself
        input_shape = (height, width, depth)
        chan_dim = -1

        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chan_dim = 1

        # define the model input and first CONV module
        inputs = Input(shape=input_shape)
        x = MiniGoogLeNet.conv_module(inputs, 96, 3, 3, (1, 1),
                                      chan_dim)

        # two Inception modules followed by a downsample module
        x = MiniGoogLeNet.inception_module(x, 32, 32, chan_dim)
        x = MiniGoogLeNet.inception_module(x, 32, 48, chan_dim)
        x = MiniGoogLeNet.downsample_module(x, 80, chan_dim)

        # four Inception modules followed by a downsample module
        x = MiniGoogLeNet.inception_module(x, 112, 48, chan_dim)
        x = MiniGoogLeNet.inception_module(x, 96, 64, chan_dim)
        x = MiniGoogLeNet.inception_module(x, 80, 80, chan_dim)
        x = MiniGoogLeNet.inception_module(x, 48, 96, chan_dim)
        x = MiniGoogLeNet.downsample_module(x, 96, chan_dim)

        # two Inception modules followed by global POOL and dropout
        x = MiniGoogLeNet.inception_module(x, 176, 160, chan_dim)
        x = MiniGoogLeNet.inception_module(x, 176, 160, chan_dim)
        x = AveragePooling2D((7, 7))(x)
        x = Dropout(0.5)(x)

        # softmax classifier
        x = Flatten()(x)
        x = Dense(classes)(x)
        x = Activation("softmax")(x)

        # create the model
        model = Model(inputs, x, name="googlenet")

        # return the constructed network architecture
        return model
