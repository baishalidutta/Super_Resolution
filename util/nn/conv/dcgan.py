__author__ = "Baishali Dutta"
__copyright__ = "Copyright (C) 2021-2022 Baishali Dutta"
__license__ = "Apache License 2.0"
__version__ = "0.1"

from tensorflow.keras.layers import BatchNormalization, Conv2DTranspose, Conv2D, LeakyReLU, Activation, Flatten, Dense, \
    Reshape
from tensorflow.keras.models import Sequential


class DCGAN:
    @staticmethod
    def build_generator(dim, depth, channels=1, input_dim=100,
                        output_dim=512):
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        model = Sequential()
        input_shape = (dim, dim, depth)
        chan_dim = -1

        # first set of FC => RELU => BN layers
        model.add(Dense(input_dim=input_dim, units=output_dim))
        model.add(Activation("relu"))
        model.add(BatchNormalization())

        # second set of FC => RELU => BN layers, this time preparing
        # the number of FC nodes to be reshaped into a volume
        model.add(Dense(dim * dim * depth))
        model.add(Activation("relu"))
        model.add(BatchNormalization())

        # reshape the output of the previous layer set, upsample +
        # apply a transposed convolution, RELU, and BN
        model.add(Reshape(input_shape))
        model.add(Conv2DTranspose(32, (5, 5), strides=(2, 2),
                                  padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chan_dim))

        # apply another upsample and transposed convolution, but
        # this time output the TANH activation
        model.add(Conv2DTranspose(channels, (5, 5), strides=(2, 2),
                                  padding="same"))
        model.add(Activation("tanh"))

        # return the generator model
        return model

    @staticmethod
    def build_discriminator(width, height, depth, alpha=0.2):
        # initialize the model along with the input shape to be
        # "channels last"
        model = Sequential()
        input_shape = (height, width, depth)

        # first set of CONV => RELU layers
        model.add(Conv2D(32, (5, 5), padding="same", strides=(2, 2),
                         input_shape=input_shape))
        model.add(LeakyReLU(alpha=alpha))

        # second set of CONV => RELU layers
        model.add(Conv2D(64, (5, 5), padding="same", strides=(2, 2)))
        model.add(LeakyReLU(alpha=alpha))

        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=alpha))

        # sigmoid layer outputting a single value
        model.add(Dense(1))
        model.add(Activation("sigmoid"))

        # return the discriminator model
        return model
