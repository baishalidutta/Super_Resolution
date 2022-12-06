__author__ = "Baishali Dutta"
__copyright__ = "Copyright (C) 2021-2022 Baishali Dutta"
__license__ = "Apache License 2.0"
__version__ = "0.1"

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Activation
# import the necessary packages
from tensorflow.keras.models import Sequential


class SRCNN:
    @staticmethod
    def build(width, height, depth):
        # initialize the model
        model = Sequential()
        input_shape = (height, width, depth)

        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            input_shape = (depth, height, width)

        # the entire SRCNN architecture consists of three CONV =>
        # RELU layers with *no* zero-padding
        model.add(Conv2D(64, (9, 9), kernel_initializer="he_normal",
                         input_shape=input_shape))
        model.add(Activation("relu"))
        model.add(Conv2D(32, (1, 1), kernel_initializer="he_normal"))
        model.add(Activation("relu"))
        model.add(Conv2D(depth, (5, 5),
                         kernel_initializer="he_normal"))
        model.add(Activation("relu"))

        # return the constructed network architecture
        return model
