__author__ = "Baishali Dutta"
__copyright__ = "Copyright (C) 2021-2022 Baishali Dutta"
__license__ = "Apache License 2.0"
__version__ = "0.1"

from tensorflow.keras.layers import Dropout, Flatten, Dense


class FCHeadNet:
    @staticmethod
    def build(base_model, classes, D):
        # initialize the head model that will be placed on top of
        # the base, then add a FC layer
        head_model = base_model.output
        head_model = Flatten(name="flatten")(head_model)
        head_model = Dense(D, activation="relu")(head_model)
        head_model = Dropout(0.5)(head_model)

        # add a softmax layer
        head_model = Dense(classes, activation="softmax")(head_model)

        # return the model
        return head_model
