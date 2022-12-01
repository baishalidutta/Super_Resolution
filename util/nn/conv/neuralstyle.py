__author__ = "Baishali Dutta"
__copyright__ = "Copyright (C) 2021-2022 Baishali Dutta"
__license__ = "Apache License 2.0"
__version__ = "0.1"

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import Model
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input


class NeuralStyle(Model):
    def __init__(self, style_layers, content_layers):
        # call the parent constructor
        super(NeuralStyle, self).__init__()

        # construct our network with the given set of layers
        self.vgg = self.vggLayers(style_layers + content_layers)

        # store the style layers, content layers, the number of style
        # layers, then set our network to non-trainable (if it is not
        # set already)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        # scale the pixel values of the image back to the [0, 255]
        # range and preprocess them
        inputs = inputs * 255.0
        preprocessed_input = preprocess_input(inputs)

        # run the preprocessed image through our network and grab the
        # style and content outputs
        outputs = self.vgg(preprocessed_input)
        (style_outputs, content_outputs) = (
            outputs[:self.num_style_layers],
            outputs[self.num_style_layers:])

        # compute the gram matrix between the  different style outputs
        style_outputs = [self.gramMatrix(style_output)
                         for style_output in style_outputs]

        # loop over the content layers (and their corresponding
        # outputs) and prepare a dictionary
        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}

        # loop over the style layers (and their corresponding outputs)
        # and prepare a dictionary
        style_dict = {styleName: value
                      for styleName, value
                      in zip(self.styleLayers, style_outputs)}

        # return a dictionary containing the style features, and the
        # content features
        return {"content": content_dict, "style": style_dict}

    @staticmethod
    def vgg_layers(layerNames):
        # load our model from disk and set it non-trainable
        vgg = VGG19(include_top=False, weights="imagenet")
        vgg.trainable = False

        # construct a list of outputs of the specified layers, and then
        # create the model
        outputs = [vgg.get_layer(name).output for name in layerNames]
        model = Model([vgg.input], outputs)

        # return the model
        return model

    @staticmethod
    def gram_matrix(input_tensor):
        # the gram matrix is the dot product between the input vectors
        # and their respective transpose
        result = tf.linalg.einsum("bijc,bijd->bcd",
                                  input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        locations = tf.cast(input_shape[1] * input_shape[2],
                            tf.float32)

        # return normalized gram matrix
        return (result / locations)

    @staticmethod
    def style_content_loss(outputs, style_targets, content_targets,
                         style_weight, content_weight):
        # extract the style and content outputs respectively
        style_outputs = outputs["style"]
        content_outputs = outputs["content"]

        # iterate over each of the style layers, grab their outputs,
        # and determine the mean-squared error with respect to the
        # original style content
        style_loss = [tf.reduce_mean((
                                             style_outputs[name] - style_targets[name]) ** 2)
                      for name in style_outputs.keys()]

        # add the individual style layer losses and normalize them
        style_loss = tf.add_n(style_loss)
        style_loss *= style_weight

        # iterate over each content layers, grab their outputs, and
        # determine the mean-squared error with respect to the
        # original  image content
        contentLoss = [tf.reduce_mean((content_outputs[name] -
                                       content_targets[name]) ** 2)
                       for name in content_outputs.keys()]

        # add the indvidual content layer losses and normalize them
        content_loss = tf.add_n(contentLoss)
        content_loss *= content_weight

        # add the final style and content losses
        loss = style_loss + content_loss

        # return the combined loss
        return loss

    @staticmethod
    def clipPixels(image):
        # clip any pixel values in the image falling outside the
        # range [0, 1] and return the image
        return tf.clip_by_value(image,
                                clip_value_min=0.0,
                                clip_value_max=1.0)

    @staticmethod
    def tensorToImage(tensor):
        # scale pixels back to the range [0, 255] and convert the
        # the data type of the pixels to integer
        tensor = tensor * 255
        tensor = np.array(tensor, dtype=np.uint8)

        # remove the batch dimension from the image if it is
        # present
        if np.ndim(tensor) > 3:
            tensor = tensor[0]

        # return the image in a PIL format
        return Image.fromarray(tensor)
