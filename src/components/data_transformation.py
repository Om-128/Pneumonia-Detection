import os
import sys
import os
import tensorflow as tf
from src.exception import CustomException
from tensorflow.keras.applications.resnet50 import preprocess_input

class ImageTransformation:
    def __init__(self, img_height=224, img_width=224):
        self.img_height = img_height
        self.img_width = img_width

    def transform_image(self, image_path):
        try:
            # Load and decode image
            image = tf.io.read_file(image_path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, [self.img_height, self.img_width])

            # Add batch dimension
            image = tf.expand_dims(image, axis=0)

            # Preprocess using ResNet50 preprocessing
            image = preprocess_input(image)

            return image

        except Exception as e:
            raise CustomException(e, sys)