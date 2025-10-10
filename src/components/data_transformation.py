import os
import sys
import os
import tensorflow as tf
from src.exception import CustomException


class ImageTransformation:
    def __init__(self, img_height=224, img_width=224):
        self.img_height = img_height
        self.img_width = img_width

    def transform_image(self, image_path):
        try:
            image = tf.io.read_file(image_path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, [self.img_height, self.img_width])
            image = image / 255.0
            return image
        except Exception as e:
            raise CustomException(e, sys)