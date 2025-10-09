import os
import sys
import os
import tensorflow as tf
from src.exception import CustomException

class DataTransformation:
    """
    Preprocess datasets for training, validation, and testing.

    Steps:
        - Resize images
        - Normalize pixel values
        - Optionally augment data
        - Batch and prefetch for efficiency
    """
    def __init__(self, img_height=224, img_width=224, batch_size=32):
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
    
    def preprocess(self, image, label):
        try:
            #Resize images
            image = tf.image.resize(image, [self.img_height, self.img_width])
            #Normalize pixel values
            image = image / 255.0
            return image, label
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def augment(self, image, label):
        try:
            '''
                Apply simple augmentation to data
                Eg. Rotate, Adjust brightness
                This helps to reduce the chances of overfitting
            '''
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, max_delta=0.1)
            return image, label
        except Exception as e:
            raise CustomException(e, sys)

    def preprocess_data(self, dataset_path, augment=False):
        """
        Loads a dataset and applies preprocessing.

        Args:
            dataset_path (str): Path to TFRecord dataset.
            augment (bool): Whether to apply data augmentation.

        Returns:
            tf.data.Dataset: Preprocessed dataset.
        """
        try:
            #Load the dataset from path
            dataset = tf.data.Dataset.load(dataset_path)

            #Resize and rescale
            dataset = dataset.map(self.preprocess, num_parallel_calls=tf.data.AUTOTUNE)

            #If augment is true then apply autotune
            if augment:
                dataset = dataset.map(self.augment, num_parallel_calls=tf.data.AUTOTUNE)

            dataset = dataset.prefetch(tf.data.AUTOTUNE)

            return dataset

        except Exception as e:
            raise CustomException(e, sys)

class ImageTransformation:
    """
    Preprocess a single image for inference (web app).

    Steps:
        - Read image from path
        - Resize
        - Normalize pixels
    """

    def __init__(self, img_height=224, img_width=224):
        self.img_height = img_height
        self.img_width = img_width

    def transform_image(self, image_path):
        image = tf.io.read_file(image_path) #Read file
        image = tf.image.decode_jpeg(image, channels=3)  #Decode jpeg
        image = tf.image.resize(image, [self.img_height, self.img_width]) #Resize
        image = image / 255.0                                         # Normalize
        return image
        print("Image processed successfully")