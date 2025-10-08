import os
import sys
import tensorflow as tf
from src.exception import CustomException

def save_data(dataset, output):
    """
    Saves a TensorFlow dataset to the specified directory in TFRecord format.

    Args:
        dataset (tf.data.Dataset): TensorFlow dataset to save.
        output (str): Directory path where dataset will be saved.
        
    """
    try:
        os.makedirs(output, exist_ok=True)
        tf.data.Dataset.save(dataset, output)
    except Exception as e:
        raise CustomException(e, sys)