import os
import sys
import tensorflow as tf
from src.exception import CustomException
import pickle

def save_data(dataset, output):
    """
    Saves a TensorFlow dataset to the specified directory in TFRecord format.

    """
    try:
        if not os.path.exists(output):
            os.makedirs(output, exist_ok=True)
            tf.data.Dataset.save(dataset, output)
        else:
            print(f"Dataset already exists at {output}, skipping save.")
    except Exception as e:
        raise CustomException(e, sys)

def save_preprocessor(preprocessor, path):
    """
    Save any preprocessor object to disk using pickle.
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(preprocessor, f)
    print(f"✅ Preprocessor saved at {path}")

def load_preprocessor(path):
    """
    Load a saved preprocessor object.
    
    """
    with open(path, "rb") as f:
        preprocessor = pickle.load(f)
    return preprocessor
