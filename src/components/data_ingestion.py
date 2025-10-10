import pandas as pd
import tensorflow as tf
import os
import sys
from tensorflow.keras.utils import image_dataset_from_directory
from src.exception import CustomException
from dataclasses import dataclass
from src.utils import save_data

@dataclass
class DataIngestionConfig:
    base_dir: str = os.path.join("data", "chest_xray")
    artifact_dir: str = os.path.join("artifacts", "datasets")
    img_height: int = 224
    img_width: int = 224
    batch_size: int = 32

'''
    Loads images from directories and converts them into tf.data.Dataset objects.
    Converts these datasets into TFRecords format for efficient storage.
    Saves the TFRecords to data paths for train, test, and validation sets.
    Returns the paths to these saved datasets for later use.
    
'''
class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            train_path = os.path.join(self.config.base_dir, "train")
            test_path = os.path.join(self.config.base_dir, "test")
            val_path = os.path.join(self.config.base_dir, "val")  # optional if available

            if not os.path.exists(train_path) or not os.path.exists(test_path):
                raise FileNotFoundError(f"Train/Test directories not found in {self.config.base_dir}")

            return train_path, test_path

        except Exception as e:
            raise CustomException(e, sys)