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
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            train_img_data_path = os.path.join(self.ingestion_config.base_dir, "train")
            test_img_data_path = os.path.join(self.ingestion_config.base_dir, "test")
            val_img_data_path = os.path.join(self.ingestion_config.base_dir, "val")
            
            #Load train data
            train_data = image_dataset_from_directory(
                train_img_data_path,
                image_size=(self.ingestion_config.img_height, self.ingestion_config.img_width),
                batch_size=self.ingestion_config.batch_size,
                label_mode='binary'
            )

            #Load test data
            test_data = image_dataset_from_directory(
                test_img_data_path,
                image_size=(self.ingestion_config.img_height, self.ingestion_config.img_width),
                batch_size=self.ingestion_config.batch_size,
                label_mode='binary'
            )

            #Load validation data
            val_data = image_dataset_from_directory(
                val_img_data_path,
                image_size=(self.ingestion_config.img_height, self.ingestion_config.img_width),
                batch_size=self.ingestion_config.batch_size,
                label_mode='binary'
            )

            #Output Data Paths
            train_data_path = os.path.join(self.ingestion_config.artifact_dir, "train")
            test_data_path = os.path.join(self.ingestion_config.artifact_dir, "test")
            val_data_path = os.path.join(self.ingestion_config.artifact_dir, "val")

            save_data(train_data, train_data_path)
            save_data(test_data, test_data_path)
            save_data(val_data, val_data_path)

            return train_data_path, test_data_path, val_data_path

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()

