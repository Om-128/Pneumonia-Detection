import os
import sys

from src.components.data_ingestion import DataIngestionConfig, DataIngestion
from src.components.data_transformation import DataTransformation, ImageTransformation
from src.utils import save_preprocessor, load_preprocessor

import pickle

from src.exception import CustomException

def train_pipeline():
    try:

        """
        Train pipeline for pneumonia detection:
        - Runs data ingestion
        - Preprocesses datasets
        - Saves preprocessors
        - Returns preprocessed datasets and preprocessor objects
        """

        # Step 1 — Data Ingestion
        data_ingestion = DataIngestion()
        train_path, test_path, val_path = data_ingestion.initiate_data_ingestion()

        # Step 2 — Data Preprocessing
        dataset_preprocessor = DataTransformation()
        train_ds = dataset_preprocessor.preprocess_data(train_path, augment=True)
        test_ds = dataset_preprocessor.preprocess_data(test_path, augment=False)
        val_ds = dataset_preprocessor.preprocess_data(val_path, augment=False)

        image_preprocessor = ImageTransformation()

        # Step 3 — Save preprocessors
        save_preprocessor(dataset_preprocessor, "artifacts/dataset_preprocessor.pkl")
        save_preprocessor(image_preprocessor, "artifacts/image_preprocessor.pkl")

        return train_ds, val_ds, test_ds, dataset_preprocessor, image_preprocessor
    except Exception as e:
        raise CustomException(e, sys)

if __name__ == "__main__":
    train_ds, val_ds, test_ds, dataset_preprocessor, image_preprocessor = train_pipeline()
