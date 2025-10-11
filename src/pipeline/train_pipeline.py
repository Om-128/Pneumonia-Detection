import os
import sys

from src.components.data_ingestion import DataIngestionConfig, DataIngestion
from src.components.data_transformation import ImageTransformation
from src.components.model_trainer import ModelTrainerConfig, ModelTrainer
from src.utils import save_preprocessor, load_preprocessor
from tensorflow.keras.applications.resnet50 import preprocess_input

import pickle
import tensorflow as tf

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
        train_path, test_path = data_ingestion.initiate_data_ingestion()

        # Step 2️⃣ — Create Train/Val Split from Train Folder
        print("\n📂 Creating training & validation datasets...")

        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_path,
            image_size=(224, 224),
            batch_size=32,
            validation_split=0.2,
            subset='training',
            seed=42
        )

        train_ds = train_ds.shuffle(buffer_size=1000).map(lambda x, y: (preprocess_input(x), y)).prefetch(tf.data.AUTOTUNE)


        val_ds = tf.keras.utils.image_dataset_from_directory(
            train_path,
            image_size=(224, 224),
            batch_size=32,
            validation_split=0.2,
            subset='validation',
            seed=42
        )

        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_path,
            image_size=(224, 224),
            batch_size=32,
            shuffle=False
        )

        # Step 2 — Data Preprocessing
        print("\n🔧 Applying preprocessing...")
        val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y))
        test_ds = test_ds.map(lambda x, y: (preprocess_input(x), y))

        val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
        test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
        
        # Step 3 — Save preprocessors
        image_preprocessor = ImageTransformation()
        save_preprocessor(image_preprocessor, "artifacts/image_preprocessor.pkl")

        # Step 4 — Model Training
        model = ModelTrainer()
        model, history = model.train_model(train_ds, val_ds)
        
        print(history)
        
        return history, model, train_ds, val_ds, test_ds, image_preprocessor

    except Exception as e:
        raise CustomException(e, sys)

if __name__ == "__main__":
    history, model, train_ds, val_ds, test_ds, image_preprocessor = train_pipeline()

