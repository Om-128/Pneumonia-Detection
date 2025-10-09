import os
import sys
import tensorflow as tf
from dataclasses import dataclass
from src.exception import CustomException

class ModelTrainerConfig:
    model_path = os.path.join("artifacts", "model.h5")
    epochs : int = 10
    learning_rate : float = 0.0001
    dropout_rate : float = 0.6  

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def build_model(self, input_shape=(224, 224, 3)):
        try:
            """
            Builds a transfer learning model using ResNet50 pretrained on ImageNet.
            """
            base_model = tf.keras.applications.ResNet50(
                weights='imagenet', 
                include_top=False, 
                input_shape=input_shape
            )

            base_model.trainable = False # Freeze base model weights initially

            model = tf.keras.Sequential([
                base_model,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(self.config.dropout_rate),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])

            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )

            return model

        except Exception as e:
            raise CustomException(e, sys) 

    def train_model(self, train_ds, val_ds):
        """
        Trains the ResNet50-based pneumonia detection model.
        """
        try:
            model = self.build_model()

            # Early stopping callback
            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True,
                verbose=1
            )

            # Reduce learning rate when stuck
            lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=2,
                verbose=1
            )

            history = model.fit(
                train_ds,
                validation_data = val_ds,
                epochs = self.config.epochs,
                callbacks=[early_stop, lr_scheduler]
            )

            # Save model
            os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)
            model.save(self.config.model_path)

            return model, history

        except Exception as e:
            CustomException(e, sys)
