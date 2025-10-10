import os
import sys
import tensorflow as tf
from dataclasses import dataclass
from src.exception import CustomException
import numpy as np


class ModelTrainerConfig:
    model_path = os.path.join("artifacts", "model.h5")
    epochs : int = 10
    learning_rate : float = 0.0001
    dropout_rate : float = 0.6
    fine_tune_layers: int = 30

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

            #Add Data Augmentation
            data_augmentation = tf.keras.Sequential([
                    tf.keras.layers.RandomFlip("horizontal"),
                    tf.keras.layers.RandomRotation(0.1),
                    tf.keras.layers.RandomZoom(0.1)
            ])

            inputs = tf.keras.Input(shape=input_shape)
            x = data_augmentation(inputs)
            x = base_model(x, training=False)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.Dense(128, activation='relu')(x)
            x = tf.keras.layers.Dropout(self.config.dropout_rate)(x)
            outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

            model = tf.keras.Model(inputs, outputs)

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

            # Checkpoint Saving
            checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
                filepath=self.config.model_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )

            history = model.fit(
                train_ds,
                validation_data = val_ds,
                epochs = self.config.epochs,
                callbacks=[early_stop, lr_scheduler, checkpoint_cb]
            )

            # print("\n🔧 Unfreezing top layers for fine-tuning...")
            # for layer in model.layers[-self.config.fine_tune_layers:]:
            #     if hasattr(layer, "trainable"):
            #         layer.trainable = True

            # model.compile(
            #     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            #     loss='binary_crossentropy',
            #     metrics=['accuracy']
            # )

            # history2 = model.fit(
            #     train_ds,
            #     validation_data=val_ds,
            #     epochs=5,
            #     callbacks=[early_stop, lr_scheduler]
            # )

            # history = {}
            # for k in history1.history.keys():
            #     history[k] = history1.history[k] + history2.history.get(k, [])

            return model, history

        except Exception as e:
            raise CustomException(e, sys)