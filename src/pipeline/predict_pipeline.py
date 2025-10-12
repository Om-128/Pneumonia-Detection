import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import numpy as np
from src.exception import CustomException
from tensorflow.keras.preprocessing import image
from dataclasses import dataclass
from src.utils import load_preprocessor
from tensorflow.keras.models import load_model

@dataclass
class PredictPipelineConfig:
    model_path = os.path.join('artifacts', 'model.h5')
    image_processor_path = os.path.join('artifacts', 'image_preprocessor.pkl')

class PredictPipeline:

    _model = None
    _image_preprocessor = None

    def __init__(self, config: PredictPipelineConfig):
        self.config = config

        #Loading model once
        if PredictPipeline._model is None:
            PredictPipeline._model = load_model(config.model_path)
        
        # Load preprocessor once globally
        if PredictPipeline._image_preprocessor is None:
            if not os.path.exists(config.image_processor_path):
                raise FileNotFoundError(f"Image preprocessor not found at {config.image_processor_path}")
            print("🔹 Loading preprocessor once...")
            PredictPipeline._image_preprocessor = load_preprocessor(config.image_processor_path)

        self.model = PredictPipeline._model
        self.image_preprocessor = PredictPipeline._image_preprocessor

    def predict(self, img_path):
        try:
            scaled_img = selfimage_preprocessor.transform_image(img_path)
            result = self.model.predict(scaled_img)[0][0]
            return result
        except Exception as e:
            print(e)
            raise CustomException(e, sys)


