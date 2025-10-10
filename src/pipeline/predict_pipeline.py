import os
import sys
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

    def __init__(self, config: PredictPipelineConfig):
        self.config = config
    
    def loadImagePreprocessor(self):
        try:
            if not os.path.exists(self.config.image_processor_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            else:
                return load_preprocessor(self.config.image_processor_path)
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, img_path):
        try:
            image_preprocessor = self.loadImagePreprocessor()
            model = load_model(self.config.model_path)
            scaled_img = image_preprocessor.transform_image(img_path)
            result = model.predict(scaled_img)
            return result
        except Exception as e:
            print(e)
            raise CustomException(e, sys)


if __name__ == "__main__":
    config = PredictPipelineConfig()
    predict_pipeline = PredictPipeline(config=config)
    result = predict_pipeline.predict('normal3.jpeg')
    if result > 0.5:
        print("Prediction: Pneumonia")
    else:
        print("Prediction: Normal")
