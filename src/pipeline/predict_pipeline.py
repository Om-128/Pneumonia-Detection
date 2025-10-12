import os
import sys
from dataclasses import dataclass
from src.exception import CustomException
from tensorflow.keras.models import load_model
from src.utils import load_preprocessor

@dataclass
class PredictPipelineConfig:
    model_path = os.path.join('artifacts', 'model.h5')
    image_processor_path = os.path.join('artifacts', 'image_preprocessor.pkl')

class PredictPipeline:

    def __init__(self, config: PredictPipelineConfig):
        self.config = config
        self.model: Optional[object] = None
        self.preprocessor: Optional[object] = None
    
    def load_model_lazy(self):
        if self.model is None:
            if not os.path.exists(self.config.model_path):
                raise FileNotFoundError(f"Model not found at {self.config.model_path}")
            self.model = load_model(self.config.model_path)
        return self.model

    def load_preprocessor_lazy(self):
        if self.preprocessor is None:
            if not os.path.exists(self.config.image_processor_path):
                raise FileNotFoundError(f"Preprocessor not found at {self.config.image_processor_path}")
            self.preprocessor = load_preprocessor(self.config.image_processor_path)
        return self.preprocessor

    def predict(self, img_path) -> float:
        try:
            model = self.load_model_lazy()
            preprocessor = self.load_preprocessor_lazy()

            scaled_img = preprocessor.transform_image(img_path)
            prediction = model.predict(scaled_img)[0][0]

            return prediction

        except Exception as e:
            raise CustomException(e, sys)