import os
import sys
from dataclasses import dataclass
from src.exception import CustomException

@dataclass
class PredictPipelineConfig:
    model_path = os.path.join('artifacts', 'model.h5')
    image_processor_path = os.path.join('artifacts', 'image_preprocessor.pkl')

class PredictPipeline:

    def __init__(self, config: PredictPipelineConfig):
        self.config = config
        self.model = None  # Lazy load
    
    def load_model_lazy(self):
        if self.model is None:
            from tensorflow.keras.models import load_model
            if not os.path.exists(self.config.model_path):
                raise FileNotFoundError(f"Model file not found at {self.config.model_path}")
            self.model = load_model(self.config.model_path)
        return self.model
    
    def load_image_preprocessor(self):
        from src.utils import load_preprocessor
        if not os.path.exists(self.config.image_processor_path):
            raise FileNotFoundError(f"Preprocessor not found at {self.config.image_processor_path}")
        return load_preprocessor(self.config.image_processor_path)

    def predict(self, img_path):
        try:
            # Lazy load
            model = self.load_model_lazy()
            preprocessor = self.load_image_preprocessor()

            scaled_img = preprocessor.transform_image(img_path)

            result = self.model.predict(scaled_img)[0][0]
            return result
            
        except Exception as e:
            print(e)
            raise CustomException(e, sys)


