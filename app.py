from flask import Flask, request, jsonify, render_template, url_for
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from src.pipeline.predict_pipeline import PredictPipeline, PredictPipelineConfig
from src.exception import CustomException
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")  # <- render_template instead of jsonify

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        # Replace previous images
        for old_file in os.listdir(UPLOAD_FOLDER):
            os.remove(os.path.join(UPLOAD_FOLDER, old_file))

        #Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        #Make prediction
        predict_pipeline_cofig = PredictPipelineConfig()
        predict_pipeline = PredictPipeline(predict_pipeline_cofig)
        prediction = predict_pipeline.predict(filepath)

        if prediction > 0.5:
            message = "You have symptoms of Pneumonia!"
            label = "PNEUMONIA"
        else:
            message = "Your X-ray is normal."
            label = "NORMAL"

        return jsonify({
            "prediction": label,
            "message": message
        })

    except Exception as e:
        raise CustomException(e, sys)

if __name__ == "__main__":
    app.run(debug=True)
    #89f7fe
