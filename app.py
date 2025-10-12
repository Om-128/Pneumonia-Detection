import streamlit as st
import tempfile
from PIL import Image
from src.pipeline.predict_pipeline import PredictPipeline, PredictPipelineConfig
import io

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🫁",
    layout="centered"
)

# -----------------------------
# Title and description
# -----------------------------
st.markdown("<h1 style='text-align: center;'>🫁 Pneumonia Detection</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center;'>Upload a chest X-ray image and the model will predict if it shows signs of pneumonia.</p>",
    unsafe_allow_html=True
)

# -----------------------------
# File uploader
# -----------------------------
uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Read file content once
    file_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(file_bytes))

    # Center image preview using columns
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption="Uploaded X-ray", width=300)

    # Load pipeline once
    config = PredictPipelineConfig()
    pipeline = PredictPipeline(config)

    with col2:
        predict = st.button("Predict", width=300, type="primary")

    if predict:
        with col2:
            with st.spinner("Predicting... ⏳", width='stretch'):
                # Save uploaded file to a temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    tmp.write(file_bytes)
                    tmp_path = tmp.name

                # Run prediction
                prediction = pipeline.predict(tmp_path)

                # Display result centered
        # Show result centered
        if prediction > 0.7:
            st.markdown(
                "<h3 style='text-align: center; color:red;'>⚠️ Prediction: Pneumonia detected!</h3>",
                unsafe_allow_html=True
            )
            st.markdown(
                "<p style='text-align: center; color:red;'>You have symptoms of Pneumonia! Please consult a doctor.</p>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<h3 style='text-align: center; color:green;'>✅ Prediction: Normal</h3>",
                unsafe_allow_html=True
            )
            st.markdown(
                "<p style='text-align: center; color:green;'>Your X-ray is normal, Don't worry.</p>",
                unsafe_allow_html=True
            )