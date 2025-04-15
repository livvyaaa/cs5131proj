import numpy as np
import rasterio
import tensorflow as tf
from PIL import Image
from flask import Blueprint, Flask, request, render_template, jsonify
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for matplotlib
import matplotlib.pyplot as plt
import io
import base64

# Create the Blueprint
app1 = Blueprint('app1', __name__, static_folder='static', template_folder='templates')
@app1.route('/')
def app1_home():
    return render_template('upload.html')  # The template for app1

# Load the trained model
model = tf.keras.models.load_model("landsat/fire_detection_model.keras")

# Image size expected by the model
IMG_SIZE = (128, 128)
BANDS = 4  # Red, NIR, SWIR1, SWIR2

# Normalize image band
def normalize_band(band):
    band = band.astype(np.float32)
    return (band - band.min()) / (band.max() - band.min() + 1e-6)

# Convert 4-band to RGB (using SWIR1, NIR, Red)
def convert_4band_to_rgb(img_4band):
    return img_4band[:, :, [2, 1, 0]]  # SWIR1 (B6), NIR (B5), Red (B4)

# Function to process the image and make predictions
def predict_fire(image_file):
    with rasterio.open(image_file) as src:
        bands = []
        for i in range(4):  # Assuming there are 4 bands
            band = src.read(i + 1)
            band_normalized = normalize_band(band)
            resized = Image.fromarray((band_normalized * 255).astype(np.uint8)).resize(IMG_SIZE)
            bands.append(np.array(resized) / 255.0)

        img = np.stack(bands, axis=-1)  # Stack the bands into a 4-channel image
        img_rgb = convert_4band_to_rgb(img)  # Convert to 3-channel RGB (SWIR1, NIR, Red)

    img_input = np.expand_dims(img_rgb, axis=0)  # Expand dims to match input shape
    prediction = model.predict(img_input)
    return "Fire" if prediction > 0.5 else "No Fire", img_rgb

# Function to convert image to base64 for preview
def image_to_base64(img_rgb):
    # Convert RGB image to PNG byte data
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_rgb)
    ax.axis('off')  # Hide axes
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64

@app1.route('/')
def upload_form():
    return render_template('upload.html')


# Route to handle the uploaded file and prediction via AJAX
@app1.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"result": "No file part", "img_base64": ""})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"result": "No selected file", "img_base64": ""})

    if file and file.filename.endswith('.tif'):
        # Save file temporarily and make prediction
        result, img_rgb = predict_fire(file)
        
        # Convert the image to base64 for displaying it on the frontend
        img_base64 = image_to_base64(img_rgb)

        return jsonify({"result": result, "img_base64": img_base64})

    return jsonify({"result": "Invalid file type. Only .tif is allowed", "img_base64": ""})

if __name__ == '__main__':
    app1.run(debug=True)
