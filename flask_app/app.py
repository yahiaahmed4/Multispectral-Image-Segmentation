from flask import Flask, request, render_template, send_from_directory
import numpy as np
import rasterio
import tensorflow as tf
import io
from PIL import Image
import matplotlib.pyplot as plt
import traceback
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
model = tf.keras.models.load_model('U-NET_Model.h5')

def load_image(image_stream):
    with rasterio.open(image_stream) as src:
        image = src.read()  # (bands, height, width)
        min_values = np.min(image, axis=(1, 2))
        max_values = np.max(image, axis=(1, 2))
        normalized_image = (image - min_values[:, np.newaxis, np.newaxis]) / \
                           (max_values[:, np.newaxis, np.newaxis] - min_values[:, np.newaxis, np.newaxis])
    return np.moveaxis(normalized_image, 0, -1)  # (height, width, bands)

def create_rgb_composite(image):
    red = image[:, :, 1]
    green = image[:, :, 2]
    blue = image[:, :, 3]
    rgb_image = np.stack([red, green, blue], axis=-1)
    return rgb_image




def save_image(image_array, filename, size=(224, 224), is_rgb=False):
    if is_rgb:
        # Handle RGB image
        img = Image.fromarray((image_array * 255).astype(np.uint8), mode='RGB')
    else:
        # Ensure mask is binary (0 and 1)
        binary_mask = (image_array > 0.5).astype(np.uint8)  # Convert to 0 and 1
        img = Image.fromarray((binary_mask * 255).astype(np.uint8), mode='L')  # For black and white masks (0=black, 255=white)

    # Resize the image to 224x224
    img = img.resize(size, Image.Resampling.LANCZOS)  # High-quality resizing

    # Save the image
    img.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))



def visualize_rgb_and_label(image, label):
    rgb_filename = 'rgb_composite.png'
    mask_filename = 'predicted_mask.png'
    save_image(create_rgb_composite(image), rgb_filename, is_rgb=True)
    save_image(label, mask_filename, is_rgb=False)

    return rgb_filename, mask_filename

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return "No file part", 400
        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400
        if file and file.filename.lower().endswith('.tif'):
            image_stream = io.BytesIO(file.read())
            new_image = load_image(image_stream)
            new_image = tf.expand_dims(new_image, axis=0)
            predicted_mask = model.predict(new_image)
            predicted_mask_for_display = predicted_mask.squeeze()
            rgb_filename, mask_filename = visualize_rgb_and_label(new_image[0], predicted_mask_for_display)
            return render_template('result.html', rgb_composite=rgb_filename, predicted_mask=mask_filename)
        else:
            return "Invalid file format", 400
    except Exception as e:
        print(f"An error occurred: {e}")
        print(traceback.format_exc())
        return "An error occurred", 500

@app.route('/uploads/<filename>')
def serve_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
