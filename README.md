# Multispectral Image Segmentation

Welcome to the **Multispectral Image Segmentation** project. This repository contains code and documentation for a comprehensive image segmentation task using multispectral satellite images. The primary objective is to build, fine-tune, and evaluate a U-Net model for segmenting water pixels from satellite imagery.

## Project Overview

This project aims to segment specific features from multispectral images, particularly focusing on identifying water bodies. The dataset used comprises multispectral images with 12 bands, and the segmentation model developed is based on the U-Net architecture, tailored for pixel-wise classification.

## Contents

- **`Multispectral-Image-Segmentation.ipynb`**: A Jupyter Notebook that includes all code for preprocessing, model building, training, and evaluation using a U-Net model from scratch.
- **`Fine-Tuning-U-Net.ipynb`**: A Jupyter Notebook that demonstrates the fine-tuning of a pretrained U-Net model with a ResNet50 encoder on the dataset. This notebook includes modifications for handling 12-channel input data and additional training details.
- **`flask_app`**: A Flask-based web application where users can upload `.tif` images, preprocess them, and visualize predictions (RGB composite and segmentation mask) made by the model.

## Key Components

### 1. Data Preprocessing

The preprocessing pipeline involves several key steps to prepare the data for training:

- **File Cleanup**: Removing redundant label files from the labels folder.
- **Band Visualization**: Visualizing each band of multispectral images to understand the data better.
- **Composite Visualization**: Creating RGB and false-color composites for visual analysis.
- **Normalization and Augmentation**: Loading and normalizing images, applying data augmentation techniques, and preparing the dataset for training using TensorFlow's `tf.data.Dataset`.

### 2. U-Net Model (From Scratch)

The initial approach uses a U-Net model built from scratch for image segmentation:

- **Model Architecture**: The U-Net architecture consists of an encoding path to capture context and a decoding path for precise localization. Key components include convolutional layers, max pooling, upsampling, and concatenation operations.
- **Training**: The model is trained using binary crossentropy loss and optimized with the Adam optimizer. Metrics such as accuracy, precision, recall, and Intersection over Union (IoU) are monitored.

### 3. Fine-Tuning U-Net Model

In addition to the scratch model, a pretrained U-Net model with a ResNet50 encoder has been fine-tuned:

- **Pretrained Model**: The `Fine-Tuning-U-Net.ipynb` notebook demonstrates the fine-tuning of a U-Net model with a ResNet50 encoder, specifically adapted to handle 12-channel input data.
- **Training**: The fine-tuning process involves loading a pretrained U-Net model, modifying the input layer to handle 12-channel data, and training it on the dataset. The model's performance is evaluated using similar metrics as the scratch model.

### 4. Flask Web Application (`flask_app`)

A simple web-based interface has been built using Flask, allowing users to upload multispectral `.tif` images and get predictions from the trained model. The app visualizes both the RGB composite and the predicted mask.

#### Key Features:

- **File Upload**: Users can upload `.tif` images with 12 bands.
- **Prediction and Visualization**: After uploading, the app preprocesses the image, runs the model prediction, and returns an RGB composite image alongside the predicted segmentation mask (black and white).
- **Outputs**: The output includes an RGB visualization and a predicted mask displayed on the screen.

#### How to Use the Flask App:

1. **Navigate to the `flask_app` Directory**:

   ```bash
   cd flask_app
   ```

2. **Install Flask and Required Dependencies**:

   Ensure you have Flask and any other required dependencies installed. You can install them using:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Flask App**:

   To start the Flask app, use the following command:

   ```bash
   python app.py
   ```

   This will start the development server at `http://127.0.0.1:5000`.

4. **Upload a .tif Image**:

   Navigate to the local server (usually `http://127.0.0.1:5000`), and use the web interface to upload a `.tif` image. The app will display the predicted RGB composite and segmentation mask.

#### Flask App Folder Structure:

- **`app.py`**: The main Flask app file that handles image uploading, preprocessing, and prediction.
- **`static/`**: Folder containing static files such as CSS for styling the web page.
- **`templates/`**: Contains the HTML file `index.html` that defines the user interface for uploading images and displaying results.
- **`uploads/`**: A directory where uploaded `.tif` files and generated predictions (RGB composites and masks) are temporarily stored.

### 5. Evaluation and Visualization

- **Metrics**: Training and validation losses, accuracy, precision, recall, and IoU are tracked.
- **Predictions**: The model's performance is evaluated on the test dataset, and predictions are visualized alongside true labels.

## Setup

To run the code in this repository, follow these steps:

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yahiaahmed4/Multispectral-Image-Segmentation.git
   cd Multispectral-Image-Segmentation
   ```

2. **Install Dependencies**

   Ensure you have the required libraries. You can install them using pip:

   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` file includes libraries such as `tensorflow`, `numpy`, `matplotlib`, `rasterio`, `PIL`, and `Flask`.

3. **Prepare the Dataset**

   Organize your dataset into the `data/images` and `data/labels` directories. Make sure your images are in TIFF format and labels are in PNG format.

4. **Run the Notebooks**

   Open the Jupyter Notebooks `Multispectral-Image-Segmentation.ipynb` and `Fine-Tuning-U-Net.ipynb` and execute the cells to preprocess the data, build and train the models, and evaluate their performance.

5. **Run the Flask App**

   Follow the instructions in the **Flask Web Application** section to upload images and view predictions.

## Results

- **Training Metrics**: Includes loss, accuracy, precision, recall, and IoU metrics plotted over epochs for both models.
- **Test Results**: Performance metrics on the test set, including loss, accuracy, precision, recall, and IoU.
- **Flask App Output**: Upload `.tif` files and visualize the model’s predictions for RGB composite and segmentation masks in the web app.

## Contributing

Contributions are welcome! If you have suggestions or improvements, please feel free to open an issue or submit a pull request.

