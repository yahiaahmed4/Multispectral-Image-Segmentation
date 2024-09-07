# Multispectral Image Segmentation

Welcome to the **Multispectral Image Segmentation** project. This repository contains code and documentation for a comprehensive image segmentation task using multispectral satellite images. The primary objective is to build and evaluate a U-Net model for segmenting water pixels from satellite imagery.

## Project Overview

This project aims to segment specific features from multispectral images, particularly focusing on identifying water bodies. The dataset used comprises multispectral images with 12 bands, and the segmentation model developed is based on the U-Net architecture, tailored for pixel-wise classification.

## Contents

- **`Multispectral-Image-Segmentation.ipynb`**: A Jupyter Notebook that includes all code for preprocessing, model building, training, and evaluation.
- **Data**: Organized into `data/images` and `data/labels` directories for input images and corresponding labels respectively.

## Key Components

### 1. Data Preprocessing

The preprocessing pipeline involves several key steps to prepare the data for training:

- **File Cleanup**: Removing redundant label files from the labels folder.
- **Band Visualization**: Visualizing each band of multispectral images to understand the data better.
- **Composite Visualization**: Creating RGB and false-color composites for visual analysis.
- **Normalization and Augmentation**: Loading and normalizing images, applying data augmentation techniques, and preparing the dataset for training using TensorFlow's `tf.data.Dataset`.

### 2. U-Net Model

The core of this project is the U-Net model for image segmentation:

- **Model Architecture**: The U-Net architecture consists of an encoding path to capture context and a decoding path for precise localization. Key components include convolutional layers, max pooling, upsampling, and concatenation operations.
- **Training**: The model is trained using binary crossentropy loss and optimized with the Adam optimizer. Metrics such as accuracy, precision, recall, and Intersection over Union (IoU) are monitored.

### 3. Evaluation and Visualization

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

   The `requirements.txt` file should include libraries such as `tensorflow`, `numpy`, `matplotlib`, `rasterio`, and `PIL`.

3. **Prepare the Dataset**

   Organize your dataset into the `data/images` and `data/labels` directories. Make sure your images are in TIFF format and labels are in PNG format.

4. **Run the Notebook**

   Open the Jupyter Notebook `Multispectral-Image-Segmentation.ipynb` and execute the cells to preprocess the data, build and train the model, and evaluate its performance.

## Results

- **Training Metrics**: Includes loss, accuracy, precision, recall, and IoU metrics plotted over epochs.
- **Test Results**: Performance metrics on the test set, including loss, accuracy, precision, recall, and IoU.

## Contributing

Contributions are welcome! If you have suggestions or improvements, please feel free to open an issue or submit a pull request.

