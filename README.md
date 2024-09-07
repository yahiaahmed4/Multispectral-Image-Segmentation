# Multispectral Image Segmentation

## Overview

The 'Multispectral Image Segmentation' project involves classifying water pixels in 12-dimensional satellite images using a U-Net model implemented from scratch. This repository contains a Jupyter Notebook named `Multispectral-Image-Segmentation.ipynb` that provides a step-by-step guide through the data preprocessing, model training, and evaluation processes.

## Repository Structure

- `Multispectral-Image-Segmentation.ipynb`: Jupyter Notebook with detailed code and explanations.

## Code Explanation

### 1. Data Preprocessing

**Objective**: Prepare the raw multispectral images and corresponding label masks for training the U-Net model.

- **Loading Data**: Images and label masks are loaded from the specified directories. Each image is associated with a label mask, where water pixels are highlighted.
- **Normalization**: The pixel values of the images are normalized to a range of [0, 1] by dividing by 255.0. This step ensures that the model receives data in a consistent scale, improving training stability.
- **Label Processing**: The label masks were already normalized to a range of [0, 1] to match the scale of the input images. This normalization facilitates better training and performance of the model.

### 2. Model Architecture

**Objective**: Define and build the U-Net model from scratch for image segmentation.

- **U-Net Structure**: The U-Net architecture includes an encoder and decoder with skip connections to capture multi-scale features. The encoder consists of convolutional and pooling layers to downsample the input, while the decoder uses upsampling layers to restore the spatial resolution.
- **Custom Layers**: Custom convolutional and activation layers are used to ensure that the network can learn complex patterns in the multispectral images.

### 3. Model Training

**Objective**: Train the U-Net model on the preprocessed data.

- **Compilation**: The model is compiled with a suitable loss function (e.g., binary crossentropy) and an optimizer (e.g., Adam). The choice of loss function and optimizer is crucial for effective training.
- **Training Process**: The model is trained using the training dataset, with validation on a separate validation set. This helps in monitoring overfitting and adjusting hyperparameters as needed.

### 4. Model Evaluation

**Objective**: Assess the performance of the trained U-Net model.

- **Evaluation Metrics**: Metrics such as Intersection over Union (IoU) and pixel accuracy are used to evaluate the model’s performance on the test set. These metrics provide insights into how well the model segments water pixels compared to ground truth labels.
- **Visualization**: Sample predictions are visualized alongside ground truth masks to qualitatively assess the model’s segmentation quality.


## How to Use

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yahiaahmed4/Multispectral-Image-Segmentation.git
   ```

2. **Navigate to the Project Directory**:
   ```bash
   cd Multispectral-Image-Segmentation
   ```

3. **Run the Notebook**: Open `Multispectral-Image-Segmentation.ipynb` in Jupyter Notebook or JupyterLab to view and execute the code.

## Contribution

Feel free to open issues or submit pull requests if you have suggestions or improvements. Contributions are welcome!

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or further information, please contact [Your Name](mailto:your.email@example.com).

---

Let me know if there's anything else you'd like to adjust or add!
