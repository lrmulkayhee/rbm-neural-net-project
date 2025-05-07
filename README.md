# Restricted Boltzmann Machine (RBM) Neural Network Project

## Overview

This project demonstrates the use of a Restricted Boltzmann Machine (RBM) to reconstruct noisy numeral images. The RBM is trained to denoise and reconstruct visual data, showcasing its ability to learn meaningful latent features.

## Purpose

The purpose of this project is to reconstruct noisy numeral images using an RBM. This approach demonstrates the RBM's ability to denoise and reconstruct visual data effectively, even in the presence of noise.

## Key Results

- **Prediction Accuracy**: Achieved a prediction accuracy of **100%** for noisy numeral images with a noise level of **0.0% to 0.3%**.
- **Reconstruction Accuracy**: Achieved an average prediction accuracy of **72.50%** across all noise levels.
- **Reconstruction Error**: The average reconstruction error was **0.4841**, indicating effective learning overall.
- **Reconstruction Variance**: The average variance in reconstruction was **0.2414**, showcasing consistent performance across noise levels.
- **Free Energy Gap**: The average free energy gap was **40.7754**, reflecting the RBM's ability to differentiate between noisy and clean data.

## Problem Description and Network Design

### Problem Statement

The task involves reconstructing noisy numeral images (digits 0-7) represented as 10x10 binary matrices. Each pixel is either active (1) or inactive (0). The primary challenges include:

- **Noise**: Distortions in the input images make it difficult to recover the original numeral patterns.
- **Limited Training Data**: Only 8 exemplars (one for each numeral) are available, requiring the model to generalize effectively.

### Network Design

#### Architecture

The Restricted Boltzmann Machine (RBM) used in this project has:

- **Visible Units**: 100 units, corresponding to the 10x10 pixel grid of the input images.
- **Hidden Units**: 150 units, which capture latent features and patterns in the data.

#### Training Methodology

The RBM is trained using the **Contrastive Divergence (CD)** algorithm, which approximates the gradient of the log-likelihood. Key training parameters include:

- **Activation Function**: Sigmoid
- **Number of Epochs**: 1250 epochs to ensure convergence
- **Learning Rate**: 0.1, with no decay applied
- **Batch Size**: 2, allowing efficient updates while leveraging small batches of data
- **Regularization**: no regularization 

#### Noise Handling and Reconstruction

The RBM learns robust feature representations in the hidden layer. During training, it associates noisy inputs with their corresponding clean outputs. When presented with a noisy image, the RBM reconstructs the original numeral by activating the most likely visible units based on the learned weights and biases.

## Computational Performance Analysis

### Reconstruction Error

The reconstruction error was monitored over **1250 epochs** for various noise levels. Below are the final reconstruction errors for each noise level:

| **Noise Level (%)** | **Reconstruction Error** |
|----------------------|--------------------------|
| 0.0                  | 0.3240                   |
| 0.1                  | 0.3481                   |
| 0.2                  | 0.3554                   |
| 0.3                  | 0.3608                   |
| 0.4                  | 0.3728                   |
| 0.5                  | 0.3791                   |
| 0.6                  | 0.3871                   |
| 0.7                  | 0.3914                   |
| 0.8                  | 0.3836                   |
| 0.9                  | 0.3838                   |

### Reconstruction Accuracy

The RBM's accuracy in reconstructing noisy images was evaluated using a **Hamming distance threshold**. The accuracy was **100%** for noise levels up to **0.3%**, but performance degraded at higher noise levels.

#### Accuracy Table

| **Noise Level (%)** | **Prediction Accuracy (%)** |
|----------------------|-----------------------------|
| 0.0                  | 100.0                       |
| 0.1                  | 100.0                       |
| 0.2                  | 100.0                       |
| 0.3                  | 100.0                       |
| 0.4                  | 87.5                        |
| 0.5                  | 75.0                        |
| 0.6                  | 75.0                        |
| 0.7                  | 12.5                        |
| 0.8                  | 50.0                        |
| 0.9                  | 25.0                        |

### Free Energy Gap

The free energy gap, which measures the RBM's ability to distinguish between noisy and clean data, was computed for each noise level:

| **Noise Level (%)** | **Free Energy Gap** |
|----------------------|---------------------|
| 0.0                  | 18.0785            |
| 0.1                  | 38.4666            |
| 0.2                  | 50.0708            |
| 0.3                  | 47.3601            |
| 0.4                  | 23.2003            |
| 0.5                  | 20.8249            |
| 0.6                  | 52.0559            |
| 0.7                  | 37.9430            |
| 0.8                  | 59.0178            |
| 0.9                  | 60.7365            |

### Visualization

Below are examples of original, noisy, and reconstructed images:

***INSERT PICS HERE***

1. **Original Images**: Clean numeral images without noise.
2. **Noisy Images**: Input images with varying noise levels.
3. **Reconstructed Images**: Outputs generated by the RBM.

## Strengths and Weaknesses

### Strengths

- **Handling Moderate Noise**: The RBM effectively reconstructed images with low noise levels (up to **0.3% noise**).
- **Feature Learning**: The hidden layer captured meaningful latent features, enabling robust reconstructions.
- **Fast Convergence**: The reconstruction error stabilized quickly during training.

### Weaknesses

- **High Noise Levels**: The RBM struggled to reconstruct images with higher noise levels (e.g., **0.4% and above**).
- **Limited Generalization**: With only **8 training exemplars**, the RBM's ability to generalize to unseen noisy inputs is limited.
- **Scalability**: The RBM's performance may degrade when applied to larger images or more complex datasets.

## Future Improvements

### Model Enhancements

- **Deeper Architectures**: Extending the RBM to a Deep Belief Network (DBN) or stacking multiple RBMs could improve its ability to handle high noise levels.
- **Better Noise Handling**: Incorporating noise-robust training techniques, such as dropout or denoising autoencoders, could enhance performance.

### Alternative Training Methods

- **Advanced Optimization**: Using algorithms like Adam or RMSprop may improve convergence and stability.
- **Regularization**: Adding L2 weight decay could mitigate overfitting.

### Potential Extensions

- **Other Datasets**: Applying the RBM to datasets with different types of images or tasks (e.g., handwritten text or natural images) could demonstrate its versatility.
- **Hybrid Models**: Combining the RBM with convolutional neural networks (CNNs) could leverage complementary strengths for improved performance.

## Installation

1. Clone the repository:
     ```bash
     git clone <repository-url>
     cd rbm-neural-net-project
     ```

2. Install the required dependencies:
        ```bash
        pip install -r requirements.txt
        ```

## Usage

1. Run the RBM script:
     ```bash
     python rbm.py
     ```

2. Optional: Use command-line arguments to customize the RBM's parameters:
        ```bash
        python rbm.py --n_visible 100 --n_hidden 150 --learning_rate 0.1 --n_epochs 1000 --batch_size 4
        ```

3. View the visualizations of the original, noisy, and reconstructed images, as well as the weight heatmap.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- **Authors**: Special thanks to the contributors who developed and tested the RBM implementation.
- **Inspiration**: This project was inspired by foundational research on Restricted Boltzmann Machines and their applications in image processing.
- **Libraries Used**: The project leverages Python libraries such as NumPy, Matplotlib, and Scikit-learn for numerical computations, visualizations, and machine learning utilities.
- **Community**: Gratitude to the open-source community for providing resources and support.
