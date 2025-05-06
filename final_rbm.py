import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("")
formatter = logging.Formatter('%(message)s')

class RestrictedBoltzmannMachine:
    """
    A class representing a Restricted Boltzmann Machine (RBM).
    """

    def __init__(self, n_visible, n_hidden, learning_rate=0.1, n_epochs=1000,
                 batch_size=10, decay_rate=0.99, activation='sigmoid',
                 regularization='normal', reg_lambda=0.001):
        """
        Initialize the RBM with the given parameters.
        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.decay_rate = decay_rate

        # Activation functions
        self.activation_fn = {
            'relu': self.relu,
            'leaky': self.leaky_relu,
            'sigmoid': self.sigmoid,
            'tanh': self.tanh,
        }
        self.activation = self.activation_fn[activation]

        # Regularization functions
        regularization_fn = {
            'normal': np.vectorize(lambda x: 0),
            'l1': np.vectorize(lambda x: np.abs(x)),
            'l2': np.vectorize(lambda x: x**2),
        }
        self.regularization = regularization_fn[regularization]
        self.reg_lambda = reg_lambda

        # Initialize weights and biases
        self.weights = np.random.uniform(-0.1, 0.1, (n_visible, n_hidden))
        self.visible_bias = np.zeros(n_visible)
        self.hidden_bias = np.zeros(n_hidden)

    # Activation functions
    def relu(self, x):
        return np.maximum(x, 0)

    def leaky_relu(self, x, a=0.01):
        return np.where(x > 0, x, a * x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def sample_probabilities(self, probs):
        """
        Sample binary states based on probabilities.
        """
        return (np.random.rand(*probs.shape) < probs).astype(np.float32)

    def contrastive_divergence(self, data):
        """
        Perform one step of contrastive divergence to update weights and biases.
        """
        # Positive phase
        pos_hidden_activations = np.dot(data, self.weights) + self.hidden_bias
        pos_hidden_probs = self.sigmoid(pos_hidden_activations)
        pos_hidden_states = self.sample_probabilities(pos_hidden_probs)
        pos_associations = np.dot(data.T, pos_hidden_probs)

        # Negative phase
        neg_visible_activations = np.dot(pos_hidden_states, self.weights.T) + self.visible_bias
        neg_visible_probs = self.sigmoid(neg_visible_activations)
        neg_hidden_activations = np.dot(neg_visible_probs, self.weights) + self.hidden_bias
        neg_hidden_probs = self.sigmoid(neg_hidden_activations)
        neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)

        # Update weights and biases
        self.weights += self.learning_rate * (
            (pos_associations - neg_associations) / data.shape[0] - self.reg_lambda * self.weights
        )
        self.visible_bias += self.learning_rate * np.mean(data - neg_visible_probs, axis=0)
        self.hidden_bias += self.learning_rate * np.mean(pos_hidden_probs - neg_hidden_probs, axis=0)

    def train(self, data):
        """
        Train the RBM using the provided data.
        """
        total_times = []
        errors = []

        for epoch in range(self.n_epochs):
            np.random.shuffle(data)
            start_time = time.time()

            for i in range(0, data.shape[0], self.batch_size):
                batch = data[i:i + self.batch_size]
                self.contrastive_divergence(batch)

            elapsed_time = time.time() - start_time
            error = np.mean((data - self.reconstruct(data)) ** 2)
            error += self.reg_lambda * np.sum(self.regularization(self.weights))

            total_times.append(elapsed_time)
            errors.append(error)

            # Apply learning rate decay
            self.learning_rate *= self.decay_rate

            if (epoch + 1) % 100 == 0:
                logger.info(f"Epoch {epoch + 1}/{self.n_epochs}, Reconstruction Error: {error:.4f}, Elapsed Time: {elapsed_time:.4f}")

        logger.info(f"Average Error: {np.mean(errors)} Average Epoch Time: {np.mean(total_times)}")

    def reconstruct(self, data):
        """
        Reconstruct visible units from hidden units.
        """
        hidden_probs = self.activation(np.dot(data, self.weights) + self.hidden_bias)
        visible_probs = self.activation(np.dot(hidden_probs, self.weights.T) + self.visible_bias)
        return visible_probs

    def visualize_weights(self):
        """
        Visualize weights as a heatmap.
        """
        plt.figure(figsize=(10, 8))
        plt.imshow(self.weights, cmap='viridis', aspect='auto')
        plt.colorbar(label="Weight Magnitude")
        plt.title("Weight Heatmap")
        plt.xlabel("Hidden Units")
        plt.ylabel("Visible Units")
        plt.show()


def generate_numerals():
    """
    Generate 10x10 binary arrays representing the digits 0-7.
    """
    numerals = [
        # Digit representations (0-7)
        # Each digit is a 10x10 binary array
            # Digit 0
            np.array([
            [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 1, 1, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [1, 1, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]),
            # Digit 1
            np.array([
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 1, 0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]),
            # Digit 2
            np.array([
            [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]),
            # Digit 3
            np.array([
            [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
            [1, 1, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            ]),
            # Digit 4
            np.array([
            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 1, 1, 0, 0, 0],
            [0, 1, 1, 0, 0, 1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]),
            # Digit 5
            np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [1, 1, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            ]),
            # Digit 6
            np.array([
            [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 1, 1, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 1, 1, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [1, 1, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            ]),
            # Digit 7
            np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]),
    ]

    # Flatten each 10x10 array into a 1D array of size 100
    flatten_numerals = [numeral.flatten() for numeral in numerals]
    return np.array(flatten_numerals)


def add_custom_noise(data, noise_level=0.2):
    """
    Add custom noise to the data by flipping bits with a given probability.
    """
    noise = np.random.binomial(1, noise_level, data.shape)
    return np.abs(data - noise)


def binarize_data(data, threshold=0.5):
    """
    Convert probabilities to binary values (0 or 1) based on a threshold.
    """
    data = np.array(data)  # Ensure input is a NumPy array
    return (data >= threshold).astype(np.float32)


def calculate_reconstruction_error(original, reconstructed):
    """
    Calculate the mean squared error between original and reconstructed data.
    """
    return np.mean((original - reconstructed) ** 2)


def hamming_distance(a, b):
    """
    Calculate the Hamming distance between two binary arrays.
    """
    return np.sum(a != b)

def get_accuracy(data, reconstructed_data, threshold=20):
    """
    Calculate the accuracy of the reconstructed data compared to the original data.
    """
    # Ensure inputs are NumPy arrays
    data = np.array(data)
    reconstructed_data = np.array(reconstructed_data)

    data = binarize_data(data)
    reconstructed_data = binarize_data(reconstructed_data)

    correct_reconstructions = 0
    for i in range(len(data)):
        distance = hamming_distance(data[i], reconstructed_data[i])
        if distance <= threshold:
            correct_reconstructions += 1

    accuracy = correct_reconstructions / len(data) * 100
    return accuracy

def calculate_var(data, reconstructed):
    """Calculate the mean squared error between original and reconstructed data."""
    return np.var((data - reconstructed) ** 2)

def calc_free_energy(v, W, vbias, hbias):
    """
    Compute the free energy of visible vector(s) v in an RBM.

    Parameters:
    - v: shape (batch_size, n_visible)
    - W: shape (n_visible, n_hidden)
    - vbias: shape (n_visible,)
    - hbias: shape (n_hidden,)

    Returns:
    - Free energy for each sample in the batch, shape (batch_size,)
    """
    linear_term = np.dot(v, vbias)
    hidden_term = np.dot(v, W) + hbias  # shape: (batch_size, n_hidden)
    hidden_term_logsum = np.sum(np.log1p(np.exp(hidden_term)), axis=1)  # log(1 + exp(x))
    return -linear_term - hidden_term_logsum

def calc_free_energy_gap(v, W, vbias, hbias):
    energies = calc_free_energy(v, W, vbias, hbias)
    return np.max(energies) - np.min(energies)

def generate_noisy_data(data, noise_level):
    """
    Generate noisy data and maintain correspondence with original indices.
    """
    indices = np.arange(len(data))
    noisy_data = add_custom_noise(data, noise_level=noise_level)
    noisy_data_with_indices = list(zip(noisy_data, indices))
    noisy_data_with_indices.sort(key=lambda x: x[1])  # Sort by indices to maintain order
    noisy_data = np.array([item[0] for item in noisy_data_with_indices])
    return noisy_data, indices


def train_rbm(noisy_data, opts):
    """
    Train the RBM with the given noisy data and options.
    """
    rbm = RestrictedBoltzmannMachine(
        n_visible=opts.n_visible,
        n_hidden=opts.n_hidden,
        learning_rate=opts.learning_rate,
        n_epochs=opts.n_epochs,
        batch_size=opts.batch_size,
        activation=opts.activation,
        regularization=opts.regularization,
        reg_lambda=opts.reg_lambda
    )
    rbm.train(noisy_data)
    return rbm


def reconstruct_data(rbm, noisy_data, indices):
    """
    Reconstruct data using the trained RBM and sort it back to the original order.
    """
    reconstructed_data = rbm.reconstruct(noisy_data)
    reconstructed_data = binarize_data(reconstructed_data)
    reconstructed_data_with_indices = list(zip(reconstructed_data, indices))
    reconstructed_data_with_indices.sort(key=lambda x: x[1])  # Sort by indices
    reconstructed_data = np.array([item[0] for item in reconstructed_data_with_indices])
    return reconstructed_data


def calculate_metrics(data, reconstructed_data, rbm):
    """
    Calculate reconstruction error, accuracy, variance, and free energy gap.
    """
    reconstruction_error = calculate_reconstruction_error(data, reconstructed_data)
    accuracy = get_accuracy(data, reconstructed_data)
    variance = calculate_var(data, reconstructed_data)
    free_energy_gap_lvl = calc_free_energy_gap(data, rbm.weights, rbm.visible_bias, rbm.hidden_bias)
    return reconstruction_error, accuracy, variance, free_energy_gap_lvl


def verify_correspondence(data, noisy_data, reconstructed_data):
    """
    Log and verify the correspondence between original, noisy, and reconstructed data.
    """
    for i in range(len(data)):
        logger.info(f"Original Data Index {i}:")
        logger.info(f"Original:\n{data[i].reshape(10, 10)}")
        logger.info(f"Noisy:\n{noisy_data[i].reshape(10, 10)}")
        logger.info(f"Reconstructed:\n{reconstructed_data[i].reshape(10, 10)}")


def plot_metrics(noise_levels, errors, accs, variances, free_energies):
    """
    Plot metrics across noise levels.
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    axs[0, 0].plot(noise_levels, errors, label="Reconstruction Error", color='blue')
    axs[0, 0].set_xlabel("Noise Level")
    axs[0, 0].set_ylabel("Reconstruction Error")
    axs[0, 0].set_title("Reconstruction Error vs Noise Level")
    axs[0, 0].grid(True)

    axs[0, 1].plot(noise_levels, accs, label="Accuracy", color='green')
    axs[0, 1].set_xlabel("Noise Level")
    axs[0, 1].set_ylabel("Accuracy (%)")
    axs[0, 1].set_title("Accuracy vs Noise Level")
    axs[0, 1].grid(True)

    axs[1, 0].plot(noise_levels, variances, label="Variance", color='orange')
    axs[1, 0].set_xlabel("Noise Level")
    axs[1, 0].set_ylabel("Variance")
    axs[1, 0].set_title("Variance vs Noise Level")
    axs[1, 0].grid(True)

    axs[1, 1].plot(noise_levels, free_energies, label="Free Energy Gap", color='red')
    axs[1, 1].set_xlabel("Noise Level")
    axs[1, 1].set_ylabel("Free Energy Gap")
    axs[1, 1].set_title("Free Energy Gap vs Noise Level")
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = ArgumentParser()

    # Command-line arguments
    parser.add_argument('--n_visible', type=int, default=100)
    parser.add_argument('--n_hidden', type=int, default=150)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--n_epochs', type=int, default=1250)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('-o', '--output', default=None, type=str, help="Output for the metrics")
    parser.add_argument('--activation', default='sigmoid', choices=['relu', 'leaky', 'sigmoid', 'tanh'])
    parser.add_argument('--regularization', default='l2', choices=['normal', 'l1', 'l2'])
    parser.add_argument('--reg_lambda', default=0.0005, type=float)

    opts = parser.parse_args()

    if opts.output is not None:
        fileHandler = logging.FileHandler(opts.output)
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)

    # Generate data and initialize variables
    data = generate_numerals()
    labels = np.arange(len(data))  # True labels for the numerals
    noise_levels = np.arange(0.0, 1.0, 0.1)

    # Initialize metrics storage
    errors = []
    prediction_accuracies = []
    reconstruction_variances = []
    free_energy_gaps = []

    # Process each noise level
    for noise_level in noise_levels:
        logger.info(f"Processing noise level: {noise_level}")

        # Generate noisy data
        noisy_data, indices = generate_noisy_data(data, noise_level)

        # Train RBM
        rbm = train_rbm(noisy_data, opts)

        # Reconstruct data
        reconstructed_data = reconstruct_data(rbm, noisy_data, indices)

        # Predict labels for reconstructed data using Hamming distance
        predicted_labels = []
        for reconstructed in reconstructed_data:
            distances = [hamming_distance(reconstructed, numeral) for numeral in data]
            predicted_labels.append(np.argmin(distances))

        # Sort reconstructed data and original data by predicted labels
        sorted_indices = np.argsort(predicted_labels)
        sorted_reconstructed_data = [reconstructed_data[i] for i in sorted_indices]
        sorted_data_by_prediction = [data[i] for i in sorted_indices]
        sorted_reconstructed_data_by_prediction = [reconstructed_data[i] for i in sorted_indices]
        sorted_labels_by_prediction = [predicted_labels[i] for i in sorted_indices]

        # Calculate metrics
        error, reconstruction_accuracy, variance, free_energy_gap = calculate_metrics(data, reconstructed_data, rbm)
        correct_predictions = sum(1 for i in range(len(labels)) if sorted_labels_by_prediction[i] == labels[i])
        prediction_accuracy = (correct_predictions / len(labels)) * 100

        # Store metrics
        errors.append(error)
        prediction_accuracies.append(prediction_accuracy)
        reconstruction_variances.append(variance)
        free_energy_gaps.append(free_energy_gap)

        # Log metrics
        logger.info(f"Noise Level: {noise_level}")
        logger.info(f"Reconstruction Error: {error}")
        logger.info(f"Prediction Accuracy: {prediction_accuracy}%")
        logger.info(f"Reconstruction Variance: {variance}")
        logger.info(f"Free Energy Gap: {free_energy_gap}")


        # Display original and reconstructed data in order of predicted labels
        fig, axs = plt.subplots(len(data), 2, figsize=(10, len(data) * 3))
        for i in range(len(data)):
            axs[i, 0].imshow(data[i].reshape(10, 10), cmap='gray')
            axs[i, 0].set_title(f"Original (Label: {i})")
            axs[i, 0].axis('off')

            axs[i, 1].imshow(sorted_reconstructed_data_by_prediction[i].reshape(10, 10), cmap='gray')
            axs[i, 1].set_title(f"Reconstructed (Predicted: {sorted_labels_by_prediction[i]})")
            axs[i, 1].axis('off')

        plt.tight_layout()
        plt.show()

    # Plot metrics
    plot_metrics(noise_levels, errors, prediction_accuracies, reconstruction_variances, free_energy_gaps)

    # Summarize results
    logger.info("Summary of Results:")
    logger.info(f"Average Reconstruction Error: {np.mean(errors):.4f}")
    logger.info(f"Average Reconstruction Accuracy: {np.mean(prediction_accuracies):.2f}%")
    logger.info(f"Average Reconstruction Variance: {np.mean(reconstruction_variances):.4f}")
    logger.info(f"Average Free Energy Gap: {np.mean(free_energy_gaps):.4f}")