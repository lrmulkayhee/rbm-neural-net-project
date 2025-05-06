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

def get_accuracy(data, reconstructed_data, threshold=30):
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
    parser.add_argument('--reg_lambda', default=0.001, type=float)

    opts = parser.parse_args()

    if opts.output is not None:
        fileHandler = logging.FileHandler(opts.output)
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)

    # Generate data and add noise
    data = generate_numerals()
    noisy_data = add_custom_noise(data, noise_level=0.01)

    # Initialize and train RBM
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
    rbm.visualize_weights()

    # Reconstruct and visualize data
    reconstructed_data = rbm.reconstruct(noisy_data)
    reconstructed_data = binarize_data(reconstructed_data)

    fig, axes = plt.subplots(3, 8, figsize=(15, 8))
    for i in range(8):
        axes[0, i].imshow(data[i].reshape(10, 10), cmap='gray')
        axes[0, i].set_title("Original")
        axes[0, i].axis('off')

        axes[1, i].imshow(noisy_data[i].reshape(10, 10), cmap='gray')
        axes[1, i].set_title("Noisy")
        axes[1, i].axis('off')

        axes[2, i].imshow(reconstructed_data[i].reshape(10, 10), cmap='gray')
        axes[2, i].set_title("Reconstructed")
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.show()

    # Calculate and display reconstruction error
    reconstruction_error = calculate_reconstruction_error(data, reconstructed_data)
    logger.info(f"Reconstruction Error: {reconstruction_error:.4f}")

    # Evaluate accuracy using Hamming distance
    accuracy = get_accuracy(data, reconstructed_data)
    logger.info(f"Accuracy: {accuracy:.2f}%")

    # Evaluate the variance
    variance = calculate_var(data, reconstructed_data)
    logger.info(f"Variance: {variance}")

    # Evaluate the free energy gap
    free_energy_gap = calc_free_energy_gap(data, rbm.weights,
                                          rbm.visible_bias, rbm.hidden_bias)

    for i in range(len(data)):
        error = np.mean((data[i] - reconstructed_data[i]) ** 2)
        logger.info(f"Reconstruction error for sample {i}: {error:.4f}")

    for i in range(len(data)):
        logger.info(f"Original index {i}:")
        logger.info(f"Original:\n{data[i].reshape(10, 10)}")
        logger.info(f"Noisy:\n{noisy_data[i].reshape(10, 10)}")
        logger.info(f"Reconstructed:\n{reconstructed_data[i].reshape(10, 10)}")

    # Calculate reconstruction error
reconstruction_error = calculate_reconstruction_error(data, reconstructed_data)
logger.info(f"Reconstruction Error: {reconstruction_error:.4f}")

# Evaluate accuracy using Hamming distance
threshold = 30
correct_reconstructions = sum(
    hamming_distance(data[i], reconstructed_data[i]) <= threshold for i in range(len(data))
)
accuracy = correct_reconstructions / len(data) * 100
logger.info(f"Accuracy: {accuracy:.2f}%")

# Assign predicted labels based on the closest original numeral
true_labels = np.arange(len(data))  # True labels for the original numerals (0-7)
predicted_labels = []

for reconstructed in reconstructed_data:
    distances = [hamming_distance(reconstructed, original) for original in data]
    predicted_labels.append(np.argmin(distances))  # Label of the closest numeral

# Calculate prediction accuracy
correct_predictions = sum(true_labels[i] == predicted_labels[i] for i in range(len(true_labels)))
prediction_accuracy = correct_predictions / len(true_labels) * 100
logger.info(f"Prediction Accuracy: {prediction_accuracy:.2f}%")

# Visualize results with true and predicted labels
fig, axes = plt.subplots(3, 8, figsize=(15, 8))
for i in range(8):
    axes[0, i].imshow(data[i].reshape(10, 10), cmap='gray')
    axes[0, i].set_title(f"Original: {true_labels[i]}")
    axes[0, i].axis('off')

    axes[1, i].imshow(noisy_data[i].reshape(10, 10), cmap='gray')
    axes[1, i].set_title("Noisy")
    axes[1, i].axis('off')

    axes[2, i].imshow(reconstructed_data[i].reshape(10, 10), cmap='gray')
    axes[2, i].set_title(f"Reconstructed: {predicted_labels[i]}")
    axes[2, i].axis('off')

plt.tight_layout()
plt.show()

