import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("")
formatter = logging.Formatter('%(message)s')

class RestrictedBoltzmannMachine:
    def __init__(self, n_visible, n_hidden, activation, learning_rate=0.1, n_epochs=1000, batch_size=10, decay_rate=0.99):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.decay_rate = decay_rate
        self.activation = activation

        # Initialize weights and biases
        self.weights = np.random.uniform(-0.1, 0.1, (n_visible, n_hidden))
        self.visible_bias = np.zeros(n_visible)
        self.hidden_bias = np.zeros(n_hidden)

    def activate(self, x):
        """Apply the selected activation function."""
        if self.activation == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif self.activation == "relu":
            return np.maximum(0, x)
        elif self.activation == "leaky_relu":
            return np.where(x > 0, x, 0.01 * x)
        else:
            raise ValueError("Unsupported activation function")
        
    def sample_probabilities(self, probs):
        """Sample binary states based on probabilities."""
        return (np.random.rand(*probs.shape) < probs).astype(np.float32)

    def contrastive_divergence(self, data):
        """Perform one step of contrastive divergence."""
        # Positive phase
        pos_hidden_probs = self.sigmoid(np.dot(data, self.weights) + self.hidden_bias)
        pos_hidden_states = self.sample_probabilities(pos_hidden_probs)
        pos_associations = np.dot(data.T, pos_hidden_probs)

        # Negative phase
        neg_visible_probs = self.sigmoid(np.dot(pos_hidden_states, self.weights.T) + self.visible_bias)
        neg_hidden_probs = self.sigmoid(np.dot(neg_visible_probs, self.weights) + self.hidden_bias)
        neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)

        # Update weights and biases
        self.weights += self.learning_rate * (pos_associations - neg_associations) / data.shape[0]
        self.visible_bias += self.learning_rate * np.mean(data - neg_visible_probs, axis=0)
        self.hidden_bias += self.learning_rate * np.mean(pos_hidden_probs - neg_hidden_probs, axis=0)

    def train(self, data):
        """Train the RBM using the provided data."""
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

            total_times.append(elapsed_time)
            errors.append(error)

            # Apply learning rate decay
            self.learning_rate *= self.decay_rate

            # Calculate reconstruction error
            if (epoch + 1) % 100 == 0:
                logger.info(f"Epoch {epoch + 1}/{self.n_epochs}, Reconstruction Error: {error:.4f}, Elapsed Time: {elapsed_time:.4f}")

        logger.info(f"Average Error: {np.mean(errors)} Average Epoch Time: {np.mean(total_times)}")

    def reconstruct(self, data):
        """Reconstruct visible units from hidden units."""
        hidden_probs = self.sigmoid(np.dot(data, self.weights) + self.hidden_bias)
        visible_probs = self.sigmoid(np.dot(hidden_probs, self.weights.T) + self.visible_bias)
        return visible_probs

    def visualize_weights(self):
        """Visualize weights as a heatmap."""
        plt.figure(figsize=(10, 8))
        plt.imshow(self.weights, cmap='viridis', aspect='auto')
        plt.colorbar(label="Weight Magnitude")
        plt.title("Weight Heatmap")
        plt.xlabel("Hidden Units")
        plt.ylabel("Visible Units")
        plt.show()

def generate_numerals():
    """Generate 10x10 binary arrays representing the digits 0-7."""
    numerals = [
        # Digit 0
        np.array([
            [0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]),
        # Digit 1
        np.array([
            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]),
        # Digit 2
        np.array([
            [0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]),
        # Digit 3
        np.array([
            [0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]),
        # Digit 4
        np.array([
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]),
        # Digit 5
        np.array([
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [1, 1, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]),
        # Digit 6
        np.array([
            [0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]),
        # Digit 7
        np.array([
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]),
    ]

    # Flatten each 10x10 array into a 1D array of size 100
    flattened_numerals = [numeral.flatten() for numeral in numerals]
    return np.array(flattened_numerals)

def add_custom_noise(data, noise_level=0.2):
    noisy_data = data.copy()
    noise = np.random.binomial(1, noise_level, data.shape)
    noisy_data = np.abs(noisy_data - noise)  # Flip bits based on noise
    return noisy_data

# Function to binarize the reconstructed data to ensure black and white output
def binarize_data(data, threshold=0.5):
    """Convert probabilities to binary values (0 or 1) based on a threshold."""
    return (data >= threshold).astype(np.float32)

# Function to calculate reconstruction error
def calculate_reconstruction_error(original, reconstructed):
    """Calculate the mean squared error between original and reconstructed data."""
    return np.mean((original - reconstructed) ** 2)

# Function to calculate Hamming distance
def hamming_distance(a, b):
    """Calculate the Hamming distance between two binary arrays."""
    return np.sum(a != b)

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--n_visible', type=int, default=100)
    parser.add_argument('--n_hidden', type=int, default=150)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('-o', '--output', default=None, type=str, help="Output for the metrics")

    opts = parser.parse_args()

    if opts.output is not None:
        fileHandler = logging.FileHandler(opts.output)
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)

    data = generate_numerals()
    logger.info(f"Shape of data: {data.shape}")  # Should print (8, 100)

    noisy_data = add_custom_noise(data, noise_level=0.01)  # Reduced noise level
    logger.info(f"Shape of noisy data: {noisy_data.shape}")  # Should also be (8, 100)

    activations = ["sigmoid", "relu", "leaky_relu"]
    results = []

    for activation in activations:
        logger.info(f"Testing activation function: {activation}")
        rbm = RestrictedBoltzmannMachine(
            n_visible=opts.n_visible,
            n_hidden=opts.n_hidden,
            learning_rate=opts.learning_rate,
            n_epochs=opts.n_epochs,
            batch_size=opts.batch_size,
            activation=activation
        )
        rbm.train(noisy_data)
        reconstructed_data = rbm.reconstruct(noisy_data)
        reconstructed_data = binarize_data(reconstructed_data)

        # Calculate reconstruction error
        reconstruction_error = calculate_reconstruction_error(data, reconstructed_data)

        # Evaluate accuracy using Hamming distance
        threshold = 30
        correct_reconstructions = 0
        for i in range(len(data)):
            distance = hamming_distance(data[i], reconstructed_data[i])
            if distance <= threshold:
                correct_reconstructions += 1

        accuracy = correct_reconstructions / len(data) * 100
        results.append((activation, reconstruction_error, accuracy))

    # Display results
    logger.info("Activation Function Results:")
    logger.info(f"{'Activation':<15} {'Reconstruction Error':<20} {'Accuracy (%)':<15}")
    for activation, error, acc in results:
        logger.info(f"{activation:<15} {error:<20.4f} {acc:<15.2f}")
