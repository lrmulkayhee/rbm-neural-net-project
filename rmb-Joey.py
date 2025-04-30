import time
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from argparse import ArgumentParser

# Adding a random seed at the start to set a uniform starting point for the random number generator.
np.random.seed(45)

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("")
formatter = logging.Formatter('%(message)s')

# --- RBM Class ---
class RestrictedBoltzmannMachine:
    def __init__(self, n_visible, n_hidden, learning_rate=0.1, n_epochs=500, batch_size=10, decay_rate=0.99):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.decay_rate = decay_rate
        self.weights = np.random.uniform(-0.1, 0.1, (n_visible, n_hidden))
        self.visible_bias = np.zeros(n_visible)
        self.hidden_bias = np.zeros(n_hidden)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sample_probabilities(self, probs):
        return (np.random.rand(*probs.shape) < probs).astype(np.float32)

    def contrastive_divergence(self, data):
        pos_hidden_probs = self.sigmoid(np.dot(data, self.weights) + self.hidden_bias)
        pos_hidden_states = self.sample_probabilities(pos_hidden_probs)
        pos_associations = np.dot(data.T, pos_hidden_probs)

        neg_visible_probs = self.sigmoid(np.dot(pos_hidden_states, self.weights.T) + self.visible_bias)
        neg_hidden_probs = self.sigmoid(np.dot(neg_visible_probs, self.weights) + self.hidden_bias)
        neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)

        self.weights += self.learning_rate * (pos_associations - neg_associations) / data.shape[0]
        self.visible_bias += self.learning_rate * np.mean(data - neg_visible_probs, axis=0)
        self.hidden_bias += self.learning_rate * np.mean(pos_hidden_probs - neg_hidden_probs, axis=0)

    def train(self, data):
        errors = []
        for epoch in range(self.n_epochs):
            np.random.shuffle(data)
            for i in range(0, data.shape[0], self.batch_size):
                batch = data[i:i + self.batch_size]
                self.contrastive_divergence(batch)
            error = np.mean((data - self.reconstruct(data)) ** 2)
            errors.append(error)
        return errors

    def reconstruct(self, data):
        hidden_probs = self.sigmoid(np.dot(data, self.weights) + self.hidden_bias)
        visible_probs = self.sigmoid(np.dot(hidden_probs, self.weights.T) + self.visible_bias)
        return visible_probs

# --- Helper Functions ---
def generate_numerals():
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
    flattened_numerals = [numeral.flatten() for numeral in numerals]
    return np.array(flattened_numerals)

def add_custom_noise(data, noise_level=0.2):
    noise = np.random.binomial(1, noise_level, data.shape)
    noisy_data = np.abs(data - noise)
    return noisy_data

def binarize_data(data, threshold=0.5):
    return (data >= threshold).astype(np.float32)

def calculate_reconstruction_error(original, reconstructed):
    return np.mean((original - reconstructed) ** 2)

def hamming_distance(a, b):
    return np.sum(a != b)

# --- Main Script ---
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-o', '--output', default='sweep_results.csv', type=str, help="CSV Output for the metrics")
    opts = parser.parse_args()

    data = generate_numerals()
    logger.info(f"Original data shape: {data.shape}")

    n_hidden_list = [50, 75, 100, 150, 200, 250, 300]
    batch_size_list = [2, 4, 8, 10]
    noise_level_list = [0.01, 0.05, 0.1, 0.3]

    results = []

    for n_hidden in n_hidden_list:
        for batch_size in batch_size_list:
            for noise_level in noise_level_list:
                logger.info(f"Running: Hidden={n_hidden}, Batch={batch_size}, Noise={noise_level}")
                noisy_data = add_custom_noise(data, noise_level=noise_level)
                rbm = RestrictedBoltzmannMachine(
                    n_visible=100,
                    n_hidden=n_hidden,
                    learning_rate=0.1,
                    n_epochs=500,
                    batch_size=batch_size
                )
                start_time = time.time()
                rbm.train(noisy_data)
                elapsed_time = time.time() - start_time

                reconstructed = binarize_data(rbm.reconstruct(noisy_data))
                error = calculate_reconstruction_error(data, reconstructed)

                threshold = 30
                correct = sum(hamming_distance(data[i], reconstructed[i]) <= threshold for i in range(len(data)))
                accuracy = correct / len(data) * 100

                results.append({
                    'n_hidden': n_hidden,
                    'batch_size': batch_size,
                    'noise_level': noise_level,
                    'final_error': round(error, 5),
                    'accuracy(%)': round(accuracy, 2),
                    'time_per_run(s)': round(elapsed_time, 2)
                })

                # Visualization of Original, Noisy, and Reconstructed Data for the first parameter set
                if n_hidden == 100 and batch_size == 4 and noise_level == 0.01:
                    fig, axes = plt.subplots(3, 8, figsize=(15, 6))
                    for i in range(8):
                        axes[0, i].imshow(data[i].reshape(10, 10), cmap='gray')
                        axes[0, i].set_title("Original")
                        axes[0, i].axis('off')

                        axes[1, i].imshow(noisy_data[i].reshape(10, 10), cmap='gray')
                        axes[1, i].set_title("Noisy")
                        axes[1, i].axis('off')

                        axes[2, i].imshow(reconstructed[i].reshape(10, 10), cmap='gray')
                        axes[2, i].set_title("Reconstructed")
                        axes[2, i].axis('off')

                    plt.tight_layout()
                    plt.savefig("sample_reconstructions.png")
                    plt.show()

    df = pd.DataFrame(results)
    df.to_csv(opts.output, index=False)
    print(df)

    noise_levels = sorted(df['noise_level'].unique())
    batch_sizes = sorted(df['batch_size'].unique())

    for noise in noise_levels:
        plt.figure(figsize=(10, 6))
        for batch_size in batch_sizes:
            subset = df[(df['batch_size'] == batch_size) & (df['noise_level'] == noise)]
            plt.plot(subset['n_hidden'], subset['final_error'], marker='o', label=f'Batch {batch_size}')
            
            # Annotate points with error value
            for _, row in subset.iterrows():
                plt.annotate(f"{row['final_error']:.3f}", 
                            (row['n_hidden'], row['final_error']), 
                            textcoords="offset points", xytext=(0, 5), 
                            ha='center', fontsize=8)

        plt.title(f"Final Reconstruction Error vs Hidden Units (Noise = {noise:.2f})")
        plt.xlabel("Hidden Units")
        plt.ylabel("Final Error")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"final_error_noise_{noise:.2f}.png")
        plt.show()

    # --- Unified Accuracy and Time Plot ---
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Accuracy Plot
    for noise in noise_levels:
        plt.figure(figsize=(10, 6))
        for batch_size in batch_sizes:
            subset = df[(df['batch_size'] == batch_size) & (df['noise_level'] == noise)]
            plt.plot(subset['n_hidden'], subset['accuracy(%)'], marker='o', label=f'Batch {batch_size}')
            
            # Annotate each point with accuracy value
            for _, row in subset.iterrows():
                plt.annotate(f"{row['accuracy(%)']:.1f}%", 
                            (row['n_hidden'], row['accuracy(%)']), 
                            textcoords="offset points", xytext=(0, 5), 
                            ha='center', fontsize=8)

        plt.title(f"Accuracy vs Hidden Units (Noise = {noise:.2f})")
        plt.xlabel("Hidden Units")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"accuracy_noise_{noise:.2f}.png")
        plt.show()


    # Time Plot
    for noise in noise_levels:
        plt.figure(figsize=(10, 6))
        for batch_size in batch_sizes:
            subset = df[(df['batch_size'] == batch_size) & (df['noise_level'] == noise)]
            plt.plot(subset['n_hidden'], subset['time_per_run(s)'], marker='o', label=f'Batch {batch_size}')
            
            # Annotate each point with time value
            for _, row in subset.iterrows():
                plt.annotate(f"{row['time_per_run(s)']:.2f}s", 
                            (row['n_hidden'], row['time_per_run(s)']), 
                            textcoords="offset points", xytext=(0, 5), 
                            ha='center', fontsize=8)

        plt.title(f"Training Time vs Hidden Units (Noise = {noise:.2f})")
        plt.xlabel("Hidden Units")
        plt.ylabel("Training Time (s)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"time_noise_{noise:.2f}.png")
        plt.show()

    # --- Training Time vs Epochs Analysis ---
    epoch_values = [100, 250, 500, 750, 1000]
    fixed_hidden = 100
    fixed_batch_size = 8
    fixed_noise = 0.01
    training_times = []

    for n_epochs in epoch_values:
        noisy_data = add_custom_noise(data, noise_level=fixed_noise)
        rbm = RestrictedBoltzmannMachine(
            n_visible=100,
            n_hidden=fixed_hidden,
            learning_rate=0.1,
            n_epochs=n_epochs,
            batch_size=fixed_batch_size
        )
        start_time = time.time()
        rbm.train(noisy_data)
        elapsed = time.time() - start_time
        training_times.append(elapsed)

    # --- Plot Training Time vs Epochs ---
    plt.figure(figsize=(8, 6))
    plt.plot(epoch_values, training_times, marker='o', color='purple')
    for i, t in enumerate(training_times):
        plt.annotate(f"{t:.2f}s", (epoch_values[i], training_times[i]), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8)

    plt.title("Training Time vs Number of Epochs\n(Hidden Units = 100, Batch = 8, Noise = 0.1)")
    plt.xlabel("Epochs")
    plt.ylabel("Training Time (s)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_time_vs_epochs.png")
    plt.show()

