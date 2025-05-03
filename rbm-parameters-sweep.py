import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging
from itertools import product
from sklearn.metrics import mean_squared_error

NOISE_LEVEL = 0.1
THRESHOLD = 20

# Adding a random seed at the start to set a uniform starting point for the random number generator.
np.random.seed(46)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RBM Sweep")

# Utility functions
def add_custom_noise(data, noise_level=0.1):
    noise = np.random.binomial(1, noise_level, data.shape)
    return np.abs(data - noise)

def binarize_data(data, threshold=0.5):
    return (data >= threshold).astype(np.float32)

def hamming_distance(a, b):
    return np.sum(a != b)

def calculate_accuracy(original, reconstructed, threshold=THRESHOLD):
    correct = sum(hamming_distance(original[i], reconstructed[i]) <= threshold for i in range(len(original)))
    return correct / len(original) * 100

# RBM class
class RestrictedBoltzmannMachine:
    def __init__(self, config):
        self.n_visible = config["n_visible"]
        self.n_hidden = config["n_hidden"]
        self.learning_rate = config["learning_rate"]
        self.n_epochs = config["n_epochs"]
        self.batch_size = config["batch_size"]
        self.decay_rate = config["decay_rate"]
        self.lambda_reg = config["lambda_reg"]
        self.cd_k = config["cd_k"]

        self.weights = np.random.uniform(-0.1, 0.1, (self.n_visible, self.n_hidden))
        self.visible_bias = np.zeros(self.n_visible)
        self.hidden_bias = np.zeros(self.n_hidden)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sample_probabilities(self, probs):
        return (np.random.rand(*probs.shape) < probs).astype(np.float32)

    def contrastive_divergence(self, data):
        v0 = data
        h0_prob = self.sigmoid(np.dot(v0, self.weights) + self.hidden_bias)
        h0_sample = self.sample_probabilities(h0_prob)

        vk = v0.copy()
        hk = h0_sample.copy()

        for _ in range(self.cd_k):
            vk_prob = self.sigmoid(np.dot(hk, self.weights.T) + self.visible_bias)
            vk = self.sample_probabilities(vk_prob)
            hk_prob = self.sigmoid(np.dot(vk, self.weights) + self.hidden_bias)
            hk = self.sample_probabilities(hk_prob)

        pos_associations = np.dot(v0.T, h0_prob)
        neg_associations = np.dot(vk.T, hk_prob)

        self.weights += self.learning_rate * ((pos_associations - neg_associations) / data.shape[0] - self.lambda_reg * self.weights)
        self.visible_bias += self.learning_rate * np.mean(v0 - vk, axis=0)
        self.hidden_bias += self.learning_rate * np.mean(h0_prob - hk_prob, axis=0)

    def train(self, data):
        for epoch in range(self.n_epochs):
            np.random.shuffle(data)
            for i in range(0, data.shape[0], self.batch_size):
                batch = data[i:i + self.batch_size]
                self.contrastive_divergence(batch)
            self.learning_rate *= self.decay_rate

    def reconstruct(self, data):
        hidden_probs = self.sigmoid(np.dot(data, self.weights) + self.hidden_bias)
        visible_probs = self.sigmoid(np.dot(hidden_probs, self.weights.T) + self.visible_bias)
        return visible_probs

# Digits 0–7 as 10×10 binary arrays (flattened)
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

# Sweep parameters
sweep_params = {
    "learning_rate": [0.1],
    "cd_k": [3],
    "n_hidden": [250],
    "batch_size": [4],
    "n_epochs": [1000],
    "decay_rate": [1.0, 0.99, 0.95, 0.9, 0.8, 0.7, 0.6]  # Add decay rates to sweep
}

# Create parameter combinations
combinations = list(product(
    sweep_params["learning_rate"],
    sweep_params["cd_k"],
    sweep_params["n_hidden"],
    sweep_params["batch_size"],
    sweep_params["n_epochs"],
    sweep_params["decay_rate"]
))

# Sweep setup
results = []
base_config = {
    "n_visible": 100,
    "lambda_reg": 0.001
}

data = generate_numerals()
noisy_data = add_custom_noise(data, noise_level=NOISE_LEVEL)

# Repeat sweep 5 times
all_runs = []
NUM_RUNS = 5

for run in range(NUM_RUNS):
    logger.info(f"Starting run {run + 1}/{NUM_RUNS}")
    results = []

    for lr, k, hidden, bs, epochs, decay in combinations:
        config = base_config.copy()
        config.update({
            "learning_rate": lr,
            "cd_k": k,
            "n_hidden": hidden,
            "batch_size": bs,
            "n_epochs": epochs,
            "decay_rate":decay
        })
        rbm = RestrictedBoltzmannMachine(config)

        start_time = time.time()
        rbm.train(noisy_data)
        training_time = time.time() - start_time

        recon = binarize_data(rbm.reconstruct(noisy_data))
        acc = calculate_accuracy(data, recon)

        results.append({
            "run": run + 1,
            "learning_rate": lr,
            "cd_k": k,
            "n_hidden": hidden,
            "batch_size": bs,
            "n_epochs": epochs,
            "accuracy": acc,
            "decay_rate":decay,
            "training_time_sec": training_time
        })

    run_df = pd.DataFrame(results)
    all_runs.append(run_df)

# Combine all results
final_df = pd.concat(all_runs, ignore_index=True)
final_df.to_csv("sweep_results_multiple_runs.csv", index=False)


plt.figure(figsize=(10, 6))
for run in range(1, NUM_RUNS + 1):
    subset = final_df[final_df["run"] == run]
    plt.plot(subset["decay_rate"], subset["accuracy"], marker='o', label=f"Run {run}")
plt.title("Decay Rate vs Accuracy\n(CD-3, LR=0.1, Hidden=250, Batch=4, Epochs=1000)")
plt.xlabel("Decay Rate")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
for run in range(1, NUM_RUNS + 1):
    subset = final_df[final_df["run"] == run]
    plt.plot(subset["decay_rate"], subset["training_time_sec"], marker='o', label=f"Run {run}")
plt.title("Decay Rate vs Training Time\n(CD-3, LR=0.1, Hidden=250, Batch=4, Epochs=1000)")
plt.xlabel("Decay Rate")
plt.ylabel("Training Time (sec)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

