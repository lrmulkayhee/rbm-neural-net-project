import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging
from itertools import product
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Configuration
np.random.seed(48)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RBM MNIST Sweep")

# Utility Functions
def add_custom_noise(data, noise_level=0.1):
    noise = np.random.binomial(1, noise_level, data.shape)
    return np.abs(data - noise)

def load_mnist_data(n_samples=1000, binarize=True, threshold=0.5):
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist.data / 255.0
    if binarize:
        X = (X >= threshold).astype(np.float32)
    X_train, _ = train_test_split(X, train_size=n_samples, random_state=42)
    return X_train

def hamming_distance(a, b):
    return np.sum(a != b)

def calculate_accuracy(original, reconstructed, threshold):
    reconstructed_bin = (reconstructed >= 0.5).astype(np.float32)
    correct = sum(hamming_distance(original[i], reconstructed_bin[i]) <= threshold for i in range(len(original)))
    return correct / len(original) * 100

def reconstruction_error(original, reconstructed):
    return np.mean((original - reconstructed) ** 2)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

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

    def sample_probabilities(self, probs):
        return (np.random.rand(*probs.shape) < probs).astype(np.float32)

    def contrastive_divergence(self, data):
        v0 = data
        h0_prob = sigmoid(np.dot(v0, self.weights) + self.hidden_bias)
        h0_sample = self.sample_probabilities(h0_prob)
        vk = v0.copy()
        hk = h0_sample.copy()
        for _ in range(self.cd_k):
            vk_prob = sigmoid(np.dot(hk, self.weights.T) + self.visible_bias)
            vk = self.sample_probabilities(vk_prob)
            hk_prob = sigmoid(np.dot(vk, self.weights) + self.hidden_bias)
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
        hidden_probs = sigmoid(np.dot(data, self.weights) + self.hidden_bias)
        visible_probs = sigmoid(np.dot(hidden_probs, self.weights.T) + self.visible_bias)
        return visible_probs

# Sweep Parameters
sweep_params = {
    "learning_rate": [0.03],
    "cd_k": [3],
    "n_hidden": [256],
    "batch_size": [32],
    "n_epochs": [100, 150, 200, 300, 400, 500],
    "decay_rate": [0.95],
    "threshold": [50]
}

sweep_combinations = list(product(*sweep_params.values()))
param_keys = list(sweep_params.keys())
base_config = {
    "n_visible": 784,
    "lambda_reg": 0.0005
}

data = load_mnist_data(n_samples=1000)
noisy_data = add_custom_noise(data, noise_level=0.05)

# Run Sweep
NUM_RUNS = 5
all_runs = []

for run in range(NUM_RUNS):
    logger.info(f"Starting run {run + 1}/{NUM_RUNS}")
    results = []

    for values in sweep_combinations:
        config = base_config.copy()
        sweep_dict = dict(zip(param_keys, values))
        config.update({k: sweep_dict[k] for k in sweep_dict if k != "threshold"})

        rbm = RestrictedBoltzmannMachine(config)
        start_time = time.time()
        rbm.train(noisy_data.copy())
        training_time = time.time() - start_time

        recon = rbm.reconstruct(noisy_data)
        acc = calculate_accuracy(data, recon, sweep_dict["threshold"])
        err = reconstruction_error(data, recon)

        result = sweep_dict.copy()
        result.update({
            "run": run + 1,
            "accuracy": acc,
            "reconstruction_error": err,
            "training_time_sec": training_time
        })
        results.append(result)

    run_df = pd.DataFrame(results)
    all_runs.append(run_df)

# Combine and plot
final_df = pd.concat(all_runs, ignore_index=True)

plt.figure(figsize=(10, 6))
for run in range(1, NUM_RUNS + 1):
    subset = final_df[final_df["run"] == run]
    plt.plot(subset["n_epochs"], subset["reconstruction_error"], marker='o', label=f"Run {run}")
plt.title("Epochs vs Reconstruction Error\n(MNIST RBM)")
plt.xlabel("Epochs")
plt.ylabel("Reconstruction Error")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
for run in range(1, NUM_RUNS + 1):
    subset = final_df[final_df["run"] == run]
    plt.plot(subset["n_epochs"], subset["accuracy"], marker='o', label=f"Run {run}")
plt.title("Epochs vs Accuracy\n(MNIST RBM)")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
for run in range(1, NUM_RUNS + 1):
    subset = final_df[final_df["run"] == run]
    plt.plot(subset["n_epochs"], subset["training_time_sec"], marker='o', label=f"Run {run}")
plt.title("Epochs vs Training Time\n(MNIST RBM)")
plt.xlabel("Epochs")
plt.ylabel("Training Time (sec)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
