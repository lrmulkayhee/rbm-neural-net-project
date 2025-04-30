import time
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, Frame, LabelFrame

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
        start_time = time.time()
        for epoch in range(self.n_epochs):
            np.random.shuffle(data)
            for i in range(0, data.shape[0], self.batch_size):
                batch = data[i:i + self.batch_size]
                self.contrastive_divergence(batch)
            error = np.mean((data - self.reconstruct(data)) ** 2)
            errors.append(error)
        elapsed_time = time.time() - start_time
        return errors, elapsed_time

    def reconstruct(self, data):
        hidden_probs = self.sigmoid(np.dot(data, self.weights) + self.hidden_bias)
        visible_probs = self.sigmoid(np.dot(hidden_probs, self.weights.T) + self.visible_bias)
        return visible_probs

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
    noise = np.random.binomial(1, noise_level, data.shape)
    return np.abs(data - noise)

def binarize_data(data, threshold=0.5):
    return (data >= threshold).astype(np.float32)

def hamming_distance(a, b):
    return np.sum(a != b)

def visualize_original_noisy_reconstructed(data, noisy_data, reconstructed_data):
    fig, axes = plt.subplots(3, 8, figsize=(15, 6))
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

def run_rbm(hidden_units, epochs, batch_size, noise_level, learning_rate, final_error_label, accuracy_label, time_label):
    data = generate_numerals()
    noisy_data = add_custom_noise(data, noise_level=noise_level)

    rbm = RestrictedBoltzmannMachine(
        n_visible=100,
        n_hidden=hidden_units,
        learning_rate=learning_rate,
        n_epochs=epochs,
        batch_size=batch_size
    )

    start_time = time.time()
    rbm.train(noisy_data)
    elapsed_time = time.time() - start_time

    reconstructed = binarize_data(rbm.reconstruct(noisy_data))
    error = np.mean((data - reconstructed) ** 2)

    # Calculate Accuracy
    threshold = 30
    correct = sum(np.sum(data[i] != reconstructed[i]) <= threshold for i in range(len(data)))
    accuracy = (correct / len(data)) * 100

    visualize_original_noisy_reconstructed(data, noisy_data, reconstructed)

    final_error_label.config(text=f"Final Reconstruction Error: {error:.5f}")
    accuracy_label.config(text=f"Accuracy: {accuracy:.2f}%")
    time_label.config(text=f"Time per Run: {elapsed_time:.2f} s")

root = tk.Tk()
root.title("RBM Denoising with Adjustable Parameters")

controls_frame = Frame(root)
controls_frame.grid(row=0, column=0, padx=10, pady=10)

result_frame = LabelFrame(root, text="Results", padx=10, pady=10)
result_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

# Inputs with longer sliders
hidden_units_var = tk.IntVar(value=100)
epochs_var = tk.IntVar(value=500)
batch_size_var = tk.IntVar(value=10)
noise_level_var = tk.DoubleVar(value=0.1)
learning_rate_var = tk.DoubleVar(value=0.1)

tk.Label(controls_frame, text='Hidden Units').grid(row=0, column=0, sticky='w')
hidden_units_slider = tk.Scale(controls_frame, from_=10, to=300, orient='horizontal', variable=hidden_units_var, length=300)
hidden_units_slider.grid(row=0, column=1)

tk.Label(controls_frame, text='Epochs').grid(row=1, column=0, sticky='w')
epochs_slider = tk.Scale(controls_frame, from_=100, to=2000, orient='horizontal', variable=epochs_var, length=300)
epochs_slider.grid(row=1, column=1)

tk.Label(controls_frame, text='Batch Size').grid(row=2, column=0, sticky='w')
batch_size_slider = tk.Scale(controls_frame, from_=1, to=32, orient='horizontal', variable=batch_size_var, length=300)
batch_size_slider.grid(row=2, column=1)

tk.Label(controls_frame, text='Noise Level').grid(row=3, column=0, sticky='w')
noise_level_slider = tk.Scale(controls_frame, from_=0.0, to=0.5, resolution=0.01, orient='horizontal', variable=noise_level_var, length=300)
noise_level_slider.grid(row=3, column=1)

tk.Label(controls_frame, text='Learning Rate').grid(row=4, column=0, sticky='w')
learning_rate_slider = tk.Scale(controls_frame, from_=0.001, to=1.0, resolution=0.01, orient='horizontal', variable=learning_rate_var, length=300)
learning_rate_slider.grid(row=4, column=1)

# Results Labels
final_error_label = tk.Label(result_frame, text="Final Reconstruction Error: ")
final_error_label.pack(anchor='w')

accuracy_label = tk.Label(result_frame, text="Accuracy: ")
accuracy_label.pack(anchor='w')

time_label = tk.Label(result_frame, text="Time per Run: ")
time_label.pack(anchor='w')

def run_button_clicked():
    run_rbm(
        hidden_units_var.get(),
        epochs_var.get(),
        batch_size_var.get(),
        noise_level_var.get(),
        learning_rate_var.get(),
        final_error_label,
        accuracy_label,
        time_label
    )

run_button = tk.Button(root, text='Run RBM', command=run_button_clicked)
run_button.grid(row=2, column=0, pady=10)

root.mainloop()
