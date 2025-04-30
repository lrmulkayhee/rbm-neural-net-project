import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# --- RBM Class ---
class RestrictedBoltzmannMachine:
    def __init__(self, n_visible, n_hidden, learning_rate=0.1, n_epochs=1000, batch_size=10, decay_rate=0.99):
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
        for epoch in range(self.n_epochs):
            np.random.shuffle(data)
            for i in range(0, data.shape[0], self.batch_size):
                batch = data[i:i + self.batch_size]
                self.contrastive_divergence(batch)
            self.learning_rate *= self.decay_rate
            if (epoch + 1) % 100 == 0:
                error = np.mean((data - self.reconstruct(data)) ** 2)
                print(f"Epoch {epoch + 1}/{self.n_epochs}, Reconstruction Error: {error:.4f}")

        plt.plot(error)
        plt.title("Reconstruction Error Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Reconstruction Error")
        plt.grid(True)
        plt.show()

    def reconstruct(self, data):
        hidden_probs = self.sigmoid(np.dot(data, self.weights) + self.hidden_bias)
        visible_probs = self.sigmoid(np.dot(hidden_probs, self.weights.T) + self.visible_bias)
        return visible_probs

# --- Helper Functions ---
def load_images_from_folder(folder_path, size=(28, 28), threshold=0.5):
    images = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path).convert('L')
            img = img.resize(size)
            img_arr = np.asarray(img) / 255.0
            binary_img = (img_arr >= threshold).astype(np.float32)
            images.append(binary_img.flatten())
    return np.array(images)

def binarize_data(data, threshold=0.5):
    return (data >= threshold).astype(np.float32)

def calculate_reconstruction_error(original, reconstructed):
    return np.mean((original - reconstructed) ** 2)

# --- Main ---
if __name__ == "__main__":
    folder_path = "images"  # Make sure this folder contains valid images
    training_data = load_images_from_folder(folder_path)
    print(f"Loaded {training_data.shape[0]} images with input size {training_data.shape[1]}")

    # Initialize RBM (28x28 = 784 inputs)
    rbm = RestrictedBoltzmannMachine(n_visible=784, n_hidden=250, learning_rate=0.1, n_epochs=1000, batch_size=4)
    rbm.train(training_data)

    num_images = training_data.shape[0]
    fig, axes = plt.subplots(2, num_images, figsize=(3 * num_images, 6))

    for i in range(num_images):
        sample = training_data[i].reshape(1, -1)
        reconstructed = rbm.reconstruct(sample)
        reconstructed_bin = binarize_data(reconstructed)

        axes[0, i].imshow(sample.reshape(28, 28), cmap='gray', interpolation='nearest')
        axes[0, i].set_title(f"Original {i+1}")
        axes[0, i].axis('off')

        axes[1, i].imshow(reconstructed_bin.reshape(28, 28), cmap='gray', interpolation='nearest')
        axes[1, i].set_title(f"Reconstructed {i+1}")
        axes[1, i].axis('off')

        plt.tight_layout()
        plt.show()

    # Evaluate
    error = calculate_reconstruction_error(sample, reconstructed_bin)
    print(f"Reconstruction Error: {error:.4f}")
