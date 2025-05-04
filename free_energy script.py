def free_energy(self, data):
    """Compute the free energy of the given data."""
    vb_term = -np.dot(data, self.visible_bias)
    hidden_term = -np.sum(np.log(1 + np.exp(np.dot(data, self.weights) + self.hidden_bias)), axis=1)
    return np.mean(vb_term + hidden_term)

def train(self, data, validation_data=None, monitor_interval=10):
    """Train the RBM using the provided data and monitor overfitting."""
    total_times = []
    errors = []
    training_subset = data[:min(100, len(data))]  # Use a fixed subset of training data for monitoring

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

        # Monitor overfitting every few epochs
        if validation_data is not None and (epoch + 1) % monitor_interval == 0:
            train_free_energy = self.free_energy(training_subset)
            val_free_energy = self.free_energy(validation_data)
            logger.info(f"Epoch {epoch + 1}/{self.n_epochs}, Train Free Energy: {train_free_energy:.4f}, "
                        f"Validation Free Energy: {val_free_energy:.4f}")
            logger.info(f"Free Energy Gap: {val_free_energy - train_free_energy:.4f}")

        # Log reconstruction error
        if (epoch + 1) % 100 == 0:
            logger.info(f"Epoch {epoch + 1}/{self.n_epochs}, Reconstruction Error: {error:.4f}, Elapsed Time: {elapsed_time:.4f}")

    logger.info(f"Average Error: {np.mean(errors)} Average Epoch Time: {np.mean(total_times)}")

# When calling the train method, pass the validation data as an additional argument:
    rbm.train(noisy_data, validation_data=validation_data, monitor_interval=10)