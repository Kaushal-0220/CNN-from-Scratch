import os
import sys

import matplotlib.pyplot as plt
import cupy as cp
import numpy as np

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from src.data import download_dataset, get_classes, load_dataset
from src.model import CNN


train_path, test_path = download_dataset()
classes = get_classes(train_path)

x_train, y_train = load_dataset(train_path, classes)
x_test, y_test = load_dataset(test_path, classes)


model = CNN()

epochs = 20
batch_size = 32

num_samples = len(x_train)

train_losses = []
train_accuracies = []


for epoch in range(epochs):

    indices = np.random.permutation(num_samples)

    x_train_shuffled = x_train[indices]
    y_train_shuffled = y_train[indices]

    epoch_loss = 0
    epoch_correct = 0

    for start in range(0, num_samples, batch_size):

        end = min(start + batch_size, num_samples)

        X_batch = cp.asarray(x_train_shuffled[start:end])
        Y_batch = cp.asarray(y_train_shuffled[start:end])

        loss, correct = model.train_step(X_batch, Y_batch)

        epoch_loss += loss * len(X_batch)
        epoch_correct += correct

    avg_loss = epoch_loss / num_samples
    avg_acc = epoch_correct / num_samples

    train_losses.append(float(avg_loss))
    train_accuracies.append(float(avg_acc))

    print(
        f"Epoch: {epoch + 1}/{epochs} | "
        f"Acc: {avg_acc:.4f} | "
        f"Loss: {avg_loss:.4f} | "
        f"LR: {model.optimizer.current_learning_rate:.6f}"
    )


os.makedirs("results", exist_ok=True)
plt.figure()
plt.plot(train_losses)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")

plt.savefig("results/training_loss.png")
plt.close()


plt.figure()
plt.plot(train_accuracies)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Accuracy")

plt.savefig("results/training_accuracy.png")
plt.close()

os.makedirs("results", exist_ok=True)
cp.savez("results/model_weights.npz",

    conv1_weights=model.conv1.weights,
    conv1_biases=model.conv1.biases,

    conv2_weights=model.conv2.weights,
    conv2_biases=model.conv2.biases,

    conv3_weights=model.conv3.weights,
    conv3_biases=model.conv3.biases,

    dense1_weights=model.dense1.weights,
    dense1_biases=model.dense1.biases,

    dense2_weights=model.dense2.weights,
    dense2_biases=model.dense2.biases
)