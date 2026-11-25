from src import *
import os
import numpy as np
import matplotlib.pyplot as plt

train_path, test_path = download_dataset()
classes = get_classes(train_path)
x_test, y_test = load_dataset(test_path, classes)


model = CNN(num_classes=len(classes))


weights = cp.load("results/model_weights.npz")

model.conv1.weights = weights["conv1_weights"]
model.conv1.biases = weights["conv1_biases"]

model.conv2.weights = weights["conv2_weights"]
model.conv2.biases = weights["conv2_biases"]

model.conv3.weights = weights["conv3_weights"]
model.conv3.biases = weights["conv3_biases"]

model.dense1.weights = weights["dense1_weights"]
model.dense1.biases = weights["dense1_biases"]

model.dense2.weights = weights["dense2_weights"]
model.dense2.biases = weights["dense2_biases"]

print("-" * 50 + "\n")
print("TESTING MODEL")


batch_size = 32
total_correct = 0
total_samples = len(x_test)

for start in range(0, total_samples, batch_size):

    end = min(start + batch_size, total_samples)

    X_batch = cp.asarray(x_test[start:end])
    y_batch = cp.asarray(y_test[start:end])

    predictions = model.predict(X_batch)

    total_correct += int(
        cp.asnumpy(cp.sum(predictions == y_batch))
    )

test_accuracy = (total_correct / total_samples) * 100

print(f"\nTesting Accuracy: {test_accuracy:.2f}%")
print(f"Correct predictions: {total_correct}/{total_samples}")


results_dir = "results"
os.makedirs(results_dir, exist_ok=True)
evaluation_file = os.path.join(results_dir, "evaluation_results.txt")

with open(evaluation_file, "w", encoding="utf-8") as f:
    f.write("CNN FROM SCRATCH - EVALUATION RESULTS\n")
    f.write("-" * 50 + "\n")
    f.write(f"Testing images: {total_samples}\n")
    f.write(f"Correct predictions: {total_correct}\n")
    f.write(f"Testing accuracy: {test_accuracy:.2f}%\n")
    f.write("\nClasses:\n")

    for class_name in classes:
        f.write(f"- {class_name}\n")

print(f"Evaluation results saved to: {evaluation_file}")


num_examples = min(8, len(x_test))

rng = np.random.default_rng()
indices = rng.choice(len(x_test), size=num_examples, replace=False)

x_samples = np.stack([x_test[int(j)] for j in indices])
y_samples = np.asarray([int(y_test[int(j)]) for j in indices])

predictions = model.predict(cp.asarray(x_samples))
predictions = cp.asnumpy(predictions)

print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)

for i, idx in enumerate(indices):

    predicted_class = classes[int(predictions[i])]
    actual_class = classes[int(y_samples[i])]

    result = "CORRECT" if predicted_class == actual_class else "WRONG"

    print(
        f"Image {i + 1}: "
        f"Predicted = {predicted_class} | "
        f"Actual = {actual_class} | "
        f"{result}"
    )


fig, axes = plt.subplots(2, 4, figsize=(14, 8))
axes = axes.flatten()

for i in range(num_examples):
    image = x_samples[i].transpose(1, 2, 0)

    predicted_class = classes[int(predictions[i])]
    actual_class = classes[int(y_samples[i])]

    axes[i].imshow(image)
    axes[i].set_title(
        f"Predicted: {predicted_class}\n"
        f"Actual: {actual_class}",
        fontsize=11
    )
    axes[i].axis("off")


for i in range(num_examples, len(axes)):
    axes[i].axis("off")


fig.suptitle(
    f"CNN Brain Tumor Classification Results\n"
    f"Test Accuracy: {test_accuracy:.2f}%",
    fontsize=16
)

plt.tight_layout()

output_path = "results/test_predictions.png"

plt.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print(f"\nPrediction visualization saved to: {output_path}")