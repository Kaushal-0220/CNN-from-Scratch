# Brain Tumor MRI Classification: CNN from Scratch

A convolutional neural network built entirely from scratch (no PyTorch, TensorFlow or Keras) using **CuPy** for GPU-accelerated array math. It classifies brain MRI images into four categories:

- `glioma`
- `meningioma`
- `notumor`
- `pituitary`

Every component is implemented by hand, including the forward and backward passes: convolution (via im2col), max pooling, dense layers, dropout, ReLU, softmax with cross-entropy loss, and the Adam optimizer.

## Results

| Metric | Value |
|---|---|
| Test images | 1600 |
| Correct predictions | 1444 |
| **Test accuracy** | **90.25%** |
| Final training accuracy | ~99% |
| Final training loss | ~0.02 |

Training for 20 epochs gives smooth convergence (see `results/training_accuracy.png` and `results/training_loss.png`). The gap between training accuracy (~99%) and test accuracy (~90%) indicates some overfitting. `results/test_predictions.png` shows a random sample of test predictions each time `test.py` runs.

## Architecture

Input: `3 x 128 x 128` RGB image, scaled to `[0, 1]`.

```
Conv(3 -> 32, 3x3, pad 1)   -> ReLU -> MaxPool(2x2)   # 32 x 64 x 64
Conv(32 -> 64, 3x3, pad 1)  -> ReLU -> MaxPool(2x2)   # 64 x 32 x 32
Conv(64 -> 128, 3x3, pad 1) -> ReLU -> MaxPool(2x2)   # 128 x 16 x 16
Flatten                                               # 32768
Dense(32768 -> 256) -> ReLU -> Dropout(0.5)
Dense(256 -> 4)     -> Softmax + Categorical Cross-Entropy
```

**Training setup**

- Optimizer: Adam (`lr=0.001`, `decay=1e-5`, `beta1=0.9`, `beta2=0.999`)
- Batch size: 32
- Epochs: 20
- Convolution weights: He initialization; dense weights: `0.01 * N(0, 1)`
- Dropout is applied only during training and disabled at inference

## Project Structure

```
.
├── src/
│   ├── __init__.py      # Exposes the whole package
│   ├── library.py       # Shared imports (numpy, cupy, cv2, os, kagglehub)
│   ├── activation.py    # ReLU, Softmax
│   ├── data.py          # Dataset discovery, loading and preprocessing
│   ├── layers.py        # Convolution (im2col), pooling, flatten, dense, dropout
│   ├── loss.py          # Cross-entropy and combined softmax + cross-entropy
│   ├── optimizer.py     # Adam optimizer
│   └── model.py         # CNN class: forward, backward, update, train_step, predict
├── dataset/             # Place the dataset here (see below)
├── results/             # Generated: weights, plots, evaluation output
├── train.py             # Train the model and save weights and plots
├── test.py              # Evaluate saved weights on the test set
└── requirements.txt
```

## Requirements

- Python 3.9+
- An NVIDIA GPU with a compatible CUDA toolkit (required by CuPy)
- Dependencies in `requirements.txt`:

```
numpy
cupy-cuda13x[ctk]
opencv-python
kagglehub
matplotlib
```

Install with:

```bash
pip install -r requirements.txt
```

> The `cupy-cuda13x` package targets CUDA 13. If your system uses a different CUDA version, install the matching CuPy build (for example `cupy-cuda12x`) instead.

## Dataset

The code expects a `dataset/` folder in the project root containing `Training` and `Testing` folders (found recursively, case-insensitive). Each should contain one subfolder per class:

```
dataset/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

A Brain Tumor MRI dataset from Kaggle with this layout works. Download it and extract it into `dataset/`.

Images are read with OpenCV, resized to 128x128, normalized to `[0, 1]`, and transposed to channel-first format `(N, C, H, W)`. Class labels come from the alphabetically sorted folder names.

## Usage

### 1. Train

```bash
python train.py
```

This will:

1. Load the training and testing data
2. Train for 20 epochs, printing accuracy, loss and learning rate per epoch
3. Save to `results/`:
   - `training_loss.png`
   - `training_accuracy.png`
   - `model_weights.npz`

### 2. Evaluate

```bash
python test.py
```

This will:

1. Load weights from `results/model_weights.npz`
2. Compute accuracy on the full test set
3. Save `results/evaluation_results.txt`
4. Print and plot predictions for 8 random test images, saved to `results/test_predictions.png`

## How It Works

- **Convolution**: implemented with `im2col`, which turns each receptive field into a column so the convolution becomes one matrix multiplication. The backward pass uses `col2im` with `add.at` to accumulate gradients.
- **Pooling**: 2x2 max pooling via reshaping; the backward pass routes gradients to the max positions with a mask.
- **Softmax + cross-entropy**: combined into one class for a numerically stable, simplified gradient (`softmax(output) - y_true`, averaged over the batch).
- **Adam**: per-parameter momentum and cache with bias correction and learning rate decay.

## Possible Improvements

- Add a validation split and early stopping to reduce overfitting
- Data augmentation (flips, rotations, brightness shifts)
- Batch normalization and L2 regularization
- Per-class metrics (precision, recall, confusion matrix), which matter for a medical task
- Save and restore optimizer state to resume training

## Disclaimer

This is an educational project for learning how CNNs work internally. It is **not** a medical device and must not be used for diagnosis or clinical decisions.
