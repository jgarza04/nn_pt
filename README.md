# MNIST Neural Network - Testing Guide

This project contains a simple neural network for MNIST digit classification using PyTorch.

## Files

- `simple_NN_PT.py` - Main training script that trains the model and saves it
- `test_image.py` - Test script for predicting digits in images
- `mnist_model.pth` - Trained model weights (created after training)

## Setup

Useful commands: 
- To train: 
```bash
python absolute/path/to/simple__NN_PT.py"
```
- To test on costum image:
```bash
python python /absolute/path/to/Simple_NN_PT/test_image.py /absolute/path/to/your_digit.png
```
Make sure you have the required dependencies:

```bash
pip install torch torchvision pillow numpy
```

## Training the Model

First, train the model by running:

```bash
python simple_NN_PT.py
```

This will:
- Download the MNIST dataset (if not already present)
- Train the neural network for 10 epochs
- Test accuracy on training and test sets
- Save the trained model to `mnist_model.pth`

## Testing the Model

### Option 1: Test on Random Samples from Test Set

Run the test script without arguments:

```bash
python test_image.py
```

This will test on 5 random samples from the MNIST test set and show:
- True label
- Predicted label
- Confidence score
- Whether the prediction was correct

### Option 2: Test on Your Own Image

You can test on your own image file:

```bash
python test_image.py path/to/your/image.png
```

**Image Requirements:**
- The image will be automatically converted to grayscale
- Resized to 28x28 pixels (MNIST size)
- Works best with:
  - Black digits on white background
  - Clear, centered digits
  - High contrast images

### Example

```bash
# Test on a custom image
python test_image.py my_digit.png

# Output:
# Prediction: 7
# Confidence: 95.23%
```

## How It Works

1. **Training**: The model is a simple fully connected neural network with:
   - Input layer: 784 nodes (28×28 pixels)
   - Hidden layer: 50 nodes with ReLU activation
   - Output layer: 10 nodes (one for each digit 0-9)

2. **Prediction**: 
   - Images are preprocessed to match MNIST format (28×28 grayscale, normalized)
   - The model outputs probabilities for each digit (0-9)
   - The digit with the highest probability is the prediction


