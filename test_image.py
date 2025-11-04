"""
Test script for MNIST digit classification.
Can test on images from the test set or on custom uploaded images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import sys
import os

# Import the model architecture
from simple_NN_PT import NN

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model parameters (must match training)
input_size = 784
num_classes = 10

# Initialize model
model = NN(input_size=input_size, num_classes=num_classes).to(device)

# Load trained model
model_path = "mnist_model.pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded from {model_path}")
else:
    print(f"Error: Model file {model_path} not found. Please train the model first by running simple_NN_PT.py")
    sys.exit(1)


def preprocess_image(image_path):
    """
    Preprocess an image for MNIST classification.
    Converts to grayscale, resizes to 28x28, inverts if needed, and normalizes.
    """
    # Load image
    img = Image.open(image_path)
    
    # Convert to grayscale if needed
    if img.mode != 'L':
        img = img.convert('L')
    
    # Resize to 28x28 (MNIST size)
    img = img.resize((28, 28), Image.LANCZOS)
    
    # Convert to tensor and normalize (MNIST images are inverted - black background, white digits)
    # If your image has white background with black digits, we'll handle inversion
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
    ])
    
    # Convert PIL to tensor
    img_tensor = transforms.ToTensor()(img)
    
    # Check if image needs inversion (MNIST has white digits on black background)
    # If image is mostly white (mean > 0.5), invert it
    if img_tensor.mean() > 0.5:
        img_tensor = 1 - img_tensor
    
    # Normalize
    img_tensor = transforms.Normalize((0.1307,), (0.3081,))(img_tensor)
    
    # Reshape to (1, 784) for the model
    img_tensor = img_tensor.reshape(1, -1)
    
    return img_tensor


def predict_image(image_path):
    """
    Predict the digit in an image.
    """
    try:
        # Preprocess image
        img_tensor = preprocess_image(image_path)
        img_tensor = img_tensor.to(device)
        
        # Make prediction
        with torch.no_grad():
            scores = model(img_tensor)
            probabilities = F.softmax(scores, dim=1)
            confidence, prediction = torch.max(probabilities, 1)
            
        return int(prediction.item()), float(confidence.item())
    
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, None


def test_random_samples(num_samples=5):
    """
    Test on random samples from the test dataset.
    """
    from torch.utils.data import DataLoader
    import torchvision.datasets as datasets
    
    test_dataset = datasets.MNIST(root="dataset/", train=False, transform=transforms.ToTensor(), download=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)
    
    print(f"\nTesting on {num_samples} random samples from test set:")
    print("-" * 50)
    
    model.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            if i >= num_samples:
                break
            
            data = data.to(device)
            data = data.reshape(data.shape[0], -1)
            
            scores = model(data)
            probabilities = F.softmax(scores, dim=1)
            confidence, prediction = torch.max(probabilities, 1)
            
            print(f"Sample {i+1}:")
            print(f"  True label: {target.item()}")
            print(f"  Predicted: {prediction.item()}")
            print(f"  Confidence: {confidence.item()*100:.2f}%")
            print(f"  {'✓ Correct' if prediction.item() == target.item() else '✗ Incorrect'}")
            print()


if __name__ == "__main__":
    print("=" * 60)
    print("MNIST Digit Classifier - Test Script")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        # Test on provided image path
        image_path = sys.argv[1]
        if not os.path.exists(image_path):
            print(f"Error: Image file '{image_path}' not found.")
            sys.exit(1)
        
        print(f"\nTesting image: {image_path}")
        prediction, confidence = predict_image(image_path)
        
        if prediction is not None:
            print(f"\nPrediction: {prediction}")
            print(f"Confidence: {confidence*100:.2f}%")
        else:
            print("Failed to make prediction.")
    else:
        # Test on random samples from test set
        print("\nNo image provided. Testing on random samples from test set...")
        print("Usage: python test_image.py <path_to_image>")
        print()
        test_random_samples(5)


