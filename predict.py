import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import json
from PIL import Image

from torchvision import models
from model_utils import load_checkpoint, process_image, predict

# Define imshow function for displaying image
def imshow(image, ax=None, title=None):
    """Display a tensor image"""
    if ax is None:
        fig, ax = plt.subplots()
    
    # Convert image from torch tensor to numpy
    image = image.numpy().transpose((1, 2, 0))
    
    # Unnormalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    if title:
        ax.set_title(title)
    
    ax.axis("off")
    return ax

# Set up argument parser
parser = argparse.ArgumentParser(description="Predict flower name from an image using a trained deep learning model.")
parser.add_argument("--input", type=str, required=True, help="Path to input image")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
parser.add_argument("--top_k", type=int, default=5, help="Return top K most likely classes")
parser.add_argument("--category_names", type=str, help="Path to a JSON file mapping categories to real names")
parser.add_argument("--gpu", action="store_true", help="Use GPU for inference if available")

args = parser.parse_args()

# Load the model
model = load_checkpoint(args.checkpoint)

# Use GPU if specified and available
device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
model.to(device)


# Process image and make prediction
probs, classes = predict(args.input, model, args.top_k)

# Convert class indices to actual flower names if category_names.json is provided
if args.category_names:
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    # Convert numerical labels to flower names
    class_labels = [cat_to_name.get(str(cls), "Unknown") for cls in classes]
else:
    class_labels = classes  # Use class indices if no mapping is provided

print("Probabilities:", probs)  # Debugging
print("Classes:", class_labels) 

# Load and process the input image
image = process_image(args.input)

# Create the plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Display the input image using imshow
imshow(image.squeeze(), ax=ax1)  # Ensure imshow works with ax=ax1
ax1.set_title("Input Image")

# Plot the top 5 predictions
ax2.barh(np.arange(len(probs)), probs, align="center", color="blue")
ax2.set_yticks(np.arange(len(probs)))
ax2.set_yticklabels(class_labels)
ax2.invert_yaxis()  # Highest probability at the top
ax2.set_xlabel("Probability")
ax2.set_title("Top 5 Predictions")

plt.tight_layout()
fig.show()  # Explicitly show figure
plt.pause(0.001)
plt.show(block=True)  # Ensure proper display



