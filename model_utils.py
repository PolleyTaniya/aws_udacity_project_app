import torch
from torchvision import models, transforms
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import EfficientNet_V2_S_Weights
import os

class Classifier(nn.Module):
    def __init__(self, input_units, hidden_layers_1_units, hidden_layers_2_units, output_units=102):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_units, hidden_layers_1_units)
        self.fc2 = nn.Linear(hidden_layers_1_units, hidden_layers_2_units)
        self.fc3 = nn.Linear(hidden_layers_2_units, output_units)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

def build_model(arch="efficientnet_v2_s", hidden_layers_1_units=512, hidden_layers_2_units=128):
    """Loads EfficientNetV2-S, replaces classifier, and returns the model."""
    model = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    
    for param in model.features.parameters():
        param.requires_grad = False  
        
    input_units = model.classifier[1].in_features

    model.classifier = Classifier(input_units, hidden_layers_1_units, hidden_layers_2_units)
    return model

def train_model(model, trainLoader, validLoader, criterion, optimizer, epochs=5, device="cuda"):
    """Trains the model and validates it."""
    
    model.to(device)
    class_to_idx = trainLoader.dataset.class_to_idx  #  class-to-index
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0

        for inputs, labels in trainLoader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        model.eval()
        valid_loss = 0
        accuracy = 0

        with torch.no_grad():
            for inputs, labels in validLoader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()

                _, predicted = outputs.max(dim=1)
                accuracy += (predicted == labels).sum().item() / len(labels)

        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {running_loss/len(trainLoader):.3f} - "
              f"Valid Loss: {valid_loss/len(validLoader):.3f} - "
              f"Accuracy: {accuracy/len(validLoader):.3f}")
    
    return model, class_to_idx


def save_checkpoint(model, optimizer, epochs, class_to_idx, save_dir, filename="checkpoint.pth"):
    checkpoint = {
        'epochs': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_to_idx': class_to_idx,  # Save class index mapping
    }
    os.makedirs(save_dir, exist_ok=True)
    torch.save(checkpoint, os.path.join(save_dir, filename))
    print(f"Checkpoint saved to {save_dir}/{filename}")


def load_checkpoint(filepath):
    """Load a model checkpoint and rebuild the model."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint from: {filepath}")
    checkpoint = torch.load(filepath, map_location=device)
    
    arch = checkpoint.get('architecture', 'efficientnet_v2_s')
    
    model = build_model(arch)  # Ensure the correct architecture is used
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    # Ensure class_to_idx is loaded correctly
    if 'class_to_idx' in checkpoint:
        model.class_to_idx = checkpoint['class_to_idx']
    else:
        print("Warning: class_to_idx not found in checkpoint!")

    model.to(device)
    model.eval()  # Set model to evaluation mode
    return model



def process_image(image_path):
    """Preprocess an image for model prediction."""
    image = Image.open(image_path)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = transform(image)
    return image.unsqueeze(0)

def predict(image_path, model, topk=5):
    """Predict the top K classes of an image."""
    model.eval()
    device = next(model.parameters()).device

    image = process_image(image_path).to(device)

    with torch.no_grad():
        output = model(image)
    probabilities = torch.exp(output)

    top_probs, top_classes = probabilities.topk(topk, dim=1)

    # Convert class indices back to actual labels
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}  # Inverse mapping
    top_labels = [idx_to_class[i] for i in top_classes.cpu().numpy()[0]]  # Convert indices to labels

    return top_probs.cpu().numpy()[0], top_labels  # Return class labels instead of indices

#Train and Save the Model Correctly
if __name__ == "__main__":
    from load_data import load_data
    from torch import optim

    trainLoader, validLoader, _ = load_data("flowers")

    model = build_model()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)

    class_to_idx = train_model(model, trainLoader, validLoader, criterion, optimizer)

    save_checkpoint(model, optimizer, epochs=5, class_to_idx=class_to_idx, save_dir="saved_models")


