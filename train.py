import argparse
import torch
from torchvision import models
import torch.optim as optim
from load_data import load_data  # Ensure load_data is implemented
from model_utils import build_model, train_model, save_checkpoint  # Implement these in a separate file
import os

def main():
    parser = argparse.ArgumentParser(description="Train a neural network on flower data")
    parser.add_argument("--data_directory", type=str, help="Path to the dataset")
    parser.add_argument("--save_dir", type=str, default="saved_models", help="Directory to save checkpoints")
    parser.add_argument("--arch", type=str, default="efficientnet_v2_s", help="Model architecture")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden_layers_1_units", type=int, default=512, help="Number of neurons in 1st hidden layer")
    parser.add_argument("--hidden_layers_2_units", type=int, default=128, help="Number of neurons in 2nd hidden layer")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"], help="Optimizer to use")
    
    args = parser.parse_args()
    
    # Load the data
    trainLoader, validLoader, testLoader, image_datasets = load_data(args.data_directory)

    
    # Build the model
    model = build_model(args.arch, args.hidden_layers_1_units, args.hidden_layers_2_units)
    
     # Define device (GPU or CPU)
    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"

    # Define loss function
    criterion = torch.nn.CrossEntropyLoss()

    #Optimizer
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    # Train the model
    train_model(model, trainLoader, validLoader, criterion, optimizer, args.epochs, device)

    # Save the checkpoint
    save_checkpoint(model, optimizer, args.epochs, image_datasets['train'].class_to_idx, args.save_dir, filename="checkpoint.pth")


    print("Training complete.")

if __name__ == "__main__":
    main()