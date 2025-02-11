import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from model import build_model, train_model, save_checkpoint
from utils import load_data

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a neural network on flower data.")
    parser.add_argument("data_directory", type=str, help="Path to the dataset")
    parser.add_argument("--save_dir", type=str, default="saved_models", help="Directory to save checkpoints")
    parser.add_argument("--arch", type=str, default="vgg13", help="Model architecture")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--hidden_units", nargs="+", type=int, default=[512, 256], help="Number of units in hidden layers")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")
    args = parser.parse_args()

    # Load the data
    trainloader, validloader, _ = load_data(args.data_directory)

    # Build the model
    model = build_model(args.arch, args.hidden_units)

    # Train the model
    train_model(model, trainloader, validloader, args.epochs, args.learning_rate, args.gpu)

    # Save the checkpoint
    save_checkpoint(model, args.save_dir, args.arch, args.hidden_units)

if __name__ == '__main__':
    main()