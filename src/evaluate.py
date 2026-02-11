import torch
import torch.nn as nn
from torchvision import models
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import get_dataloaders
import argparse
import json

def evaluate_model(model_path, data_dir, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Data
    _, _, test_loader = get_dataloaders(data_dir, batch_size)
    
    # Load Model
    model = models.resnet18(pretrained=False) # Architecture only
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    test_loss = running_loss / len(test_loader.dataset)
    test_acc = correct / total
    
    print(f"Test Loss: {test_loss:.4f} Test Acc: {test_acc:.4f}")
    
    # Save Metrics
    metrics = {
        "test_loss": test_loss,
        "test_accuracy": test_acc
    }
    
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data")
    args = parser.parse_args()
    
    evaluate_model(args.model_path, args.data_dir)
