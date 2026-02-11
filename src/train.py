import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import mlflow
import mlflow.pytorch
import argparse
import sys
import os

# Add project root to sys.path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import get_dataloaders
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

def train_model(data_dir, epochs=5, learning_rate=0.001, batch_size=32):
    # MLflow tracking
    mlflow.set_experiment("cats_vs_dogs")
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("batch_size", batch_size)
        
        # Prepare Data
        train_loader, val_loader, _ = get_dataloaders(data_dir, batch_size)
        
        # Model (ResNet18)
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2) # Binary classification
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training Loop
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = correct / total
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            if len(val_loader.dataset) > 0:
                val_epoch_loss = val_loss / len(val_loader.dataset)
                val_epoch_acc = val_correct / val_total
            else:
                val_epoch_loss = 0.0
                val_epoch_acc = 0.0
            
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Val Loss: {val_epoch_loss:.4f} Val Acc: {val_epoch_acc:.4f}")
            
            # Log metrics
            mlflow.log_metric("loss", epoch_loss, step=epoch)
            mlflow.log_metric("accuracy", epoch_acc, step=epoch)
            mlflow.log_metric("val_loss", val_epoch_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_epoch_acc, step=epoch)

        # Save model
        os.makedirs("models", exist_ok=True)
        model_path = "models/model.pt"
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)
        
        # Also log model via mlflow model registry style
        mlflow.pytorch.log_model(model, "model")
        print("Training complete. Model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data", help="Path to dataset")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    
    args = parser.parse_args()
    
    train_model(args.data_dir, args.epochs, args.lr, args.batch_size)
