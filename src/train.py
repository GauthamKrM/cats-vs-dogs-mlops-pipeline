import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import mlflow
import mlflow.pytorch
import argparse
import sys
import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (safe for CI/servers)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Add project root to sys.path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import get_dataloaders
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def save_loss_curve(train_losses, val_losses, train_accs, val_accs, save_path="loss_curve.png"):
    """Plots and saves loss and accuracy curves."""
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss plot
    ax1.plot(epochs, train_losses, 'b-o', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-o', label='Val Loss')
    ax1.set_title('Loss per Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy plot
    ax2.plot(epochs, train_accs, 'b-o', label='Train Accuracy')
    ax2.plot(epochs, val_accs, 'r-o', label='Val Accuracy')
    ax2.set_title('Accuracy per Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


def save_confusion_matrix(all_labels, all_preds, class_names, save_path="confusion_matrix.png"):
    """Plots and saves the confusion matrix."""
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title('Confusion Matrix (Validation Set)')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


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

        # Track metrics per epoch for plotting
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []

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
            # Collect predictions for confusion matrix on last epoch
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            if len(val_loader.dataset) > 0:
                val_epoch_loss = val_loss / len(val_loader.dataset)
                val_epoch_acc = val_correct / val_total
            else:
                val_epoch_loss = 0.0
                val_epoch_acc = 0.0

            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} "
                  f"Val Loss: {val_epoch_loss:.4f} Val Acc: {val_epoch_acc:.4f}")

            # Log scalar metrics
            mlflow.log_metric("loss", epoch_loss, step=epoch)
            mlflow.log_metric("accuracy", epoch_acc, step=epoch)
            mlflow.log_metric("val_loss", val_epoch_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_epoch_acc, step=epoch)

            # Store for plotting
            train_losses.append(epoch_loss)
            val_losses.append(val_epoch_loss)
            train_accs.append(epoch_acc)
            val_accs.append(val_epoch_acc)

        # --- Generate and log Loss Curve artifact ---
        os.makedirs("artifacts", exist_ok=True)
        loss_curve_path = save_loss_curve(
            train_losses, val_losses, train_accs, val_accs,
            save_path="artifacts/loss_curve.png"
        )
        mlflow.log_artifact(loss_curve_path)
        print(f"Loss curve saved and logged: {loss_curve_path}")

        # --- Generate and log Confusion Matrix artifact ---
        class_names = ['Cat', 'Dog']
        cm_path = save_confusion_matrix(
            all_labels, all_preds, class_names,
            save_path="artifacts/confusion_matrix.png"
        )
        mlflow.log_artifact(cm_path)
        print(f"Confusion matrix saved and logged: {cm_path}")

        # Save model
        os.makedirs("models", exist_ok=True)
        model_path = "models/model.pt"
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)

        # Log model via MLflow model registry
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
