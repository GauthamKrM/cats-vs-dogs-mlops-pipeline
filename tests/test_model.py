import torch
import pytest
import os
import sys
from PIL import Image

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import get_dataloaders

@pytest.fixture
def dummy_data(tmp_path):
    """Creates a dummy dataset with Cat and Dog images"""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    (data_dir / "Cat").mkdir()
    (data_dir / "Dog").mkdir()
    
    # Create distinct dummy images (just 1x1 noise or solid color)
    for i in range(10):
        img_cat = Image.new('RGB', (224, 224), color = 'red')
        img_cat.save(data_dir / "Cat" / f"cat_{i}.jpg")
        
        img_dog = Image.new('RGB', (224, 224), color = 'blue')
        img_dog.save(data_dir / "Dog" / f"dog_{i}.jpg")
        
    return str(data_dir)

def test_data_loader_structure(dummy_data):
    """Test that dataloaders return expected splits and batches"""
    batch_size = 2
    train_loader, val_loader, test_loader = get_dataloaders(dummy_data, batch_size=batch_size)
    
    total_samples = 20 # 10 cats + 10 dogs
    
    # Check splits (approx 80/10/10)
    # len(dataset) gives number of samples
    assert len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset) == total_samples
    
    # Check batching works
    for inputs, labels in train_loader:
        assert inputs.shape == (batch_size, 3, 224, 224)
        assert labels.shape == (batch_size,)
        break

def test_model_inference_shape():
    """Test standard model architecture inference"""
    from torchvision import models
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)
    
    input_tensor = torch.randn(1, 3, 224, 224)
    output = model(input_tensor)
    
    assert output.shape == (1, 2)
