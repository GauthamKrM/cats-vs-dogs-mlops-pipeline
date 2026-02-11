import sys
import os
import io
from unittest.mock import MagicMock, patch
from PIL import Image
import torch

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

def test_health_check_api():
    """Test health endpoint. We mock the model as loaded."""
    with patch('src.app.model') as mock_model:
        # Simulate model being present
        mock_model.__class__ = MagicMock
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

def test_health_check_unhealthy():
    """Test health endpoint when model is missing."""
    with patch('src.app.model', None):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "unhealthy", "reason": "Model not loaded"}

def test_predict_api_success():
    """Test the predict endpoint with a successful mock prediction."""
    # Create a dummy image
    img = Image.new('RGB', (224, 224), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    # Mock the model output
    with patch('src.app.model') as mock_model:
        # output of model(tensor) -> logits
        # We have 2 classes. 
        # We need to mock the forward pass.
        # tensor shape: (1, 3, 224, 224)
        # output shape: (1, 2)
        
        # We need to mock the __call__ method of the model
        mock_output = torch.tensor([[2.0, 1.0]]) # Class 0 (Cat) has higher score
        mock_model.return_value = mock_output
        
        # We also need to mock torch.device to avoid cuda issues if env has cuda
        with patch('src.app.device', torch.device('cpu')):
            response = client.post(
                "/predict",
                files={"file": ("test.jpg", img_byte_arr, "image/jpeg")}
            )
    
    assert response.status_code == 200
    json_resp = response.json()
    assert json_resp["prediction"] == "Cat"
    assert "confidence" in json_resp
