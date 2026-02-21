from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
import torch
from torchvision import models, transforms
from PIL import Image
import io
import os
from prometheus_fastapi_instrumentator import Instrumentator
import logging
import time
from datetime import datetime
from contextlib import asynccontextmanager

# configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model
    load_model()
    yield

app = FastAPI(title="Cats vs Dogs Inference API", lifespan=lifespan)

Instrumentator().instrument(app).expose(app)

# Global model variable
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    global model
    try:
        # Load the same architecture as training
        model = models.resnet18(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 2)
        
        # Load weights
        model_path = "models/model.pt" 
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            print(f"Model loaded from {model_path}")
        else:
            print(f"Warning: Model not found at {model_path}. Inference will fail.")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.get("/health")
def health_check():
    if model is None:
        return {"status": "unhealthy", "reason": "Model not loaded"}
    return {"status": "healthy"}

def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return transform(image).unsqueeze(0).to(device)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    start_time = time.time()
    
    try:
        contents = await file.read()
        tensor = transform_image(contents)
        
        with torch.no_grad():
            outputs = model(tensor)
            _, predicted = torch.max(outputs, 1)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
        classes = ['Cat', 'Dog']
        class_name = classes[predicted.item()]
        confidence = probs[0][predicted.item()].item() # calculates_logger_info

        latency = time.time() - start_time #calcultes logging

        #logging
        logger.info(
        f"{datetime.now()} | "
        f"Prediction: {class_name} | "
        f"Confidence: {confidence:.4f} | "
        f"Latency: {latency:.4f}s"
    )
        
        return {
            "prediction": class_name,
            "confidence": probs[0][predicted.item()].item(),
            "probabilities": {
                "Cat": probs[0][0].item(),
                "Dog": probs[0][1].item()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        logger.error(f"Prediction error: {str(e)}") #error_logging

if __name__ == "__main__":
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)
