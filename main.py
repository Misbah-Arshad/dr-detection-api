import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile
from torchvision import transforms, models
from PIL import Image
import io

# 1. Configuration
# We updated this path to look inside your subfolder
MODEL_PATH = "DR_Backend/swin_best.pth" 
CLASS_NAMES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferative_DR"]

app = FastAPI()

# 2. Define Model Architecture (Swin Transformer)
def get_model():
    # Loading the base Swin-T architecture
    model = models.swin_t(weights=None) 
    # Adjusting the final head to match your 5 classes
    n_inputs = model.head.in_features
    model.head = nn.Linear(n_inputs, len(CLASS_NAMES))
    return model

# 3. Load the Trained Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model()

try:
    # We use map_location=cpu because Render free tier doesn't have a GPU
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print("✅ Model loaded successfully from DR_Backend folder!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# 4. Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 5. API Endpoints
@app.get("/")
def home():
    return {"message": "DR Detection API is Running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the image sent by the Android app
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    
    # Preprocess and Run Inference
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        
    result = CLASS_NAMES[predicted.item()]
    
    return {
        "prediction": result,
        "class_index": predicted.item()
    }

