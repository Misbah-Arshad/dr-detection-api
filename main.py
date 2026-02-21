import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile
from torchvision import transforms, models
from PIL import Image
import io
import gc  # Added for memory management

# 1. Configuration
MODEL_PATH = "DR_Backend/swin_best.pth" 
# FIXED: Added 6th class to match your training (change "Unknown" to your actual 6th label)
CLASS_NAMES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferative_DR", "Unknown"]

app = FastAPI()

# 2. Define Model Architecture
def get_model():
    # Loading without weights to save initial RAM
    model = models.swin_t(weights=None) 
    n_inputs = model.head.in_features
    # FIXED: Now matches the 6 classes in CLASS_NAMES
    model.head = nn.Linear(n_inputs, len(CLASS_NAMES))
    return model

# 3. Load the Trained Model with Memory Optimization
device = torch.device("cpu") # Explicitly use CPU for Render Free Tier
model = get_model()

try:
    # Use weights_only=True if using newer torch versions to save RAM
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    # Delete state_dict and clear cache to free up RAM immediately
    del state_dict
    gc.collect() 
    print("✅ Model loaded successfully!")
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
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Use inference_mode for maximum memory efficiency
        with torch.inference_mode():
            input_tensor = transform(image).unsqueeze(0).to(device)
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            
        result = CLASS_NAMES[predicted.item()]
        
        # Clear temporary tensors from RAM
        del image_data, image, input_tensor, outputs
        gc.collect()

        return {
            "prediction": result,
            "class_index": predicted.item()
        }
    except Exception as e:
        return {"error": str(e)}
