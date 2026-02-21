import torch
import torch.nn as nn
import timm  # Must match training
from fastapi import FastAPI, File, UploadFile
from torchvision import transforms
from PIL import Image
import io
import gc

# 1. Configuration
MODEL_PATH = "DR_Backend/swin_best.pth" 
# DDR Dataset standard labels
CLASS_NAMES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferative_DR", "Ungradable"]

app = FastAPI()

# 2. Define Model Architecture (Matches your training exactly)
def get_model():
    # Use the same model string you used in training (likely 'swin_tiny_patch4_window7_224')
    # If you aren't sure, 'swin_tiny_patch4_window7_224' is the standard timm tiny swin.
    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=6)
    return model

# 3. Load Model
device = torch.device("cpu")
model = get_model()

try:
    # Use weights_only=True for safety and memory
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    del state_dict
    gc.collect()
    print("✅ Model loaded successfully from TIMM architecture!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# 4. Preprocessing (Matches your val_transform)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.get("/")
def home():
    return {"message": "DR Detection API is Running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        with torch.inference_mode():
            input_tensor = transform(image).unsqueeze(0).to(device)
            outputs = model(input_tensor)
            # Apply softmax to see confidence (Optional but helpful)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            conf, predicted = torch.max(probabilities, 1)
            
        result = CLASS_NAMES[predicted.item()]
        confidence_score = conf.item() * 100

        del image_data, image, input_tensor, outputs
        gc.collect()

        return {
            "prediction": result,
            "confidence": f"{confidence_score:.2f}%",
            "class_index": predicted.item()
        }
    except Exception as e:
        return {"error": str(e)}
