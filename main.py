import torch
import timm
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import torchvision.transforms as T

app = FastAPI()

# 1. Load your Model
DEVICE = torch.device("cpu")
model = timm.create_model('swin_tiny_patch4_window7_224.ms_in1k', pretrained=True, num_classes=6)
model.load_state_dict(torch.load("swin_best.pth", map_location=DEVICE))
model.eval()

# 2. Define the exact same transform from Colab
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. Label Map
GRADE_NAMES = {
    0: "No Diabetic Retinopathy",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR",
    5: "Ungradable"
}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the image sent from the Android App
    img_bytes = await file.read()
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    
    # Preprocess and Predict
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).item()
    
    # Send back the result as JSON
    return {
        "grade": prediction,
        "diagnosis": GRADE_NAMES[prediction]
    }