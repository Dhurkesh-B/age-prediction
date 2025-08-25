import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import io

# --------------------
# 1. Load Model
# --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔹 Load checkpoint and rebuild the same architecture you trained
from torchvision.models import resnet18
checkpoint_path = "age_model.pth"
state_dict = torch.load(checkpoint_path, map_location=device)

# Determine number of classes from checkpoint
try:
    num_classes = state_dict["fc.weight"].shape[0]
except KeyError:
    # Fallback if checkpoint keys are nested or missing
    num_classes = 17

# Build model with correct head size and load weights
model = resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# 🔹 Define the correct 15-class mapping from your FairFace training
fairface_age_groups = [
    "0-2", "10-19", "20-29", "3-9", "30-39", "40-49",
    "50-59", "60-69", "70+", "more than 70"
]

# Create class to age mapping
if num_classes == 15:
    class2age = {
        0: "0-2", 1: "10-19", 2: "20-29", 3: "3-9", 4: "30-39",
        5: "40-49", 6: "50-59", 7: "60-69", 8: "70+", 9: "more than 70",
        10: "age_group_10", 11: "age_group_11", 12: "age_group_12",
        13: "age_group_13", 14: "age_group_14"
    }
elif num_classes == 10:
    class2age = {
        0: "0-2", 1: "3-9", 2: "10-19", 3: "20-29", 4: "30-39",
        5: "40-49", 6: "50-59", 7: "60-69", 8: "70+", 9: "more than 70"
    }
else:
    class2age = {i: f"age_group_{i}" for i in range(num_classes)}

print(f"Model loaded with {num_classes} classes")
print(f"Class to age mapping: {class2age}")

# --------------------
# 2. Preprocessing
# --------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --------------------
# 3. FastAPI App
# --------------------
app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        # Model prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            pred_class = predicted.item()

        pred_age_group = class2age[pred_class]

        return JSONResponse(content={"predicted_age_group": pred_age_group})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# --------------------
# 4. Run with Uvicorn
# --------------------
# Save this as main.py and run:
# uvicorn main:app --reload
