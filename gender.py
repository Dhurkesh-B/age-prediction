import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import io

# --------------------
# 1. Load Gender Model
# --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained gender model
try:
    # Initialize ResNet18 model with 2 classes (male, female)
    gender_model = models.resnet18(pretrained=False)
    num_ftrs = gender_model.fc.in_features
    gender_model.fc = nn.Linear(num_ftrs, 2)  # 2 classes: male, female
    
    # Load the trained weights
    gender_model.load_state_dict(torch.load("gender_model.pth", map_location=device))
    gender_model.to(device)
    gender_model.eval()
    
    print("Gender model loaded successfully")
    gender_model_available = True
except FileNotFoundError:
    print("Gender model file 'gender_model.pth' not found")
    gender_model_available = False
except Exception as e:
    print(f"Error loading gender model: {e}")
    gender_model_available = False

# Class names (based on your training dataset structure)
class_names = ['female', 'male']  # Alphabetical order as used in ImageFolder

# Image preprocessing (same as validation transforms from training)
gender_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --------------------
# 2. FastAPI App
# --------------------
app = FastAPI(title="Gender Prediction API", version="1.0.0")

@app.post("/gender")
async def predict_gender(file: UploadFile = File(...)):
    """
    Predict gender from uploaded image
    """
    try:
        if not gender_model_available:
            raise HTTPException(status_code=503, detail="Gender model not available")
            
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Apply transformations
        img_tensor = gender_transform(img).unsqueeze(0).to(device)
        
        # Model prediction
        with torch.no_grad():
            outputs = gender_model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            _, predicted = torch.max(outputs, 1)
            pred_class = predicted.item()
            confidence = probabilities[pred_class].item()
        
        # Get predicted gender
        predicted_gender = class_names[pred_class]
        
        # Get probabilities for both classes
        female_prob = probabilities[0].item()
        male_prob = probabilities[1].item()
        
        return JSONResponse(content={
            "predicted_gender": predicted_gender,
            "confidence": confidence,
            "probabilities": {
                "female": female_prob,
                "male": male_prob
            },
            "predicted_class": pred_class,
            "type": "gender"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gender prediction error: {str(e)}")

# --------------------
# 3. Run Application
# --------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)