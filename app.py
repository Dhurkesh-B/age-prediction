import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import io
import pickle
import uvicorn
import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --------------------
# 1. Load Image Model
# --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load image model checkpoint
from torchvision.models import resnet18
checkpoint_path = "age_model.pth"
state_dict = torch.load(checkpoint_path, map_location=device)

# Determine number of classes from checkpoint
try:
    num_classes = state_dict["fc.weight"].shape[0]
except KeyError:
    num_classes = 17

# Build image model
image_model = resnet18(pretrained=False)
image_model.fc = nn.Linear(image_model.fc.in_features, num_classes)
image_model.load_state_dict(state_dict)
image_model.to(device)
image_model.eval()

# Create class to age mapping for image model
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

print(f"Image model loaded with {num_classes} classes")

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --------------------
# 2. Load Text Model
# --------------------
try:
    # Load text model and vectorizer
    with open("age-text.pkl", "rb") as f:
        text_model = pickle.load(f)
    
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    
    print("Text model loaded successfully")
    text_model_available = True
except FileNotFoundError as e:
    print(f"Text model files not found: {e}")
    text_model_available = False

# --------------------
# 3. FastAPI App Setup
# --------------------
app = FastAPI(title="AGE PREDIX - Unified Age Prediction API", version="1.0.0")

# Pydantic models
class TextInput(BaseModel):
    text: str

class ContactForm(BaseModel):
    name: str
    email: str
    subject: str
    message: str

# --------------------
# 4. Serve Static Files and HTML
# --------------------
@app.get("/", response_class=HTMLResponse)
async def serve_homepage():
    """Serve the main HTML page"""
    try:
        with open("templates/index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>AGE PREDIX API</h1><p>Frontend not found. API is running at /docs</p>")

@app.get("/favicon.png")
async def get_favicon():
    """Serve the favicon"""
    try:
        with open("templates/favicon.png", "rb") as f:
            png_content = f.read()
        from fastapi.responses import Response
        return Response(
            content=png_content, 
            media_type="image/png",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Favicon not found")

# --------------------
# 5. API Endpoints
# --------------------
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "image_model_classes": num_classes,
        "text_model_available": text_model_available
    }

@app.post("/predict/image")
async def predict_image_age(file: UploadFile = File(...)):
    """
    Predict age from uploaded image
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        # Model prediction
        with torch.no_grad():
            outputs = image_model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            pred_class = predicted.item()
            confidence = torch.softmax(outputs, dim=1).max().item()

        # Map back to age group
        pred_age_group = class2age[pred_class]

        return JSONResponse(content={
            "predicted_age_group": pred_age_group,
            "predicted_class": pred_class,
            "confidence": confidence,
            "type": "image"
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image prediction error: {str(e)}")

@app.post("/predict/text")
async def predict_text_age(input_data: TextInput):
    """
    Predict age from text input
    """
    try:
        if not text_model_available:
            raise HTTPException(status_code=503, detail="Text model not available")
        
        # Transform text and predict
        X_test = vectorizer.transform([input_data.text])
        prediction = text_model.predict(X_test)[0]
        
        # Get prediction probability for confidence
        try:
            probabilities = text_model.predict_proba(X_test)[0]
            confidence = max(probabilities)
        except:
            confidence = 0.8  # Default confidence if predict_proba not available

        return JSONResponse(content={
            "input": input_data.text,
            "predicted_age_group": prediction,
            "confidence": confidence,
            "type": "text"
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text prediction error: {str(e)}")

# Legacy endpoints for backward compatibility
@app.post("/predict")
async def predict_legacy(file: UploadFile = File(...)):
    """Legacy image prediction endpoint for backward compatibility"""
    return await predict_image_age(file)

@app.post("/contact")
async def send_contact_email(contact_data: ContactForm):
    """
    Send contact form email
    """
    try:
        # Email configuration - you'll need to set these environment variables
        smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        sender_email = os.getenv("SENDER_EMAIL", "your-email@gmail.com")
        sender_password = os.getenv("SENDER_PASSWORD", "your-app-password")
        recipient_email = os.getenv("RECIPIENT_EMAIL", "your-email@gmail.com")
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = f"AGE PREDIX Contact Form: {contact_data.subject}"
        
        # Email body
        body = f"""
New contact form submission from AGE PREDIX website:

Name: {contact_data.name}
Email: {contact_data.email}
Subject: {contact_data.subject}

Message:
{contact_data.message}

---
Sent on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
From: AGE PREDIX Contact Form
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        text = msg.as_string()
        server.sendmail(sender_email, recipient_email, text)
        server.quit()
        
        return JSONResponse(content={
            "success": True,
            "message": "Email sent successfully!"
        })
        
    except Exception as e:
        print(f"Email error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": "Failed to send email. Please try again later."
            }
        )

# --------------------
# 6. Run Application
# --------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
