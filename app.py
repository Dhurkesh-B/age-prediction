import os
import torch
from flask import Flask, render_template, request, jsonify
from PIL import Image
from transformers import AutoImageProcessor, ViTForImageClassification
from werkzeug.utils import secure_filename
import base64
from io import BytesIO

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model globally
model_id = "nateraw/vit-age-classifier"
processor = AutoImageProcessor.from_pretrained(model_id)
model = ViTForImageClassification.from_pretrained(model_id)

def predict_age(image):
    """Predict age from image using the loaded model"""
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs)
    
    proba = output.logits.softmax(1)[0]
    id2label = model.config.id2label
    
    # Get the top prediction
    best_idx = int(proba.argmax())
    predicted_age = id2label[best_idx]
    confidence = float(proba[best_idx])
    
    return predicted_age, confidence

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' in request.files:
            # Handle file upload
            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            # Read and process image
            image = Image.open(file.stream).convert("RGB")
            
        elif 'image_data' in request.form:
            # Handle camera capture (base64 data)
            image_data = request.form['image_data']
            # Remove data URL prefix if present
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            # Decode base64 and open image
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            
        else:
            return jsonify({'error': 'No image provided'}), 400
        
        # Predict age
        predicted_age, confidence = predict_age(image)
        
        return jsonify({
            'age': predicted_age,
            'confidence': f"{confidence:.4f}",
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 