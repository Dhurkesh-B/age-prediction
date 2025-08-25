from fastapi import FastAPI
from pydantic import BaseModel
import pickle

# Load model and vectorizer
with open("age-text.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# FastAPI app
app = FastAPI(title="Age Group Predictor")

# Request body schema
class TextInput(BaseModel):
    text: str

# Prediction endpoint
@app.post("/predict")
def predict_age(input_data: TextInput):
    X_test = vectorizer.transform([input_data.text])
    prediction = model.predict(X_test)[0]
    return {"input": input_data.text, "predicted_age_group": prediction}
