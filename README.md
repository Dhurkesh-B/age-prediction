# AgePredix: AI-Powered Age & Gender Prediction

Predict **age** and **gender** from images or text with advanced AI models.

AgePredix is a **FastAPI-based application** that leverages **deep learning** and **NLP** to predict a person's **age group** and **gender** from facial images or text input. Powered by pre-trained models and a sleek web interface, it delivers **fast** and **accurate** predictions.

🔗 **Live Demo:** [https://agepredix.bot.nu/](https://agepredix.bot.nu/)

---

## 📂 Project Structure

```bash
.
├── age_model.pth          # Pretrained age prediction model
├── gender_model.pth       # Pretrained gender prediction model
├── age-text.pkl           # Text-based age feature model
├── vectorizer.pkl         # Text vectorizer for preprocessing
├── app.py                 # FastAPI application entry point
├── training-code.py       # Model training script
├── templates/             
│   ├── index.html         # Web UI for predictions
│   └── favicon.png        # App favicon
├── tips.json              # Frontend tips and info
├── requirements.txt       # Python dependencies
└── __pycache__/           # Auto-generated Python cache
```

---

## ✨ Features

* **Image-Based Predictions**: Upload a facial image or use your webcam to predict age and gender.
* **Text-Based Predictions**: Estimate age from text input using linguistic patterns.
* **Non-Intrusive Verification**: Secure age estimation without storing personal data.
* **Age-Gating Support**: Restrict access to age-sensitive content with accurate predictions.
* **Interactive Web UI**: User-friendly interface for seamless interaction.
* **API Access**: FastAPI endpoints with Swagger documentation for developers.

---

## 🛠️ Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/agepredix.git
   cd agepredix
   ```

2. **Set Up Virtual Environment**

   ```bash
   python3 -m venv .agevenv
   source .agevenv/bin/activate   # Linux/Mac
   .agevenv\Scripts\activate     # Windows
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Running the Application

Start the FastAPI server:

```bash
uvicorn app:app --reload
```

* **Web UI:** [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
* **API Docs (Swagger):** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* **ReDoc Docs:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 🌐 API Endpoints

| Endpoint            | Method | Description                     |
| ------------------- | ------ | ------------------------------- |
| `/predict-age`      | POST   | Predict age group from an image |
| `/predict-gender`   | POST   | Predict gender from an image    |
| `/predict-text-age` | POST   | Predict age from text input     |
| `/docs`             | GET    | Interactive Swagger API docs    |
| `/redoc`            | GET    | Alternative ReDoc API docs      |

---

## 🎨 Web Interface

The web UI (`templates/index.html`) allows you to:

* Upload an image for **age and gender predictions**.
* Input text for **age prediction based on linguistic patterns**.
* View results with **confidence scores** and detailed breakdowns.

---

## 🔬 Model Training

The `training-code.py` script contains the logic for training:

* **Age Model (`age_model.pth`)** → Neural network for age group classification.
* **Gender Model (`gender_model.pth`)** → Neural network for gender classification.
* **Text Model (`age-text.pkl` + `vectorizer.pkl`)** → NLP pipeline for text-based age predictions.

---

## 📦 Dependencies

Key dependencies listed in `requirements.txt`:

* Python **3.10+**
* **FastAPI**
* **PyTorch / torchvision**
* **scikit-learn**
* **Jinja2**
* **Uvicorn**

Install them with:

```bash
pip install -r requirements.txt
```

---

## 🔗 Live Demo

👉 Try AgePredix live at: [https://agepredix.bot.nu/](https://agepredix.bot.nu/)

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 🛡️ Badges

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Framework-brightgreen)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow)

---

Would you like me to create **copy-paste badge code blocks** 
