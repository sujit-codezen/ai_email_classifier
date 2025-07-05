from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import re
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "..", "model", "email_classifier.pkl")

with open(model_path, "rb") as f:
    vectorizer, model = pickle.load(f)

app = FastAPI(title="AI Email Classifier")


class EmailInput(BaseModel):
    text: str

@app.post("/predict/")
def predict_email(email: EmailInput):
    clean_text = preprocess_text(email.text)
    vect_text = vectorizer.transform([clean_text])
    prediction = model.predict(vect_text)[0]
    return {"category": prediction}

def preprocess_text(text):
    text = re.sub(r'\W', ' ', text.lower())
    return text
