from fastapi import FastAPI
import joblib
from pydantic import BaseModel


app = FastAPI()

class InputData(BaseModel):
    store: int
    day_of_week: int
    promo: int

# Load model at startup
model = joblib.load("models/rf_200.pkl")

@app.get("/")
def home():
    return {"message": "Retail AI Copilot API is running"}

@app.get("/predict")
def predict(data: InputData):
    prediction = model.predict([[data.store, data.day_of_week, data.promo]])
    
    return {
        "store": data.store,
        "prediction": float(prediction[0])
    }