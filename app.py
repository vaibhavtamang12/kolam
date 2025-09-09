import pickle
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class InputData(BaseModel):
    features: list

# Load model at startup
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.post("/predict")
def predict(data: InputData):
    prediction = model.predict([data.features])
    return {"prediction": prediction.tolist()}
