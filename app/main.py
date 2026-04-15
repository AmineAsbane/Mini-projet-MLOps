from fastapi import FastAPI
import pickle
import numpy as np
from pydantic import BaseModel

class IrisInput(BaseModel):
    features: list[float]

app = FastAPI()

with open('app/model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.get("/health")
def health():
    return {"status": "ok", "message": "API is running"}

@app.post("/predict")
def predict(input_data: IrisInput):
    # Transformation de la liste en array numpy pour le modèle
    data = np.array(input_data.features).reshape(1, -1)
    prediction = model.predict(data)
    
    class_names = ['Setosa', 'Versicolor', 'Virginica']
    result = class_names[int(prediction[0])]
    
    return {"prediction": result, "class_index": int(prediction[0])}
