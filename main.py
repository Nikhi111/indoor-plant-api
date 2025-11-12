from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

# 🌿 Load trained model and datasetc:\Users\NIKHIL SHINDE\Downloads\plants.csv
model = joblib.load("plant_recommender.pkl")
data = pd.read_csv("plants.csv", sep=None, engine='python')

app = FastAPI(title="Indoor Plant Recommendation API 🌿", version="1.0")

# Input request format
class PlantRequest(BaseModel):
    temperature: float
    humidity: float
    sunlight: str

# Encode sunlight manually (adjust to match your dataset values)
sunlight_map = {
    "low light": 0,
    "part shade": 1,
    "full sun | part shade": 2,
    "full sun": 3
}

@app.get("/")
def home():
    return {"message": "🌿 Indoor Plant Recommendation API is online!"}

@app.post("/predict")
def predict(req: PlantRequest):
    sunlight_val = sunlight_map.get(req.sunlight.lower(), 0)
    user_vector = np.array([[req.temperature, req.temperature, sunlight_val, 1, 1, 1]])

    # Find most similar plants using distance similarity
    X = data[['hardiness_min', 'hardiness_max', 'sunlight', 'watering', 'indoor', 'tropical']]
    distances = euclidean_distances(X, user_vector)
    top3_idx = np.argsort(distances.flatten())[:3]
    top3 = data.iloc[top3_idx][['id', 'common_name', 'type', 'watering', 'sunlight']]

    recommendations = []
    for _, row in top3.iterrows():
        recommendations.append({
            "id": int(row['id']),
            "common_name": row['common_name'],
            "type": row['type'],
            "watering": row['watering'],
            "sunlight": row['sunlight']
        })
    return {"recommended_plants": recommendations}
