from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
import os

MODEL_PATH = "plant_recommender.pkl"
DATA_PATH = "plants.csv"

try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    model = None
    print(f"⚠️ Could not load model: {e}")

data = pd.read_csv(DATA_PATH, sep=None, engine='python')
print("✅ Dataset loaded successfully!")

app = FastAPI(
    title="Indoor Plant Recommendation API 🌿",
    version="1.0"
)

class PlantRequest(BaseModel):
    temperature: float
    humidity: float
    sunlight: str

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
    sunlight_val = sunlight_map.get(req.sunlight.lower().strip(), 0)
    user_vector = np.array([[req.temperature, req.humidity, sunlight_val, 1, 1, 1]])

    feature_cols = ['hardiness_min', 'hardiness_max', 'sunlight', 'watering', 'indoor', 'tropical']
    X = data[feature_cols].copy()
    X['sunlight'] = X['sunlight'].map(sunlight_map).fillna(0)

    distances = euclidean_distances(X, user_vector)
    top3_idx = np.argsort(distances.flatten())[:3]
    top3 = data.iloc[top3_idx][['id', 'common_name', 'type', 'watering', 'sunlight']]

    recommendations = []
    for _, row in top3.iterrows():
        recommendations.append({
            "id": int(row['id']),
            "common_name": str(row['common_name']),
            "type": str(row['type']),
            "watering": str(row['watering']),
            "sunlight": str(row['sunlight'])
        })

    return {"recommended_plants": recommendations}
