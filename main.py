from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

data = pd.read_csv("plants.csv", sep=None, engine='python')
print("✅ Dataset loaded!")
print("Watering unique:", data['watering'].unique())
print("Sunlight unique:", data['sunlight'].unique())

app = FastAPI(title="Indoor Plant Recommendation API 🌿", version="1.0")

class PlantRequest(BaseModel):
    temperature: float
    humidity: float
    sunlight: str

sunlight_map = {"low light": 0, "part shade": 1, "full sun | part shade": 2, "full sun": 3}
watering_map = {"minimum": 0, "none": 0, "low": 0, "average": 1, "medium": 1, "frequent": 2, "high": 2}

@app.get("/")
def home():
    return {"message": "🌿 Indoor Plant Recommendation API is online!"}

@app.post("/predict")
def predict(req: PlantRequest):
    try:
        sunlight_val = sunlight_map.get(req.sunlight.lower().strip(), 0)
        user_vector = np.array([[req.temperature, req.humidity, sunlight_val, 1, 1, 1]])

        X = data[['hardiness_min', 'hardiness_max', 'sunlight', 'watering', 'indoor', 'tropical']].copy()
        X['sunlight'] = X['sunlight'].str.lower().str.strip().map(sunlight_map).fillna(0)
        X['watering'] = X['watering'].str.lower().str.strip().map(watering_map).fillna(1)
        X['hardiness_min'] = pd.to_numeric(X['hardiness_min'], errors='coerce').fillna(0)
        X['hardiness_max'] = pd.to_numeric(X['hardiness_max'], errors='coerce').fillna(0)
        X['indoor'] = pd.to_numeric(X['indoor'], errors='coerce').fillna(0)
        X['tropical'] = pd.to_numeric(X['tropical'], errors='coerce').fillna(0)

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

    except Exception as e:
        return {"error": str(e)}
