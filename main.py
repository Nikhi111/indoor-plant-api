from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
import os

# 🌿 ---------- Safe model & dataset loading ----------

MODEL_PATH = "plant_recommender.pkl"
DATA_PATH = "plants.csv"

# If running locally, adjust dataset path if needed
if not os.path.exists(DATA_PATH):
    # Optional fallback for local testing (Windows path)
    local_path = r"C:\Users\NIKHIL SHINDE\Downloads\plants.csv"
    if os.path.exists(local_path):
        DATA_PATH = local_path
    else:
        raise FileNotFoundError("❌ plants.csv not found. Please upload it to your Render repo or local directory.")

# Try loading model safely
try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"⚠️ Could not load model ({e}). Please retrain or upload 'plant_recommender.pkl'.")

# Load dataset
data = pd.read_csv(DATA_PATH, sep=None, engine='python')
print("✅ Dataset loaded successfully!")

# 🌿 ---------- FastAPI setup ----------

app = FastAPI(
    title="Indoor Plant Recommendation API 🌿",
    version="1.0",
    description="Get the best indoor plant recommendations based on temperature, humidity, and sunlight."
)

# 🌤 Input request format
class PlantRequest(BaseModel):
    temperature: float
    humidity: float
    sunlight: str

# 🌞 Encode sunlight manually (should match your dataset)
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
    # Encode sunlight safely
    sunlight_val = sunlight_map.get(req.sunlight.lower().strip(), 0)

    # User feature vector (same structure as training features)
    user_vector = np.array([[req.temperature, req.temperature, sunlight_val, 1, 1, 1]])

    # Extract feature subset from dataset
    feature_cols = ['hardiness_min', 'hardiness_max', 'sunlight', 'watering', 'indoor', 'tropical']
    X = data[feature_cols]

    # Compute similarity distances
    distances = euclidean_distances(X, user_vector)
    top3_idx = np.argsort(distances.flatten())[:3]
    top3 = data.iloc[top3_idx][['id', 'common_name', 'type', 'watering', 'sunlight']]

    # Prepare response
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
