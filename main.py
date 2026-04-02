from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
import os

# 🌿 ---------- Load model & dataset ----------
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

# 🌿 ---------- FastAPI setup ----------
app = FastAPI(
    title="Indoor Plant Recommendation API 🌿",
    version="1.0",
    description="Get plant recommendations based on temperature, humidity, and sunlight."
)

class PlantRequest(BaseModel):
    temperature: float
    humidity: float
    sunlight: str

# Encode sunlight to numbers
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

    # ✅ Fixed: req.humidity in second position (was req.temperature twice)
    user_vector = np.array([[req.temperature, req.humidity, sunlight_val, 1, 1, 1]])

    # ✅ Fixed: encode sunlight column to numbers before distance calc
    feature_cols = ['hardiness_min', 'hardiness_max', 'sunlight', 'watering', 'indoor', 'tropical']
    X = data[feature_cols].copy()
    X['sunlight'] = X['sunlight'].map(sunlight_map).fillna(0)  # ✅ convert strings to numbers

    # Compute distances and get top 3
    distances = euclidean_distances(X, user_vector)
    top3_idx = np.argsort(distances.flatten())[:3]
    top3 = data.iloc[top3_idx][['id', 'common_name', 'type', 'watering', 'sunlight']]

    recommendations = []
    for _, row in top3.iterrows():
        recommendations.append({
            "id": int(row['id']),
            "common_name": str(row['common_name']),  # ✅ safe JSON types
            "type": str(row['type']),
            "watering": str(row['watering']),
            "sunlight": str(row['sunlight'])
        })

    return {"recommended_plants": recommendations}
```

---

### ✅ Fixed `requirements.txt`
```
numpy==2.0.2
pandas==2.2.2
scikit-learn==1.5.1
joblib==1.4.2
fastapi==0.112.2
uvicorn==0.30.5
pydantic>=2.0          # ✅ Added — required by FastAPI
python-multipart>=0.0.7  # ✅ Added — needed for form handling
