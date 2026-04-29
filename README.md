# Indoor Plant Recommendation API

A FastAPI-based machine learning service that recommends indoor plants based on environmental conditions like temperature, humidity, and sunlight requirements.

## Features

- **Intelligent Plant Recommendations**: Uses Euclidean distance algorithm to find the most suitable plants
- **Environmental Analysis**: Considers temperature, humidity, and sunlight conditions
- **RESTful API**: Clean endpoints with proper error handling
- **Comprehensive Plant Database**: 98+ plants with detailed characteristics
- **Fast Performance**: Built with FastAPI for optimal speed

## Tech Stack

- **Backend**: FastAPI
- **Machine Learning**: scikit-learn (Euclidean distance algorithm)
- **Data Processing**: pandas, numpy
- **Validation**: Pydantic
- **Server**: Uvicorn

## Project Structure

```
indoor-plant-api/
|
|--- main.py              # Main FastAPI application
|--- plants.csv           # Plant dataset with characteristics
|--- plant_recommender.pkl # Serialized model (if exists)
|--- requirements.txt     # Python dependencies
|--- runtime.txt          # Python runtime specification
|--- README.md           # Project documentation
```

## Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd indoor-plant-api
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Usage

### Running the API

Start the development server:
```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### 1. Health Check
```http
GET /health
```
Response:
```json
{
  "status": "OK"
}
```

#### 2. Home
```http
GET /
```
Response:
```json
{
  "message": "Indoor Plant Recommendation API is online!"
}
```

#### 3. Plant Recommendation
```http
POST /predict
```
Request Body:
```json
{
  "temperature": 22.5,
  "humidity": 60.0,
  "sunlight": "part shade"
}
```

**Sunlight Options:**
- `"low light"`
- `"part shade"`
- `"full sun | part shade"`
- `"full sun"`

Response:
```json
{
  "recommended_plants": [
    {
      "id": 543,
      "common_name": "maidenhair fern",
      "type": "Fern",
      "watering": "Average",
      "sunlight": "part shade | part sun/part shade"
    },
    {
      "id": 546,
      "common_name": "delta maidenhair fern",
      "type": "Fern",
      "watering": "Average",
      "sunlight": "part shade"
    },
    {
      "id": 502,
      "common_name": "hot water plant",
      "type": "Herbs",
      "watering": "Average",
      "sunlight": "part shade"
    }
  ]
}
```

## Algorithm

The API uses **Euclidean distance** to find the most similar plants based on:

- **Temperature Range**: Plant hardiness zones (min/max)
- **Sunlight Requirements**: Mapped to numerical values (0-3)
- **Watering Needs**: Mapped to numerical values (0-2)
- **Plant Characteristics**: Indoor suitability, tropical nature

### Feature Mapping

**Sunlight Levels:**
- `"low light"`: 0
- `"part shade"`: 1
- `"full sun | part shade"`: 2
- `"full sun"`: 3

**Watering Levels:**
- `"minimum"`, `"none"`, `"low"`: 0
- `"average"`, `"medium"`: 1
- `"frequent"`, `"high"`: 2

## Dataset

The `plants.csv` contains 98+ plant records with the following columns:

- `id`: Unique identifier
- `common_name`: Plant common name
- `type`: Plant category (Fern, Herbs, Broadleaf evergreen, etc.)
- `cycle`: Plant life cycle
- `watering`: Watering frequency requirements
- `sunlight`: Sunlight requirements
- `hardiness_min/max`: Temperature hardiness zones
- `soil`: Soil type preferences
- `growth_rate`: Plant growth speed
- `maintenance`: Maintenance level
- `care_level`: Care difficulty
- `drought_tolerant`: Drought resistance
- `salt_tolerant`: Salt tolerance
- `tropical`: Tropical plant indicator
- `indoor`: Indoor suitability
- `poisonous_to_humans/pets`: Toxicity information

## API Documentation

Interactive API documentation is available at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Example Usage

### Using curl

```bash
# Health check
curl http://localhost:8000/health

# Get plant recommendations
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "temperature": 25.0,
       "humidity": 55.0,
       "sunlight": "part shade"
     }'
```

### Using Python requests

```python
import requests

# Get recommendations
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "temperature": 22.5,
        "humidity": 60.0,
        "sunlight": "part shade"
    }
)

if response.status_code == 200:
    recommendations = response.json()
    print("Recommended plants:")
    for plant in recommendations["recommended_plants"]:
        print(f"- {plant['common_name']} ({plant['type']})")
else:
    print(f"Error: {response.json()}")
```

## Error Handling

The API includes comprehensive error handling:

- **Invalid Input**: Returns descriptive error messages for invalid parameters
- **Missing Data**: Handles missing or corrupted plant data gracefully
- **Processing Errors**: Catches and reports algorithmic issues

## Performance

- **Response Time**: < 100ms for recommendation requests
- **Memory Usage**: Efficient pandas/numpy operations
- **Scalability**: Stateless design suitable for horizontal scaling

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Future Enhancements

- [ ] Add more plant characteristics to the algorithm
- [ ] Implement user preferences and history
- [ ] Add image recognition for plant identification
- [ ] Create plant care reminders and schedules
- [ ] Deploy to cloud platform (AWS, Heroku, etc.)
- [ ] Add authentication and user profiles
