import os
import json
import joblib
import pandas as pd
import mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, root_validator
from typing import List, Optional
from datetime import datetime
import uvicorn

from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Histogram

import sys
from src import custom_transformers
sys.modules['custom_transformers'] = custom_transformers

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_NAME = "olist_model"
ALIAS = "champion"
LOCAL_CACHE_PATH = "models/champion_model_cache.joblib"
MODEL_DIR = os.getenv("MODEL_DIR", "models")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def load_model():
    try:
        print(f"Attempting to load model '{MODEL_NAME}@{ALIAS}' from MLflow...")
        model_uri = f"models:/{MODEL_NAME}@{ALIAS}"
        loaded_model = mlflow.sklearn.load_model(model_uri)

        os.makedirs("models", exist_ok=True)
        joblib.dump(loaded_model, LOCAL_CACHE_PATH)
        print("Model successfully loaded from MLflow and cached locally.")
        return loaded_model
        
    except Exception as e:
        print(f"Warning: Failed to load from MLflow ({e}). Falling back to local cache...")
        if os.path.exists(LOCAL_CACHE_PATH):
            cached_model = joblib.load(LOCAL_CACHE_PATH)
            print("Model loaded from local cache.")
            return cached_model
        else:
            print("CRITICAL: No model found in MLflow or local cache! API will start, but predictions will fail.")
            return None

pipeline = load_model()

try:
    with open(os.path.join(MODEL_DIR, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
except FileNotFoundError:
    metadata = {}
    
OPTIMAL_THRESHOLD = metadata.get("optimal_threshold", 0.5)

app = FastAPI(title="Olist Review Predictor", version=metadata.get("git_sha", "unknown")[:7])

instrumentator = Instrumentator().instrument(app).expose(app, endpoint="/metrics")

PREDICTION_PROBABILITY = Histogram(
    "model_prediction_probability",
    "Distribution of prediction probabilities",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
)

class OrderItem(BaseModel):
    order_item_id: int
    product_id: str
    seller_id: str
    price: float
    freight_value: float

class OrderPayload(BaseModel):
    order_id: str
    customer_id: str
    customer_unique_id: str
    customer_zip_code_prefix: int
    customer_state: str
    order_purchase_timestamp: datetime
    order_approved_at: Optional[datetime] = None
    order_estimated_delivery_date: datetime
    order_delivered_customer_date: datetime
    payment_count: int
    total_payment: float
    max_installments: int
    items_data: List[OrderItem] = Field(..., min_items=1)

    @root_validator(pre=True)
    def check_delivered(cls, values):
        if not values.get('order_delivered_customer_date'):
            raise ValueError("Predictions require delivered orders.")
        return values

def predict(df: pd.DataFrame) -> dict:
    probability = pipeline.predict_proba(df)[:, 1][0]
    is_bad = bool(probability >= OPTIMAL_THRESHOLD)

    PREDICTION_PROBABILITY.observe(probability)
    
    return {
        "probability": float(probability),
        "is_bad_review": is_bad,
        "threshold": OPTIMAL_THRESHOLD,
        "model_version": metadata.get("git_sha", "unknown")[:7]
    }

@app.post("/predict")
def predict_endpoint(payload: OrderPayload):
    if pipeline is None:
        raise HTTPException(
            status_code=503, 
            detail="Model is not loaded yet. Train and register a model in MLflow first."
        )

    try:
        data_dict = payload.model_dump() if hasattr(payload, 'model_dump') else payload.dict()
        df = pd.DataFrame([data_dict])
        
        date_cols = ['order_purchase_timestamp', 'order_approved_at', 
                     'order_estimated_delivery_date', 'order_delivered_customer_date']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                
        result = predict(df)
        
        return {
            "order_id": payload.order_id,
            "prediction": result
        }
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": pipeline is not None
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)