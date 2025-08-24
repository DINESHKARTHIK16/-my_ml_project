from fastapi import FastAPI
import pandas as pd
import joblib
from sqlalchemy import create_engine
from fastapi.middleware.cors import CORSMiddleware

# Load trained model
model = joblib.load("fuel_blend_model.pkl")

# Connect to MySQL at global scope
engine = create_engine('mysql+pymysql://root:aids@localhost/cts')

app = FastAPI(title="Fuel Blend Prediction API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Fuel Blend Prediction API is running!"}

@app.get("/get_predictions")
def get_predictions():
    # Fetch test dataset
    df = pd.read_sql("SELECT * FROM test_data;", engine)
    
    if df.empty:
        return {"error": "Test dataset is empty"}
    
    # Drop ID if exists
    X = df.drop(columns=['ID'], errors='ignore')
    
    # Make predictions
    preds = model.predict(X)
    
    # Create new DataFrame for predictions
    pred_df = pd.DataFrame(preds, columns=[f"Predicted_BlendProperty{i+1}" for i in range(preds.shape[1])])
    
    # Combine input + predicted columns
    result_df = pd.concat([df, pred_df], axis=1)
    
    return result_df.to_dict(orient="records")
