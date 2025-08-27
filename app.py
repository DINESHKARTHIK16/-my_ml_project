from fastapi import FastAPI
import pandas as pd
import joblib
import pymysql  # Added import for the MySQL driver
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os

# Define the model and data file paths
MODEL_PATH = "model_compressed.pkl"
TEST_DATA_PATH = "test.csv"

# Load the model at global scope
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    raise RuntimeError(f"Model file not found at: {MODEL_PATH}")

app = FastAPI(title="Fuel Blend Prediction API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mapping of model output to property names
# You will need to verify this order based on your model's training
PREDICTED_PROPERTY_NAMES = [
    'Predicted_BlendProperty_Density',
    'Predicted_BlendProperty_Viscosity',
    'Predicted_BlendProperty_FlashPoint',
    'Predicted_BlendProperty_Octane',
    'Predicted_BlendProperty_Sulfur',
    'Predicted_BlendProperty_Cetane',
    # Add other properties as needed based on your model's 10 outputs
    'Predicted_BlendProperty_7',
    'Predicted_BlendProperty_8',
    'Predicted_BlendProperty_9',
    'Predicted_BlendProperty_10',
]



@app.get("/")
def read_root():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))

@app.get("/get_predictions")
def get_predictions():
    # Fetch test dataset from a local file instead of a database
    try:
        df = pd.read_csv(TEST_DATA_PATH)
    except FileNotFoundError:
        return {"error": f"Test data file not found at: {TEST_DATA_PATH}"}
    
    if df.empty:
        return {"error": "Test dataset is empty"}
    
    # Drop columns that are not features (like ID and target properties)
    features_to_predict = df.drop(
        columns=[f'ID'] + [f'BlendProperty{i}' for i in range(1, 11)], 
        errors='ignore'
    )
    
    # Make predictions
    preds = model.predict(features_to_predict)
    
    # Create new DataFrame for predictions with correct column names
    if preds.shape[1] != len(PREDICTED_PROPERTY_NAMES):
        return {"error": f"Model output shape ({preds.shape[1]}) does not match the number of expected property names ({len(PREDICTED_PROPERTY_NAMES)})."}
    
    pred_df = pd.DataFrame(preds, columns=PREDICTED_PROPERTY_NAMES)
    
    # Combine input + predicted columns
    result_df = pd.concat([df, pred_df], axis=1)
    

    return result_df.to_dict(orient="records")
