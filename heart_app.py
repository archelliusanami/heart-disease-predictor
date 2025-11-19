from fastapi import FastAPI
from pydantic import BaseModel, conint, confloat, Field
import joblib
import numpy as np

# Load the trained model
model = joblib.load('heart_disease_model1.pkl')

# Initialize FastAPI app
app = FastAPI(title="Heart Disease Prediction API")

# Define input data model


class HeartDiseaseInput(BaseModel):
    age: conint(ge=0, le=120) = Field(..., description="Age of the patient in years (0-120)")
    gender: conint(ge=0, le=1) = Field(..., description="Gender of the patient: 0 = female, 1 = male")
    chestpain: conint(ge=0, le=3) = Field(..., description="Type of chest pain (0-3)")
    restingBp: conint(ge=50, le=250) = Field(..., description="Resting blood pressure in mmHg")
    serumcholesterol: conint(ge=100, le=600) = Field(..., description="Serum cholesterol in mg/dl")
    fastingbloodsugar: conint(ge=0, le=1) = Field(..., description="Fasting blood sugar > 120 mg/dl (0 = false, 1 = true)")
    restingrelectro: conint(ge=0, le=2) = Field(..., description="Resting electrocardiographic results (0-2)")
    maxheartrate: conint(ge=60, le=220) = Field(..., description="Maximum heart rate achieved")
    exerciseinducedangina: conint(ge=0, le=1) = Field(..., description="Exercise-induced angina (0 = no, 1 = yes)")
    oldpeak: confloat(ge=0.0, le=10.0) = Field(..., description="ST depression induced by exercise relative to rest")
    slope: conint(ge=0, le=2) = Field(..., description="Slope of the peak exercise ST segment (0-2)")
    nonfmajorvessels: conint(ge=0, le=3) = Field(..., description="Number of major vessels colored by fluoroscopy (0-3)")

    
@app.post("/predict")
def predict_heart_disease(data: HeartDiseaseInput):
    # Convert input data to numpy array
    input_data = np.array([[data.age, data.gender, data.chestpain, data.restingBp,
                            data.serumcholesterol, data.fastingbloodsugar, data.restingrelectro,
                            data.maxheartrate, data.exerciseinducedangina, data.oldpeak, data.slope,
                            data.nonfmajorvessels]])
    # Make prediction
    prediction = model.predict(input_data)
    # Return prediction result
    result = "Presence of Heart Disease" if prediction[0] == 1 else "No Heart Disease" 
    return {"prediction": result}   
