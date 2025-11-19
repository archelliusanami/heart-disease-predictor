
# ❤️ Heart Disease Prediction API

This project provides a **machine learning-powered API** for predicting heart disease using clinical features.  
It uses **FastAPI**, **scikit-learn**, and **Docker** to create a production-ready prediction service.

The repository includes:

- 📘 Jupyter notebook for model development  
- 🧠 Python scripts for training  
- 🤖 A trained model (`.pkl`)  
- 🚀 A FastAPI application (`heart_app.py`)  
- 🐳 Dockerfile for containerized deployment  
- 📦 Requirements file for reproducible environments  

---

## 📁 Project Structure

heart-disease-predictor/
│
├── .dockerignore # Files to ignore when building Docker
├── Dockerfile # Docker instructions for deployment
├── README.md # Project documentation
│
├── cardivascular disease model_training.ipynb # Notebook for EDA & model training
├── cardivascular disease model_training.py # Python version of the notebook
├── cardivascular_disease_train_model.py # Script to train and export the model
│
├── heart_app.py # FastAPI application
├── heart_disease_model.pkl # Trained ML model (main)
├── heart_disease_model1.pkl # Additional/earlier saved model
│
└── requirements.txt # Required Python packages


---

## 🎯 Objective

To build a **REST API** that predicts whether a patient is likely to have heart disease based on 14 clinically relevant features, including:

- Age  
- Sex  
- Chest pain type  
- Resting blood pressure  
- Cholesterol  
- Fasting blood sugar  
- Resting ECG  
- Max heart rate  
- Exercise-induced angina  
- ST depression (oldpeak)  
- Slope  
- Number of major vessels  

---

## 🚀 Features

- ✔ **Machine learning prediction** (0 = No Disease, 1 = Disease)
- ✔ **Probability output** (`predict_proba`)
- ✔ **Input validation** using Pydantic  
- ✔ **FastAPI interactive docs** (`/docs`)
- ✔ **Docker support** for easy deployment
- ✔ Well-organized training scripts and notebooks

---

## 🧠 Model Details

- **Algorithm:** XGBoost (or best-performing model in training)
- **Dataset size:** 1,000 subjects, 14 features
- **Metrics Achieved:**  
  *(Add your real numbers if you want)*

- Accuracy: _e.g., 0.98_  
- ROC-AUC: _e.g., 0.99_  
- F1 Score: _e.g., 0.97_  

---

## 🧪 API Usage

### **Start the API locally**
build the docker image from docker file then run 

uvicorn heart_app:app --reload

http://127.0.0.1:8000/docs



