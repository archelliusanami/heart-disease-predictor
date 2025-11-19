# Use a lightweight official Python image
FROM python:3.10-slim

# Create working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your FastAPI application code
COPY heart_app.py .

# If you have a model file like model.pkl, include it:
COPY heart_disease_model1.pkl .

# Expose port for FastAPI
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "heart_app:app", "--host", "0.0.0.0", "--port", "8000"]
