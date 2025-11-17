# We use 3.11 to match your local setup where numpy/torch worked
FROM python:3.11-slim

WORKDIR /app

# Install system tools for building
RUN apt-get update && apt-get install -y build-essential

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --default-timeout=100

# Copy the rest of the app
COPY . .

# Start command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]