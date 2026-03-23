# We use 3.11 to match your local setup where numpy/torch worked
FROM python:3.11-slim

WORKDIR /app
ENV ANONYMIZED_TELEMETRY=False


# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt --default-timeout=1000 --retries 15

# Copy the rest of the app
COPY . .

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}