# services/translation_service/Dockerfile

FROM python:3.10-slim

# System dependencies (build essentials + git)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Clone the IndicTransToolkit
RUN git clone https://github.com/VarunGumma/IndicTransToolkit.git /app/IndicTransToolkit

# Create virtual environment (optional)
# RUN python -m venv /app/venv && . /app/venv/bin/activate

# Copy FastAPI app files
COPY main.py .
COPY requirements.txt .

# Install pip requirements
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -e ./IndicTransToolkit

# Expose port
EXPOSE 8004

# Run the translation server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8004"]