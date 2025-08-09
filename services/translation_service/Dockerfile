FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    python3-dev \
    git \
 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Clone IndicTransToolkit
RUN git clone https://github.com/VarunGumma/IndicTransToolkit.git /app/IndicTransToolkit

# Copy service files
COPY ./services/translation_service/main.py .
COPY ./services/common_requirements.txt ./common_requirements.txt
COPY ./services/translation_service/requirements.txt ./requirements.txt

# Install Python dependencies (common + service)
RUN pip install --no-cache-dir -r common_requirements.txt -r requirements.txt

# Install IndicTransToolkit (editable)
RUN pip install --editable ./IndicTransToolkit

# Expose port
EXPOSE 8004

# Run app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8004"]