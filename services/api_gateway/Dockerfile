# services/api_gateway/Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Copy and install common + service-specific requirements
COPY ./services/common_requirements.txt /app/common_requirements.txt
COPY ./services/api_gateway/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r common_requirements.txt -r requirements.txt

# Copy the specific service's code into the container
COPY ./services/api_gateway /app/services/api_gateway

# Command to run the Uvicorn server
CMD ["uvicorn", "services.api_gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]