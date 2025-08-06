# services/graphics_service/Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install them
COPY ./services/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the specific service's code into the container
COPY ./services/graphics_service /app/services/graphics_service

# Command to run the Uvicorn server
CMD ["uvicorn", "services.graphics_service.main:app", "--host", "0.0.0.0", "--port", "8001"]