#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

echo "--- Starting Kalaa-Setu VM Setup ---"

# 1. Update system and install basic dependencies
sudo apt-get update
sudo apt-get install -y git docker.io docker-compose

# 2. Configure Docker to be used without sudo
# The default user on GCP's Ubuntu images is 'ubuntu'
sudo usermod -aG docker
echo "Docker configured for non-sudo access."

# 3. Install NVIDIA Container Toolkit for GPU access in Docker
# This allows Docker containers to see and use the NVIDIA GPU
echo "Installing NVIDIA Container Toolkit..."
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
echo "NVIDIA Container Toolkit installed and Docker restarted."

# 4. Clone the application repository
echo "Cloning the Kalaa-Setu repository..."
cd ~
git clone https://github.com/sds-ajoshi/kalaasetu-mvc.git
cd kalaasetu-mvc

# 5. Launch the application using Docker Compose
echo "Starting the application with Docker Compose..."
# Use the 'ubuntu' user to run docker-compose
# The '-d' flag runs the containers in detached mode
sudo docker-compose up -d --build

echo "--- Kalaa-Setu Setup Complete. Application is starting. ---"