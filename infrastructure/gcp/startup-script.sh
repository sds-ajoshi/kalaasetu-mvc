#!/bin/bash
set -e  # Exit immediately on any error
set -o pipefail  # Catch errors in pipelines

echo "--- Starting Kalaa-Setu VM Setup ---"

# 1. Update system and install basic dependencies
echo "[1/6] Updating system and installing dependencies..."
sudo apt-get update -y
sudo apt-get install -y git docker.io docker-compose curl gnupg lsb-release

# 2. Configure Docker to be used without sudo
DOCKER_USER="ubuntu"
if groups $DOCKER_USER | grep -qv '\bdocker\b'; then
    sudo usermod -aG docker $DOCKER_USER
    echo "Added user '$DOCKER_USER' to docker group. A logout/login is required for this to take effect."
else
    echo "User '$DOCKER_USER' already in docker group."
fi

# 3. Install NVIDIA Container Toolkit (GPU support in Docker)
echo "[2/6] Installing NVIDIA Container Toolkit..."
distribution=$(. /etc/os-release; echo "$ID$VERSION_ID" | tr -d '.')
if ! dpkg -l | grep -q nvidia-container-toolkit; then
    curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
        sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    sudo apt-get update
    sudo apt-get install -y nvidia-docker2
    sudo systemctl restart docker
    echo "NVIDIA Container Toolkit installed."
else
    echo "NVIDIA Container Toolkit already installed."
fi

# 4. Clone the application repository
echo "[3/6] Cloning the Kalaa-Setu repository..."
cd /home/$DOCKER_USER
if [ ! -d "kalaasetu-mvc" ]; then
    git clone https://github.com/sds-ajoshi/kalaasetu-mvc.git
else
    echo "Repository already cloned. Pulling latest changes..."
    cd kalaasetu-mvc && git pull
fi

# 5. Optional: Export environment variables if needed (e.g., Hugging Face token)
# echo "HUGGING_FACE_TOKEN=..." >> /home/$DOCKER_USER/kalaasetu-mvc/.env

# 6. Start application with Docker Compose
echo "[4/6] Starting the application with Docker Compose..."
cd /home/$DOCKER_USER/kalaasetu-mvc

# Ensure docker-compose runs as correct user (non-root)
sudo -u $DOCKER_USER docker-compose up -d --build

echo "--- Kalaa-Setu VM Setup Complete. Application should be running. ---"