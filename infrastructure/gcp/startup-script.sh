#!/bin/bash
set -e
set -o pipefail

echo "--- Starting Kalaa-Setu VM Setup ---"

# 1. Update system and install basic dependencies
echo "[1/6] Updating system and installing dependencies..."
sudo apt-get update -y
sudo apt-get install -y git docker.io docker-compose curl gnupg lsb-release python3-venv python3-pip

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

# 5. Pre-warm HuggingFace cache for IndicTrans2 models
echo "[4/6] Pre-warming IndicTrans2 models from HuggingFace..."
sudo -u $DOCKER_USER bash -c "
  python3 -m pip install --upgrade pip &&
  python3 -m pip install transformers torch sentencepiece &&
  python3 -c '
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
AutoTokenizer.from_pretrained(\"ai4bharat/indictrans2-en-indic-1B\")
AutoModelForSeq2SeqLM.from_pretrained(\"ai4bharat/indictrans2-en-indic-1B\")
AutoTokenizer.from_pretrained(\"ai4bharat/indictrans2-indic-en-1B\")
AutoModelForSeq2SeqLM.from_pretrained(\"ai4bharat/indictrans2-indic-en-1B\")
'
"

# 6. Start application with Docker Compose
echo "[5/6] Starting the application with Docker Compose..."
cd /home/$DOCKER_USER/kalaasetu-mvc
sudo -u $DOCKER_USER docker-compose up -d --build

echo "--- Kalaa-Setu VM Setup Complete. Application should be running. ---"