#!/bin/bash
# startup-script.sh - To be used with the Terraform configuration

# Update and install basic dependencies
sudo apt-get update
sudo apt-get install -y git python3-pip ffmpeg

# Install Docker
sudo apt-get install -y docker.io
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Install Python libraries
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip3 install fastapi "uvicorn[standard]" diffusers transformers accelerate sentencepiece beautifulsoup4
pip3 install coqui_tts # For Coqui TTS
pip3 install google-cloud-translate-v2 # For language translation
pip3 install clip-score # For performance metrics

# Clone model repositories (optional, can be done inside Docker)
# git clone https://github.com/huggingface/diffusers.git
# git clone https://github.com/modelscope/modelscope.git
# ... etc.

echo "Kalaa-Setu environment setup complete."