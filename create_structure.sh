#!/bin/bash

# Create root files (placeholders)
touch README.md
touch LICENSE
touch .gitignore
touch requirements.txt
touch package.json
touch docker-compose.yml
touch Dockerfile

# Create src directory and subdirs
mkdir -p src/api/routers
mkdir -p src/api/utils
touch src/api/main.py
touch src/api/routers/video.py
touch src/api/routers/graphics.py
touch src/api/routers/audio.py

mkdir -p src/models/video
mkdir -p src/models/graphics
mkdir -p src/models/audio

mkdir -p src/pipelines
touch src/pipelines/video_pipeline.py
touch src/pipelines/graphics_pipeline.py
touch src/pipelines/audio_pipeline.py

mkdir -p src/frontend/public
mkdir -p src/frontend/src/components
mkdir -p src/frontend/src/pages
touch src/frontend/src/App.js
touch src/frontend/package.json
touch src/frontend/tailwind.config.js

# Create docs
mkdir -p docs
touch docs/architecture.md
touch docs/models.md
touch docs/api.md

# Create tests
mkdir -p tests
touch tests/test_api.py
touch tests/test_pipelines.py
touch tests/test_models.py

# Create scripts
mkdir -p scripts
touch scripts/setup_env.sh
touch scripts/fine_tune_models.py
touch scripts/benchmark.py

# Create data
mkdir -p data/datasets

# Create config
mkdir -p config/model_configs
touch config/env.example

echo "Directory structure created successfully!"
