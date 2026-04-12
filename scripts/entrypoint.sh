#!/bin/bash
set -e

# Step 1: Prepare model (download + export if needed)
python3 /app/Raon-SpeechChat-Demo/scripts/prepare_model.py

# Step 2: Read resolved MODEL_PATH from prepare_model.py output
if [ -f /models/.model_path ]; then
    export MODEL_PATH=$(cat /models/.model_path)
fi

# Step 3: Launch worker with any extra arguments passed via CMD
exec python3 /app/Raon-SpeechChat-Demo/launch_worker.py "$@"
