#!/bin/bash

# Function to kill backend on exit
cleanup() {
    echo "Stopping Backend..."
    kill $BACKEND_PID
}

trap cleanup EXIT

echo "Starting Deepfake Detection Backend (Port 8000)..."
python3 api.py &
BACKEND_PID=$!

# Wait for backend to be ready (optional, but good practice)
sleep 3

echo "Starting React Frontend..."
cd frontend
npm run dev
