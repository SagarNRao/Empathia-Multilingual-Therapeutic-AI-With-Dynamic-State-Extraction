#!/bin/bash
# Start the FastAPI server with uvicorn
echo "Starting Indy ADHD Copilot backend..."
echo ""
uvicorn main:app --reload --port 8000
