#!/bin/bash
# Create necessary directories
mkdir -p templates
mkdir -p uploads

# Install dependencies
pip install -r requirements.txt

echo "Setup complete! Run 'python app.py' to start the Flask application." 