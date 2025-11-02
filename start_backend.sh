#!/bin/bash

# Backend Server - FastAPI WebSocket Server

echo "🚀 Starting Backend Server (FastAPI)"
echo "====================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
    echo ""
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip and install setuptools (for Python 3.12+ compatibility)
echo "📦 Upgrading pip and setuptools..."
pip install --upgrade pip setuptools wheel -q

# Install/update dependencies
echo "📥 Installing dependencies..."
pip install -q -r requirements.txt

# Check if installation was successful
if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed"
    echo ""
    echo "🚀 Starting FastAPI Backend Server..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📍 Backend API:    http://localhost:8000"
    echo "🔌 WebSocket:      ws://localhost:8000/ws/audio"
    echo "💚 Health Check:   http://localhost:8000/health"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Press Ctrl+C to stop the server"
    echo ""
    
    # Run the server
    python main.py
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

