#!/bin/bash

# Frontend Server - Simple HTTP Server for Static Files

echo "🎨 Starting Frontend Server"
echo "============================"
echo ""
echo "📦 Serving static files from current directory..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🌐 Frontend URL:  http://localhost:3000"
echo "📄 Main Page:     http://localhost:3000/index.html"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "⚠️  Make sure backend is running on http://localhost:8000"
echo "   (Run ./start_backend.sh in another terminal)"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start Python's built-in HTTP server
python3 -m http.server 3000

