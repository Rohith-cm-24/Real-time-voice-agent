# 🎤 AI Voice Agent: Full Voice-to-Voice Interaction

A complete **AI Voice Agent** with real-time speech recognition, LLM processing, and streaming responses. Talk to AI naturally using your voice!

**🆕 Now with WebRTC and Docker support!**

## 🎯 What It Does

```
You speak 🎤 → Deepgram STT 📝 → Groq LLM 🤖 → Deepgram TTS 🔊 → AI speaks!
```

1. **🎤 You speak** into your microphone
2. **📝 Deepgram STT** transcribes your speech in real-time  
3. **🤖 Groq LLM** generates intelligent response
4. **💬 Text streams** back character-by-character
5. **🔊 Deepgram TTS** converts response to speech
6. **👂 AI speaks** the response out loud!

## Architecture

**🔵 Backend**: FastAPI + WebSocket/WebRTC + Deepgram STT/TTS + Groq LLM  
**🎨 Frontend**: Vanilla HTML/CSS/JavaScript (zero frameworks)  
**🐳 Docker**: Full containerization support

Both run **independently** on separate ports with clean separation of concerns.

## 🚀 Dual Protocol Support

Choose between two protocols for optimal performance:

### 📡 WebSocket (Traditional)
- Reliable, widely supported
- Works everywhere
- Good for general use

### ⚡ WebRTC (Low Latency)
- **Ultra-low latency** for real-time audio
- Peer-to-peer connection
- **Better performance** for voice conversations
- Compare latency metrics side-by-side!

## Features

### Backend (FastAPI + Deepgram STT/TTS + Groq)
- ✅ **Dual Protocol Support**: WebSocket **AND** WebRTC
- ✅ **WebSocket Audio Streaming**: Real-time bidirectional audio
- ✅ **WebRTC P2P Connection**: Ultra-low latency audio streaming
- ✅ **Deepgram STT**: Real-time speech-to-text transcription
- ✅ **Groq LLM Integration**: Fast, intelligent AI responses (openai/gpt-oss-120b)
- ✅ **Deepgram TTS (Aura)**: Natural text-to-speech (AI speaks!)
- ✅ **Streaming Responses**: Real-time text and audio streaming
- ✅ **Latency Tracking**: Measure STT, LLM, and TTS latency
- ✅ **Conversation Memory**: Maintains context across interactions
- ✅ **Connection Management**: Handles multiple concurrent connections
- ✅ **Audio Processing**: Receives speech, sends speech back
- ✅ **CORS Support**: Configured for cross-origin requests
- ✅ **Health Check**: `/health` endpoint for monitoring
- ✅ **Comprehensive Logging**: Detailed debugging information
- ✅ **Docker Support**: Fully containerized deployment

### Frontend (Vanilla JS)
- ✅ **Zero Dependencies**: Pure HTML/CSS/JavaScript - no build step!
- ✅ **Protocol Selector**: Switch between WebSocket and WebRTC
- ✅ **Latency Metrics Display**: Real-time performance monitoring
- ✅ **Modern UI**: Beautiful, responsive design with gradient cards
- ✅ **Live Speech Display**: See your words as you speak (interim)
- ✅ **Streaming AI Responses**: Watch AI response type out character-by-character
- ✅ **Audio Playback**: Hear AI speak responses (Web Audio API)
- ✅ **Voice Visualization**: Shows when AI is speaking
- ✅ **Conversation View**: Clear user vs AI message display
- ✅ **Real-time Stats**: Shows chunks, bytes, duration
- ✅ **Connection Management**: Easy connect/disconnect
- ✅ **Audio Recording**: Browser MediaRecorder API
- ✅ **Message Log**: Complete conversation history
- ✅ **WebRTC Integration**: Full WebRTC peer connection support

## Project Structure

```
voice-agents/
├── Backend:
│   ├── main.py              # FastAPI WebSocket server
│   ├── requirements.txt     # Python dependencies
│   └── start_backend.sh     # Backend start script
│
├── Frontend:
│   ├── index.html           # Vanilla JS audio streaming client
│   └── start_frontend.sh    # Frontend start script
│
└── README.md               # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- **Deepgram API key** (free $200 credit at [console.deepgram.com](https://console.deepgram.com/signup))
- **Groq API key** (free tier at [console.groq.com](https://console.groq.com/))

### Setup

1. **Clone or navigate to the project directory:**
   ```bash
   cd /home/rohith/Documents/voice-agents
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Linux/Mac
   # or
   venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API Keys:**
   ```bash
   # Create .env file
   nano .env
   # or
   code .env
   ```
   
   Add both API keys:
   ```env
   DEEPGRAM_API_KEY=your_deepgram_api_key_here
   GROQ_API_KEY=your_groq_api_key_here
   ```
   
   📖 **Detailed Setup**: 
   - [DEEPGRAM_SETUP.md](DEEPGRAM_SETUP.md) - Deepgram STT setup
   - [GROQ_SETUP.md](GROQ_SETUP.md) - Groq LLM integration guide
   - [QUICK_START_VOICE_AGENT.md](QUICK_START_VOICE_AGENT.md) - **Start here!**

## Quick Start

### 🐳 Option 1: Docker (Recommended - Zero Setup!)

```bash
cd /home/rohith/Documents/voice-agents

# Create .env file with your API keys
echo "DEEPGRAM_API_KEY=your_key_here" >> .env
echo "GROQ_API_KEY=your_key_here" >> .env

# Start everything with Docker Compose
docker-compose up
```

Then open: **http://localhost:8080/index.html**

- Backend: `http://localhost:8000`
- Frontend: `http://localhost:8080`
- No Python installation needed!
- No dependency issues!

### 📦 Option 2: Manual Setup

**Terminal 1 - Backend:**
```bash
cd /home/rohith/Documents/voice-agents
./start_backend.sh
```

**Terminal 2 - Frontend:**
```bash
cd /home/rohith/Documents/voice-agents
./start_frontend.sh
```

Then open your browser: **http://localhost:3000/index.html**

---

### Backend Only

Start the FastAPI WebSocket server:

```bash
./start_backend.sh
```

Or manually:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

**Backend runs on**: `http://localhost:8000`

### Frontend Only

Start the frontend HTTP server:

```bash
./start_frontend.sh
```

Or manually:
```bash
python3 -m http.server 3000
```

**Frontend runs on**: `http://localhost:3000`

### Backend API Endpoints

| Endpoint | Type | Description |
|----------|------|-------------|
| `/` | GET | API information and status |
| `/health` | GET | Health check endpoint |
| `/ws/audio` | WebSocket | WebSocket audio streaming endpoint |
| `/ws/webrtc` | WebSocket | WebRTC signaling channel |
| `/webrtc/offer` | POST | WebRTC SDP offer/answer exchange |

---

## 📊 Latency Comparison: WebSocket vs WebRTC

The UI now displays real-time latency metrics for both protocols:

| Metric | Description | Typical Range |
|--------|-------------|---------------|
| **STT Latency** | Speech-to-text processing time | 100-500ms |
| **LLM Latency** | Time to first token from LLM | 200-800ms |
| **TTS Latency** | Text-to-speech generation time | 300-1000ms |
| **Total Pipeline** | End-to-end response time | 600-2300ms |

**WebRTC typically shows 20-50% lower latency** compared to WebSocket due to:
- Direct peer-to-peer audio path
- Lower protocol overhead
- Optimized audio codecs
- Better network utilization

Try both protocols and see the difference in your browser!

---

## Usage

### Using the Frontend

1. **Start both servers** (see Quick Start above)
2. Open **http://localhost:3000/index.html** (or **http://localhost:8080/index.html** for Docker) in your browser
3. **Select Protocol**: Choose between WebSocket or WebRTC in the dropdown
4. Click **"Connect"** to establish connection
5. Click **"Start Recording"** to begin streaming audio
6. **Speak into your microphone** 🎤 (e.g., "What is Python?")
7. Watch **real-time transcriptions** appear in the transcript box!
8. See AI's **text response** stream in character-by-character
9. **Hear AI speak** the response out loud! 🔊👂
10. **Monitor latency metrics** in the metrics panel (STT, LLM, TTS)
11. Have a natural voice conversation!
12. Try **switching protocols** to compare latency!
13. Click **"Stop Recording"** when done

### 📝 Where to See Transcripts

Transcripts appear in **TWO places**:

1. **📝 Live Transcription Box** (prominent display above buttons)
   - Shows accumulated transcripts as you speak
   - Updates in real-time
   - Large, easy-to-read text

2. **Messages Log** (bottom of page)
   - Shows transcript messages with 💬 icon
   - Includes all other server messages
   - Scrollable history

### Custom Frontend Integration

```javascript
// Connect to WebSocket
const websocket = new WebSocket('ws://localhost:8000/ws/audio');

websocket.onopen = () => {
    console.log('Connected to audio server');
};

websocket.onmessage = (event) => {
    console.log('Server message:', event.data);
};

// Start recording
const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
const mediaRecorder = new MediaRecorder(stream);

mediaRecorder.ondataavailable = (event) => {
    if (event.data.size > 0) {
        websocket.send(event.data);  // Send audio chunk
    }
};

// Start recording with 250ms chunks
mediaRecorder.start(250);

// Send control messages
websocket.send('start_recording');
websocket.send('stop_recording');
```

## WebSocket Protocol

### Data Types

The WebSocket endpoint accepts two types of messages:

1. **Binary Messages (Audio Data)**
   - Send audio chunks as binary data
   - Server logs chunk size and count
   - Server sends acknowledgment every 10 chunks

2. **Text Messages (Control)**
   - `start_recording` - Notify server that recording has started
   - `stop_recording` - Notify server that recording has stopped
   - `ping` - Health check (server responds with "pong")
   - Any other text - Server echoes back

### Example Flow

```
Client → Server: [Connection]
Server → Client: "Connected to audio streaming server"

Client → Server: "start_recording" (text)
Server → Client: "Server ready to receive audio"

Client → Server: [Binary audio chunk 1]
Client → Server: [Binary audio chunk 2]
... (continues streaming)

Server → Client: "Processed 10 chunks, 245760 bytes total"

Client → Server: "stop_recording" (text)
Server → Client: "Recording complete. Received X chunks, Y bytes"
```

## Server Configuration

### Environment Variables

You can configure the server using environment variables:

```bash
export HOST="0.0.0.0"
export PORT="8000"
```

### Audio Processing

The server currently logs audio chunks. To process audio data, modify the WebSocket endpoint in `main.py`:

```python
if "bytes" in data:
    audio_data = data["bytes"]
    
    # Your processing logic here:
    # - Save to file
    # - Send to speech recognition API
    # - Process with ML model
    # - Stream to another service
```

### Example: Save Audio to File

```python
# In the websocket endpoint
if audio_chunks_received == 1:
    # Create new file for each recording session
    audio_file = open(f"recording_{datetime.now().timestamp()}.webm", "wb")

if "bytes" in data:
    audio_data = data["bytes"]
    audio_file.write(audio_data)

# When recording stops
if message == "stop_recording":
    audio_file.close()
```

## Development

### Running Backend in Development Mode

```bash
source venv/bin/activate
uvicorn main:app --reload --log-level debug
```

### Frontend Development

The frontend uses **zero frameworks** - just vanilla JavaScript. To modify:

1. Edit `index.html` directly
2. Refresh browser to see changes
3. No build step required!

### Adding Custom Audio Processing

1. Modify the `websocket_audio_endpoint` function in `main.py`
2. Add your audio processing logic in the `if "bytes" in data:` block
3. Use the `manager` to send responses back to clients

### Logging

The server uses Python's logging module. Adjust log level in `main.py`:

```python
logging.basicConfig(level=logging.DEBUG)  # More verbose
logging.basicConfig(level=logging.WARNING)  # Less verbose
```

## 🐳 Production Deployment with Docker

### Quick Docker Compose Deployment

The project includes a complete Docker Compose setup:

```bash
cd /home/rohith/Documents/voice-agents

# Create .env file with API keys
cat > .env << EOF
DEEPGRAM_API_KEY=your_deepgram_key_here
GROQ_API_KEY=your_groq_key_here
EOF

# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

**Services:**
- Backend: `http://localhost:8000` (FastAPI + WebSocket + WebRTC)
- Frontend: `http://localhost:8080` (Static HTTP server)

### Manual Docker Build

```bash
# Build backend image
docker build -t voice-agent-backend .

# Run backend container
docker run -d \
  -p 8000:8000 \
  -e DEEPGRAM_API_KEY=your_key \
  -e GROQ_API_KEY=your_key \
  --name voice-backend \
  voice-agent-backend

# Run frontend container
docker run -d \
  -p 8080:8080 \
  -v $(pwd)/index.html:/app/index.html \
  --name voice-frontend \
  python:3.11-slim \
  python -m http.server 8080
```

### Using Gunicorn (Alternative)

```bash
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Nginx Reverse Proxy

Example Nginx configuration for WebSocket:

```nginx
location /ws/audio {
    proxy_pass http://localhost:8000;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
}
```

## Troubleshooting

### Common Issues

**1. Microphone access denied**
- Ensure you're using HTTPS (or localhost for development)
- Check browser permissions for microphone access

**2. WebSocket connection failed**
- Verify the server is running
- Check firewall settings
- Ensure correct WebSocket URL (ws:// not http://)

**3. No audio data received**
- Check browser console for MediaRecorder errors
- Verify audio codec support (webm/ogg)
- Test microphone with another application

**4. Connection drops frequently**
- Increase server timeout settings
- Check network stability
- Monitor server logs for errors

### Debug Mode

Enable detailed logging:

```python
# In main.py
logging.basicConfig(level=logging.DEBUG)
```

Check browser console:
- Press F12 to open developer tools
- Check Console and Network tabs
- Look for WebSocket connection status

## API Response Examples

### GET /
```json
{
  "message": "Voice Agent Audio Streaming Server",
  "version": "1.0.0",
  "websocket_endpoint": "/ws/audio",
  "status": "running"
}
```

### GET /health
```json
{
  "status": "healthy",
  "timestamp": "2025-11-01T10:30:00.123456",
  "active_connections": 2
}
```

## License

MIT License - Feel free to use this in your projects!

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests.

## Future Enhancements

- [x] ~~WebRTC support~~ **✅ IMPLEMENTED!**
- [x] ~~Real-time transcription~~ **✅ IMPLEMENTED!**
- [x] ~~Latency tracking~~ **✅ IMPLEMENTED!**
- [x] ~~Docker containerization~~ **✅ IMPLEMENTED!**
- [ ] Audio format conversion
- [ ] Integration with additional speech services (Whisper, Google Speech-to-Text)
- [ ] Audio buffer management for long recordings
- [ ] Support for multiple audio codecs
- [ ] Authentication and authorization
- [ ] Rate limiting
- [ ] Advanced audio compression
- [ ] Multi-user voice rooms
- [ ] Voice activity detection optimization
- [ ] Custom LLM model selection in UI

## Support

For issues or questions, please create an issue in the project repository.

---

**Built with ❤️ using:**
- FastAPI
- WebSockets & WebRTC
- Deepgram STT & TTS (Aura)
- Groq LLM (openai/gpt-oss-120b)
- Docker
- Vanilla JavaScript (zero frameworks!)

