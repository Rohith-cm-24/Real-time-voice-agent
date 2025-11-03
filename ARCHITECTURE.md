# Voice Agent Architecture

## 📐 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (HTML/JS)                      │
│                  Dark-themed Voice Agent Dashboard              │
└────────────────┬──────────────────────────────┬─────────────────┘
                 │                              │
                 │ WebSocket                    │ WebRTC
                 │ /ws/audio                    │ /webrtc/offer
                 │                              │ /ws/webrtc/{id}
                 ▼                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Application                          │
│                    (main.py)                        │
└─────────────┬───────────────────────────────┬───────────────────┘
              │                               │
              │ WebSocket Handler             │ WebRTC Handler
              ▼                               ▼
┌──────────────────────────────┐   ┌─────────────────────────────┐
│ WebSocketAudioHandler        │   │ AudioTransformTrack         │
│ (websocket_audio_handler.py) │   │ (webrtc_handler.py)        │
└──────────────┬───────────────┘   └──────────────┬──────────────┘
               │                                   │
               │                                   │
               └──────────┬────────────────────────┘
                          │
                          │ Uses Services
                          ▼
         ┌────────────────────────────────────────┐
         │        Service Layer                   │
         │                                        │
         │  ┌──────────────────────────────────┐ │
         │  │   DeepgramService                │ │
         │  │   (deepgram_service.py)          │ │
         │  │   - STT (Speech-to-Text)         │ │
         │  │   - TTS (Text-to-Speech)         │ │
         │  └──────────────────────────────────┘ │
         │                                        │
         │  ┌──────────────────────────────────┐ │
         │  │   LLMService                     │ │
         │  │   (llm_service.py)               │ │
         │  │   - Groq LLM Integration         │ │
         │  │   - Conversation Management      │ │
         │  └──────────────────────────────────┘ │
         └────────────────────────────────────────┘
                          │
                          │ Uses Models & Config
                          ▼
         ┌────────────────────────────────────────┐
         │        Core & Models                   │
         │                                        │
         │  ┌──────────────────────────────────┐ │
         │  │   ConnectionManager              │ │
         │  │   (websocket_manager.py)         │ │
         │  │   - Connection Lifecycle         │ │
         │  └──────────────────────────────────┘ │
         │                                        │
         │  ┌──────────────────────────────────┐ │
         │  │   Settings                       │ │
         │  │   (config.py)                    │ │
         │  │   - Configuration                │ │
         │  │   - Environment Variables        │ │
         │  └──────────────────────────────────┘ │
         │                                        │
         │  ┌──────────────────────────────────┐ │
         │  │   Models                         │ │
         │  │   (session.py)                   │ │
         │  │   - LatencyMetrics               │ │
         │  │   - WebRTCSession                │ │
         │  │   - AudioStats                   │ │
         │  └──────────────────────────────────┘ │
         └────────────────────────────────────────┘
                          │
                          │ External APIs
                          ▼
         ┌────────────────────────────────────────┐
         │        External Services               │
         │                                        │
         │  ┌──────────────┐  ┌────────────────┐ │
         │  │  Deepgram    │  │  Groq (LLM)    │ │
         │  │  STT + TTS   │  │  Llama 3.3     │ │
         │  └──────────────┘  └────────────────┘ │
         └────────────────────────────────────────┘
```

## 🔄 Data Flow

### **Voice Input → AI Response Flow**

```
1. User Speaks
   ↓
2. Browser MediaRecorder / WebRTC captures audio
   ↓
3. Audio sent via WebSocket or WebRTC
   ↓
4. Handler receives audio data
   ↓
5. DeepgramService sends to Deepgram STT API
   ↓
6. Transcript received (interim + final)
   ↓
7. Final transcript sent to LLMService
   ↓
8. LLMService queries Groq API (streaming)
   ↓
9. LLM response streamed to frontend (text)
   ↓
10. Each sentence sent to DeepgramService for TTS
    ↓
11. Audio chunks sent back to frontend
    ↓
12. Browser plays audio response
```

## 🏗️ Module Dependencies

```
main_refactored.py
├── handlers/
│   ├── webrtc_handler.py
│   │   ├── services/deepgram_service.py
│   │   ├── services/llm_service.py
│   │   ├── core/websocket_manager.py
│   │   └── models/session.py
│   └── websocket_audio_handler.py
│       ├── services/deepgram_service.py
│       ├── services/llm_service.py
│       ├── core/websocket_manager.py
│       └── models/session.py
├── services/
│   ├── deepgram_service.py
│   │   ├── core/config.py
│   │   └── models/session.py
│   └── llm_service.py
│       ├── core/config.py
│       └── models/session.py
├── core/
│   ├── websocket_manager.py
│   └── config.py
└── models/
    └── session.py
```

## 🎭 Component Roles

### **Presentation Layer**

- **main_refactored.py**: FastAPI routes and endpoints
- **Frontend (index.html)**: User interface

### **Handler Layer**

- **webrtc_handler.py**: Processes WebRTC audio streams
- **websocket_audio_handler.py**: Processes WebSocket audio streams

### **Service Layer**

- **deepgram_service.py**: Deepgram API abstraction
- **llm_service.py**: Groq LLM API abstraction

### **Core Layer**

- **websocket_manager.py**: Connection management
- **config.py**: Configuration and settings

### **Data Layer**

- **session.py**: Data models and structures

## 📊 Latency Tracking Flow

```
┌───────────────────────────────────────────────────────┐
│                 LatencyMetrics                        │
│                                                       │
│  audio_received ────────────────┐                    │
│                                  │                    │
│  transcript_received ────────────┼─→ STT Latency     │
│                                  │                    │
│  llm_start ──────────────────────┼─┐                 │
│                                  │ │                 │
│  llm_first_token ────────────────┼─┼─→ LLM Latency  │
│                                  │ │                 │
│  tts_start ──────────────────────┼─┼─┐               │
│                                  │ │ │               │
│  tts_received ───────────────────┼─┼─┼─→ TTS Latency│
│                                  │ │ │               │
│  Total = STT + LLM + TTS ────────┴─┴─┴───────────────│
└───────────────────────────────────────────────────────┘
```

## 🔐 Security & Configuration

```
Environment Variables (.env)
├── DEEPGRAM_API_KEY ──→ deepgram_service.py
├── GROQ_API_KEY ──────→ llm_service.py
├── HOST ──────────────→ main_refactored.py
├── PORT ──────────────→ main_refactored.py
└── LOG_LEVEL ─────────→ All modules

Configuration Flow:
.env → load_dotenv() → Settings class → Service instances
```

## 🎯 Design Patterns Used

### **1. Service Pattern**

- Services encapsulate external API interactions
- Clean interfaces for business logic
- Easy to mock for testing

### **2. Handler Pattern**

- Handlers process specific types of requests
- Coordinate between services
- Manage request lifecycle

### **3. Dependency Injection**

- Services receive configuration via constructor
- Handlers receive dependencies (websocket, manager)
- Promotes testability and flexibility

### **4. Strategy Pattern**

- Different handlers for WebSocket vs WebRTC
- Same service interfaces used by both
- Easy to add new streaming protocols

### **5. Observer Pattern**

- Deepgram event handlers (on_open, on_message, etc.)
- Callback-based architecture
- Loose coupling between components

## 📈 Scalability Considerations

### **Current Architecture**

- Single-process application
- In-memory session storage
- Suitable for: Development, demos, small deployments

### **Production Enhancements**

```
┌─────────────────────────────────────────────────┐
│ Load Balancer (nginx)                           │
└───────────┬─────────────────────────────────────┘
            │
    ┌───────┼───────┐
    │               │
    ▼               ▼
┌─────────┐   ┌─────────┐
│ Worker 1│   │ Worker 2│  (Multiple uvicorn workers)
└────┬────┘   └────┬────┘
     │             │
     └──────┬──────┘
            │
            ▼
     ┌──────────────┐
     │    Redis     │  (Session storage)
     │  (Shared)    │
     └──────────────┘
            │
            ▼
     ┌──────────────┐
     │  PostgreSQL  │  (Conversation history)
     └──────────────┘
```

## 🧪 Testing Strategy

### **Unit Tests**

```python
# Test individual services
test_deepgram_service.py
test_llm_service.py
test_websocket_manager.py
test_session_models.py
```

### **Integration Tests**

```python
# Test handlers with mocked services
test_webrtc_handler.py
test_websocket_audio_handler.py
```

### **End-to-End Tests**

```python
# Test full workflows
test_voice_pipeline.py
test_websocket_flow.py
test_webrtc_flow.py
```

## 🔄 Future Enhancements

1. **Add database layer** for persistent storage
2. **Implement caching** for repeated queries
3. **Add rate limiting** per user/session
4. **Implement message queues** (RabbitMQ, Kafka)
5. **Add monitoring** (Prometheus, Grafana)
6. **Implement authentication** (JWT, OAuth)
7. **Add multi-language support**
8. **Implement conversation branching**

---

**Architecture Version**: 4.0.0  
**Last Updated**: November 3, 2025
