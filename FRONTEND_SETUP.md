# Frontend Configuration Guide

## Server Endpoints

The frontend is now properly configured to communicate with the backend servers:

### Main Chat Server
- **URL**: `http://localhost:8000`
- **Port**: 8000
- **Endpoints**:
  - `POST /chat` - Send message and get response with cognitive shifts
  - `GET /sessions/{session_id}/cognitive-shifts` - Fetch cognitive shift data for graph

### NLP Server (Internal)
- **URL**: `http://localhost:8001` (called by main server, NOT directly by frontend)
- **Port**: 8001
- **Endpoint**:
  - `POST /score` - Score text for affective/cognitive/agency values

---

## Frontend Data Flow

```
┌─────────────────────────────────────────────────┐
│  Frontend (React/Next.js - Port 3000)           │
│  - User types message                           │
│  - Click send button                            │
└──────────────────┬──────────────────────────────┘
                   │
                   ↓ POST /chat (with message + history)
┌─────────────────────────────────────────────────┐
│  Main Chat Server (FastAPI - Port 8000)         │
│  1. Receive message                             │
│  2. Call NLP Server → /score                    │
│  3. Get cognitive shift scores                  │
│  4. Call Groq API → Get LLM response            │
│  5. Extract notes from response                 │
│  6. Save session data                           │
│  7. Return reply + cognitive_shift + notes      │
└──────────────────┬──────────────────────────────┘
        ↓                              ↓
   (internal)                    (to frontend)
        ↓                              ↓
┌──────────────────────────────────────────────────────┐
│  NLP Server (PyTorch - Port 8001)                    │
│  - Receives: text                                    │
│  - Scores using fine-tuned model                     │
│  - Returns: {affective, cognitive, agency, dominant}│
└──────────────────────────────────────────────────────┘

                   ↓ Returns response
        ┌──────────────────────────────┐
        │  Frontend receives:           │
        │  - reply (LLM response)       │
        │  - cognitive_shift (scores)   │
        │  - extracted_notes            │
        └──────────────────────────────┘
```

---

## Frontend Configuration

### Server Configuration
```typescript
const CHAT_SERVER = "http://localhost:8000";
```

All frontend requests go to **Port 8000 only**. The main server internally handles:
- Calling the NLP server (port 8001) for scoring
- Calling Groq API for LLM responses
- Managing sessions and notes

### Session Management
```typescript
const SESSION_ID = 1;  // Currently hardcoded to session 1
```

### Error Handling
- **Server connection error**: Shows red indicator in header
- **API errors**: Displays error message in chat
- **Network timeouts**: 10-second timeout on NLP server calls

---

## Running the Application

### 1. Start NLP Server (Port 8001)
```bash
cd server/fineTunedModel
jupyter notebook test.ipynb
# Run cell 1 to start the server
```

### 2. Start Main Chat Server (Port 8000)
```bash
cd server
jupyter notebook main.ipynb
# Run cell 3 (uvicorn cell) to start the server
```

### 3. Start Frontend (Port 3000)
```bash
npm run dev
# Frontend will be available at http://localhost:3000
```

---

## Key Features

✅ **Server Status Indicator**
- Green pulsing dot = servers connected
- Red dot with warning = connection error

✅ **Error Messages**
- Shows specific error messages from server
- Helps debug connection issues

✅ **Cognitive Shift Graph**
- Click chart icon to view graph
- Shows 3-line visualization of affective/cognitive/agency scores over conversation

✅ **Notes Sidebar**
- Displays extracted notes from conversations
- Merges notes from multiple messages

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Server: Connection refused" | Start main server on port 8000 first |
| NLP scores not working | Start NLP server on port 8001 |
| Graph shows "No data" | Need at least 1 message exchange |
| Slow responses | Check if NLP model is still loading (first time takes ~30s) |

---

## Architecture Summary

```
Frontend → Main Server (8000) → NLP Server (8001)
    ↓           ↓                    ↓
React      FastAPI              PyTorch
Next.js    Session Mgmt         Fine-tuned
TypeScript LLM Integration      Model
Tailwind   Notes Extraction     Scoring
```

The **frontend only knows about port 8000**. Everything else happens internally! 🚀
