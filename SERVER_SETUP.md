# Dual-Server Architecture Setup

## Overview
The application now uses **2 separate servers** for better modularity:

### 1. **Fine-Tuned NLP Server** (Port 8001)
- **Location**: `server/fineTunedModel/test.ipynb`
- **Purpose**: Scores text using your fine-tuned model (affective/cognitive/agency)
- **Endpoint**: `POST /score`

### 2. **Main Chat Server** (Port 8000)
- **Location**: `server/main.ipynb`
- **Purpose**: Handles chat, LLM responses, note extraction, and session management
- **Calls**: Internally calls the NLP server for scoring
- **Endpoints**: 
  - `POST /chat` - Chat with Indy
  - `GET /sessions/{session_id}/cognitive-shifts` - Get cognitive shift graph data

---

## How to Run

### Step 1: Start the Fine-Tuned NLP Server
```bash
cd server/fineTunedModel
jupyter notebook test.ipynb
# Run cell 1 to start the server
# The server will run on http://localhost:8001
```

### Step 2: Start the Main Chat Server
In a new terminal:
```bash
cd server
jupyter notebook main.ipynb
# Run cell 3 (the uvicorn cell) to start the server
# The server will run on http://localhost:8000
```

### Step 3: Start the Frontend
In a new terminal:
```bash
npm run dev
# Frontend runs on http://localhost:3000
```

---

## Data Flow

```
Frontend (port 3000)
    ↓
Main Chat Server (port 8000)
    ├→ Calls NLP Server (port 8001) for scoring
    ├→ Calls Groq API for LLM responses
    └→ Saves session data + cognitive shifts

NLP Server (port 8001)
    └→ Scores text using fine-tuned model
```

---

## Key Files

| File | Purpose |
|------|---------|
| `server/fineTunedModel/test.ipynb` | NLP scoring server |
| `server/main.ipynb` | Main chat server |
| `server/fineTunedModel/` | Fine-tuned model weights |
| `src/app/page.tsx` | Frontend chat UI |

---

## Environment Variables

Create a `.env` file in the root directory:
```
GROQ_API_KEY=your_groq_api_key_here
```

---

## Troubleshooting

**NLP Server won't start?**
- Make sure you're in the `server/fineTunedModel` directory
- Check that port 8001 is not in use

**Main server can't connect to NLP server?**
- Ensure NLP server is running on port 8001 first
- Check that the server address is correct: `http://localhost:8001`

**Model loading slow?**
- First load downloads the model (~1.6GB) - this is normal
- Subsequent runs will be faster
