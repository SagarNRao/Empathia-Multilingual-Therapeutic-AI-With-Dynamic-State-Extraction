# Empathia Pipeline — Setup Guide

## Architecture

```
Next.js (page.tsx)
    │
    ├── POST /chat          ─────►  pipeline.py (FastAPI :8001)
    │                                  ├── RoBERTa classifier → affective / cognitive / agency
    │                                  ├── Sentence encoder   → embed prompt
    │                                  ├── RAG retrieval      → top-k similar past turns
    │                                  ├── Gemini 1.5 Flash   → therapist response
    │                                  └── sessions/session_N.json (persisted)
    │
    └── GET /sessions/{id}/cognitive-shifts  →  chart data
```

---

## 1. Backend Setup

### Install dependencies

```bash
pip install -r requirements.txt
```

> First run downloads ~350 MB of models (RoBERTa NLI + MiniLM). Cached after that.

### Set your Gemini API key

```bash
export GEMINI_API_KEY="your-key-here"
```

Get a key at → https://aistudio.google.com/app/apikey

### Start the server

```bash
python pipeline.py
```

Server starts on **http://localhost:8001**

---

## 2. Frontend Setup

### Copy the chart component

```bash
cp cognitive-shift-chart.tsx your-nextjs-app/components/cognitive-shift-chart.tsx
```

Your `page.tsx` already imports it dynamically — no changes needed.

### Install recharts (if not already)

```bash
npm install recharts
```

---

## 3. Session JSON format

Each session is saved to `sessions/session_N.json`:

```json
{
  "session_id": 1,
  "turns": [
    {
      "turn_num": 1,
      "timestamp": "2025-06-17T10:00:00",
      "prompt": "I feel so overwhelmed",
      "response": "That sounds really heavy...",
      "cognitive_label": "affective",
      "cognitive_scores": {
        "affective": 0.82,
        "cognitive": 0.12,
        "agency": 0.06
      },
      "embedding": [0.12, -0.34, ...]   ← used for RAG
    }
  ]
}
```

---

## 4. How RAG works

On every `/chat` call:
1. The user's message is embedded with `all-MiniLM-L6-v2`
2. Cosine similarity is computed against all stored turn embeddings
3. Top 3 most similar past exchanges are injected into the Gemini prompt as hidden context
4. Gemini uses this context to maintain thematic coherence across sessions

---

## 5. Cognitive shift classification

Uses `cross-encoder/nli-roberta-base` in zero-shot mode.

Candidate labels:
| Label | Description |
|-------|-------------|
| affective | expressing emotions, feelings, distress, sadness, fear, love, anger |
| cognitive | thinking, reasoning, reflecting, understanding, analyzing, wondering |
| agency | taking action, making decisions, expressing control, planning, choosing |

The model returns a probability for each. The highest is marked `dominant`.

---

## 6. GPU acceleration (optional)

In `pipeline.py`, change:
```python
classifier = hf_pipeline(..., device=-1)   # CPU
# to
classifier = hf_pipeline(..., device=0)    # first CUDA GPU
```
