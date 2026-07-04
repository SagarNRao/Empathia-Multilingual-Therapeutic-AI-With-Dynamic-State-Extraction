# Empathia

A conversational AI therapist that understands your psychological state and remembers what matters to you.

## Features

**Cognitive State Tracking**  
Every message is classified into three psychological dimensions — affective (emotional), cognitive (reflective), and agency (action-oriented). A real-time chart visualizes your conversation arc so you can see patterns in how you think and feel.

**Long-Term Memory**  
Share a goal, fear, or important belief once — Empathia remembers it. The system quietly extracts and stores durable facts about you (goals, traumas, triggers, wins, insights, etc.) and feeds them back into future conversations so you don't have to repeat yourself.

**Context-Aware Responses**  
Behind the scenes, a RAG system surfaces relevant past exchanges based on semantic similarity, while your stored notes inform the LLM's understanding of your situation — all without overwhelming the context window.

**Warm, Non-Judgmental**  
Built on Llama-3.3-70B with a system prompt that prioritizes listening, reflection, and genuine curiosity. No unsolicited advice. No diagnosis. Just presence.

## Architecture

**Backend** (`pipeline.py`)
- **FastAPI** server with session management and CORS
- **RoBERTa** (cross-encoder) for zero-shot cognitive state classification
- **Sentence Transformers** (all-MiniLM-L6-v2) for semantic embeddings + RAG
- **Groq API** (Llama-3.3-70B) for both main chat and note extraction
- **Local JSON** session store (turns, cognitive shifts, long-term notes)

**Frontend** (`page.tsx`)
- **Next.js** (App Router) with **shadcn/ui** for component library
- Real-time chat interface with streaming message loading states
- **Recharts** visualization of cognitive shifts over time
- **Sidebar** for viewing extracted notes (goals, beliefs, traumas, etc.)

## Setup

### Backend
```bash
pip install fastapi uvicorn groq transformers sentence-transformers numpy pydantic
export GROQ_API_KEY="your_key_here"
python pipeline.py
# Runs on http://localhost:8001
```

### Frontend
```bash
cd frontend
npm install
npm run dev
# Runs on http://localhost:3000
```

The frontend talks to the backend at `http://localhost:8001`. Both must be running for the full experience.

## How It Works

1. **You send a message** → Backend classifies its cognitive flavor (affective/cognitive/agency)
2. **RAG + Notes context** → System retrieves similar past turns and feeds in your known goals/beliefs
3. **LLM responds** → Groq generates a warm, contextual reply that feels like it understands you
4. **Note extraction** → If your message contains a durable fact (a goal, a trigger, etc.), the system quietly saves it
5. **Data persists** → Everything lives in a JSON session file, so memory survives restarts

## File Breakdown

- `pipeline.py` — FastAPI + Groq + classifier + RAG + note-taking
- `page.tsx` — Main chat UI, modal for cognitive shift graph
- `cognitive-shift-chart.tsx` — Recharts visualization + turn-by-turn breakdown
- `notes-sidebar.tsx` — Displays extracted notes with category badges

---

**Note:** This is an educational project designed to explore how conversational AI can maintain long-term memory, track psychological state, and respond with empathy. It is not a replacement for professional mental health care. If you or someone you know is in crisis, please reach out to a mental health professional or crisis helpline.
