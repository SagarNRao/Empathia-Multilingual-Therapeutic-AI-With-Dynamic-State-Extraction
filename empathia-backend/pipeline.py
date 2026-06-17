"""
Empathia Backend Pipeline
- Groq (llama-3.3-70b) as therapist LLM
- RoBERTa zero-shot classifier for cognitive shift (affective / cognitive / agency)
- JSON session store (one file per session)
- RAG via cosine similarity on stored embeddings
- FastAPI server on port 8001
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# -- Groq ---------------------------------------------------------------------
from groq import Groq

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "YOUR_GROQ_API_KEY_HERE")
groq_client = Groq(api_key=GROQ_API_KEY)
GROQ_MODEL = "llama-3.3-70b-versatile"

# -- RoBERTa zero-shot classifier ---------------------------------------------
from transformers import pipeline as hf_pipeline

LABEL_DESCRIPTIONS = {
    "affective": "expressing emotions, feelings, distress, sadness, fear, love, anger",
    "cognitive":  "thinking, reasoning, reflecting, understanding, analyzing, wondering",
    "agency":    "taking action, making decisions, expressing control, planning, choosing",
}

print("Loading zero-shot classifier (RoBERTa)...")
classifier = hf_pipeline(
    "zero-shot-classification",
    model="cross-encoder/nli-roberta-base",
    device=-1,  # CPU; change to 0 for GPU
)
print("Classifier ready.")

# -- Sentence embeddings for RAG ----------------------------------------------
from sentence_transformers import SentenceTransformer

print("Loading sentence encoder...")
encoder = SentenceTransformer("all-MiniLM-L6-v2")
print("Encoder ready.")

# -- Session store ------------------------------------------------------------
SESSIONS_DIR = Path("sessions")
SESSIONS_DIR.mkdir(exist_ok=True)


def session_path(session_id: int) -> Path:
    return SESSIONS_DIR / f"session_{session_id}.json"


def load_session(session_id: int) -> dict:
    p = session_path(session_id)
    if p.exists():
        return json.loads(p.read_text())
    return {"session_id": session_id, "turns": []}


def save_session(session_id: int, data: dict):
    session_path(session_id).write_text(json.dumps(data, indent=2))


# -- RAG helpers --------------------------------------------------------------

def cosine_similarity(a: list, b: list) -> float:
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def retrieve_context(query: str, turns: list, top_k: int = 3) -> str:
    if not turns:
        return ""
    q_emb = encoder.encode(query).tolist()
    scored = []
    for t in turns:
        emb = t.get("embedding")
        if emb:
            score = cosine_similarity(q_emb, emb)
            scored.append((score, t))
    scored.sort(key=lambda x: x[0], reverse=True)
    snippets = []
    for score, t in scored[:top_k]:
        snippets.append(
            f"[past - {t['cognitive_label']} | sim={score:.2f}]\n"
            f"User: {t['prompt']}\nTherapist: {t['response']}"
        )
    return "\n\n".join(snippets)


# -- Cognitive shift classifier -----------------------------------------------

def classify_shift(text: str) -> dict:
    candidate_labels = list(LABEL_DESCRIPTIONS.values())
    result = classifier(text, candidate_labels, multi_label=False)
    desc_to_key = {v: k for k, v in LABEL_DESCRIPTIONS.items()}
    scores = {}
    for label, score in zip(result["labels"], result["scores"]):
        key = desc_to_key.get(label, label)
        scores[key] = round(score, 4)
    dominant = max(scores, key=scores.get)
    return {"scores": scores, "dominant": dominant}


# -- Therapist system prompt --------------------------------------------------

SYSTEM_PROMPT = """You are Empathia, a warm, compassionate AI therapist.

Guidelines:
- Listen first. Reflect the user's emotions before offering any perspective.
- Never diagnose. You are a supportive presence, not a clinician.
- Use open, curious questions to help the user explore their own thoughts.
- When the user seems stuck in emotions (affective), gently invite reflection.
- When the user is reflecting (cognitive), validate their insight and encourage agency.
- When the user is ready to act (agency), affirm their autonomy and help them plan concretely.
- Keep responses concise: 2-4 sentences unless the user shares a lot.
- Never give unsolicited advice. Wait to be asked.
- If the user expresses suicidal thoughts or crisis, respond with empathy and direct them to professional help immediately.
- Maintain a calm, non-judgmental tone at all times.
"""

# -- FastAPI app --------------------------------------------------------------

app = FastAPI(title="Empathia Pipeline")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    session_id: int
    message: str
    history: Optional[List[dict]] = []


class ChatResponse(BaseModel):
    reply: str
    cognitive_shift: dict
    extracted_notes: Optional[dict] = None


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session = load_session(req.session_id)
    turns = session["turns"]

    # 1. Classify cognitive shift
    shift = classify_shift(req.message)

    # 2. RAG context
    rag_context = retrieve_context(req.message, turns)

    # 3. Build messages for Groq
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Inject RAG context as a hidden system note
    if rag_context:
        messages.append({
            "role": "system",
            "content": (
                "Relevant past exchanges for context (do not reference these directly):\n\n"
                + rag_context
            )
        })

    # Add conversation history (last 6 turns)
    for m in (req.history or [])[-6:]:
        role = "user" if m["role"] == "user" else "assistant"
        messages.append({"role": role, "content": m["content"]})

    # Add current message
    messages.append({"role": "user", "content": req.message})

    # 4. Call Groq
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=512,
        )
        reply = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"GROQ ERROR: {type(e).__name__}: {e}")
        raise HTTPException(status_code=502, detail=f"Groq error: {e}")

    # 5. Embed the user message for future RAG
    embedding = encoder.encode(req.message).tolist()

    # 6. Persist turn
    turn = {
        "turn_num": len(turns) + 1,
        "timestamp": datetime.utcnow().isoformat(),
        "prompt": req.message,
        "response": reply,
        "cognitive_label": shift["dominant"],
        "cognitive_scores": shift["scores"],
        "embedding": embedding,
    }
    turns.append(turn)
    session["turns"] = turns
    save_session(req.session_id, session)

    return ChatResponse(
        reply=reply,
        cognitive_shift=shift,
        extracted_notes=None,
    )


@app.get("/sessions/{session_id}/cognitive-shifts")
async def get_cognitive_shifts(session_id: int):
    session = load_session(session_id)
    shifts = []
    for t in session["turns"]:
        scores = t.get("cognitive_scores", {})
        shifts.append({
            "message_num": t["turn_num"],
            "affective": scores.get("affective", 0),
            "cognitive": scores.get("cognitive", 0),
            "agency":    scores.get("agency", 0),
            "dominant":  t.get("cognitive_label", "unknown"),
            "timestamp": t.get("timestamp", ""),
            "content":   t.get("prompt", ""),
        })
    return {"shifts": shifts}


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)