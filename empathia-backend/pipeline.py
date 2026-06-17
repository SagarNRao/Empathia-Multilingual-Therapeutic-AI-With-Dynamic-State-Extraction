"""
Empathia Backend Pipeline
- Gemini as therapist LLM
- RoBERTa zero-shot classifier for cognitive shift (affective / cognitive / agency)
- JSON session store (one file per session)
- RAG via cosine similarity on stored embeddings
- FastAPI server on port 8001
"""

import json
import os
import time
import math
from pathlib import Path
from datetime import datetime
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Gemini ──────────────────────────────────────────────────────────────────
import google.generativeai as genai

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# ── RoBERTa zero-shot classifier ─────────────────────────────────────────────
from transformers import pipeline as hf_pipeline

LABEL_DESCRIPTIONS = {
    "affective": "expressing emotions, feelings, distress, sadness, fear, love, anger",
    "cognitive":  "thinking, reasoning, reflecting, understanding, analyzing, wondering",
    "agency":    "taking action, making decisions, expressing control, planning, choosing",
}

print("Loading zero-shot classifier (RoBERTa)…")
classifier = hf_pipeline(
    "zero-shot-classification",
    model="cross-encoder/nli-roberta-base",   # RoBERTa NLI model
    device=-1,  # CPU; change to 0 for GPU
)
print("Classifier ready.")

# ── Sentence embeddings for RAG ───────────────────────────────────────────────
from sentence_transformers import SentenceTransformer

print("Loading sentence encoder…")
encoder = SentenceTransformer("all-MiniLM-L6-v2")
print("Encoder ready.")

# ── Session store ─────────────────────────────────────────────────────────────
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


# ── RAG helpers ───────────────────────────────────────────────────────────────

def cosine_similarity(a: list, b: list) -> float:
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def retrieve_context(query: str, turns: list, top_k: int = 3) -> str:
    """Return the top-k most similar past exchanges as a context block."""
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
            f"[past – {t['cognitive_label']} | sim={score:.2f}]\n"
            f"User: {t['prompt']}\nTherapist: {t['response']}"
        )
    return "\n\n".join(snippets)


# ── Cognitive shift classifier ─────────────────────────────────────────────────

def classify_shift(text: str) -> dict:
    """Return scores for affective / cognitive / agency."""
    candidate_labels = list(LABEL_DESCRIPTIONS.values())
    result = classifier(text, candidate_labels, multi_label=False)

    # Map back from description → label key
    desc_to_key = {v: k for k, v in LABEL_DESCRIPTIONS.items()}
    scores = {}
    for label, score in zip(result["labels"], result["scores"]):
        key = desc_to_key.get(label, label)
        scores[key] = round(score, 4)

    dominant = max(scores, key=scores.get)
    return {"scores": scores, "dominant": dominant}


# ── Therapist system prompt ────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are Empathia, a warm, compassionate AI therapist.

Guidelines:
- Listen first. Reflect the user's emotions before offering any perspective.
- Never diagnose. You are a supportive presence, not a clinician.
- Use open, curious questions to help the user explore their own thoughts.
- When the user seems stuck in emotions (affective), gently invite reflection.
- When the user is reflecting (cognitive), validate their insight and encourage agency.
- When the user is ready to act (agency), affirm their autonomy and help them plan concretely.
- Keep responses concise: 2–4 sentences unless the user shares a lot.
- Never give unsolicited advice. Wait to be asked.
- If the user expresses suicidal thoughts or crisis, respond with empathy and direct them to professional help immediately.
- Maintain a calm, non-judgmental tone at all times.
"""


# ── FastAPI app ────────────────────────────────────────────────────────────────

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

    # 3. Build Gemini prompt
    context_block = ""
    if rag_context:
        context_block = (
            "\n\n--- Relevant past exchanges (for context only, do not mention these) ---\n"
            + rag_context
            + "\n--- End of context ---\n"
        )

    # Build conversation history for Gemini
    history_text = ""
    for m in (req.history or [])[-6:]:  # last 6 turns to keep context window sane
        role = "User" if m["role"] == "user" else "Therapist"
        history_text += f"{role}: {m['content']}\n"

    full_prompt = (
        SYSTEM_PROMPT
        + context_block
        + "\n\nConversation so far:\n"
        + history_text
        + f"\nUser: {req.message}\nTherapist:"
    )

    # 4. Call Gemini
    try:
        gemini_response = gemini_model.generate_content(full_prompt)
        reply = gemini_response.text.strip()
    except Exception as e:
        print(f"GEMINI ERROR: {type(e).__name__}: {e}")  # add this line
        raise HTTPException(status_code=502, detail=f"Gemini error: {e}")

    # 5. Embed the user message for future RAG
    embedding = encoder.encode(req.message).tolist()

    # 6. Persist turn to session JSON
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
        extracted_notes=None,  # extend here if you want note extraction
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
