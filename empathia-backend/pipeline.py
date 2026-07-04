import json
import os
import re
from pathlib import Path
from datetime import datetime
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# -- Groq ---------------------------------------------------------------------
from groq import Groq

# GROQ_API_KEY = os.getenv("GROQ_API_KEY", "YOUR_GROQ_API_KEY_HERE")
GROQ_API_KEY = "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"
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
        data = json.loads(p.read_text())
        data.setdefault("notes", {})
        return data
    return {"session_id": session_id, "turns": [], "notes": {}}


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


# -- Long-term notes ----------------------------------------------------------
# Notes are a small, evolving memory of durable facts about the user
# (goals, traumas, beliefs, wins, etc). They live inside the session's
# "notes" dict, keyed by a slug of the note title, and get folded into the
# RAG context on every turn so the LLM can use them without the user having
# to repeat themselves. They're only written to when the extraction model
# judges a message actually contains something worth remembering.

NOTE_CATEGORIES = [
    "goal", "dream", "trauma", "trigger", "belief",
    "relationship", "win", "blocker", "insight", "personal_info",
]

NOTE_EXTRACTION_PROMPT = """You are a quiet note-taker working alongside a therapy chatbot.

Read the user's latest message and decide whether it contains a durable fact worth
remembering long-term about the user: a goal, dream, trauma, trigger, belief,
relationship detail, win, blocker, insight, or basic personal info.

Do NOT extract a note for small talk, greetings, questions, vague statements,
or anything already fully captured by an existing note with no new detail added.

You are given the user's existing notes (key, title, category, content). If the
new message adds to, updates, or restates one of those notes with more detail,
return an "update" op referencing that note's exact key. If it's genuinely new
information, return a "create" op. If there's nothing worth saving, return an
empty list — this should be the common case, most messages need no note.

Respond with ONLY valid JSON, no markdown fences, no commentary, in exactly this
shape:
{"notes": [{"action": "create", "title": "<short title>", "content": "<1-2 sentence note, written in third person>", "category": "<one of: goal, dream, trauma, trigger, belief, relationship, win, blocker, insight, personal_info>"}]}

For an update, include the existing key:
{"notes": [{"action": "update", "key": "<existing note key>", "title": "<short title>", "content": "<merged, updated 1-2 sentence note>", "category": "<category>"}]}

If nothing to save: {"notes": []}
"""


def slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return s[:50] or "note"


def extract_notes(message: str, existing_notes: dict) -> list:
    """Ask the LLM whether this message contains a durable note-worthy fact.
    Returns a list of note ops (possibly empty) — it does NOT write to the
    session itself, see apply_note_ops for that."""
    if not message or len(message.strip()) < 3:
        return []

    notes_summary = [
        {"key": k, "title": n.get("title"), "category": n.get("category"), "content": n.get("content")}
        for k, n in existing_notes.items()
    ]

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": NOTE_EXTRACTION_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps({"message": message, "existing_notes": notes_summary}),
                },
            ],
            temperature=0,
            max_tokens=400,
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
        ops = parsed.get("notes", [])
        return ops if isinstance(ops, list) else []
    except Exception as e:
        # Note-taking is a nice-to-have; never let it break the chat turn.
        print(f"NOTE EXTRACTION ERROR: {type(e).__name__}: {e}")
        return []


def apply_note_ops(session: dict, ops: list) -> list:
    """Applies create/update ops to session['notes'] in place.
    Returns the list of notes touched this turn (each tagged with its key)."""
    notes = session.setdefault("notes", {})
    touched = []

    for op in ops:
        category = op.get("category")
        title = (op.get("title") or "").strip()
        content = (op.get("content") or "").strip()
        if category not in NOTE_CATEGORIES or not title or not content:
            continue

        key = op.get("key")
        if op.get("action") == "update" and key and key in notes:
            notes[key].update(
                title=title,
                content=content,
                category=category,
                updated_at=datetime.utcnow().isoformat(),
            )
            touched.append({**notes[key], "key": key})
        else:
            new_key = base_key = slugify(title)
            i = 2
            while new_key in notes:
                new_key = f"{base_key}_{i}"
                i += 1
            notes[new_key] = {
                "title": title,
                "content": content,
                "category": category,
                "updated_at": datetime.utcnow().isoformat(),
            }
            touched.append({**notes[new_key], "key": new_key})

    return touched


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

    # 2. RAG context (past turn similarity + long-term notes)
    rag_context = retrieve_context(req.message, turns)

    notes_context = ""
    if session.get("notes"):
        notes_context = "\n".join(
            f"- [{n['category']}] {n['title']}: {n['content']}"
            for n in session["notes"].values()
        )

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

    # Inject long-term notes as a hidden system note
    if notes_context:
        messages.append({
            "role": "system",
            "content": (
                "Known long-term notes about this user, built up over past sessions "
                "(use to inform your response naturally, do not recite them verbatim):\n\n"
                + notes_context
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

    # 6. Note-taking: only writes a note if the model judges it necessary
    note_ops = extract_notes(req.message, session.get("notes", {}))
    touched_notes = apply_note_ops(session, note_ops)

    # 7. Persist turn + notes
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
        extracted_notes={"notes": touched_notes} if touched_notes else None,
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


@app.get("/sessions/{session_id}/notes")
async def get_notes(session_id: int):
    session = load_session(session_id)
    return {"notes": session.get("notes", {})}


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)