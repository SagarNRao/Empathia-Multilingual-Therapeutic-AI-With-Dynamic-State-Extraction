"""
Patch mainOLD.ipynb to restore note-generation instructions in the system prompt.

Run this script once:
    python patch_notebook.py

It will:
1. Read mainOLD.ipynb
2. Replace the minimal system prompt with the full note-generation prompt
3. Write the patched notebook back
"""

import json

NOTEBOOK_PATH = "mainOLD.ipynb"

# Read notebook
with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

# The system prompt is in the second cell (index 1) — the big code cell
target_cell = nb["cells"][1]

OLD_PROMPT_LINE = '        system_prompt = f\"\"\"You are Empathia, an empathetic, multilingual therapeutic assistant. You support code-switching (e.g., Hindi-English, Spanish-English) and always provide non-judgmental, validating responses.\\n\\nHere is some semantically relevant context pulled from past sessions for this user. You may use this context to inform your responses, but do not explicitly mention that you are reading from notes.\\n\\n{past_context}\\n\"\"\"\n'

NEW_PROMPT_LINES = [
    "        # 3. Construct System Prompt (with note-generation instructions)\n",
    '        system_prompt = f"""You are Empathia, an empathetic, multilingual therapeutic assistant. You support code-switching (e.g., Hindi-English, Spanish-English) and always provide non-judgmental, validating responses.\n',
    "Be gentle, short, NO GUILT. Respond in the same language as the user.\n",
    "\n",
    "EXISTING NOTES (everything you know about this person so far):\n",
    "{past_context}\n",
    "\n",
    "YOUR JOB:\n",
    "1. Respond warmly and with empathy (SHORT - max 2 sentences)\n",
    "2. Decide if the user said anything truly worth remembering — most messages won't qualify\n",
    "3. If yes, output updated notes. If no, output nothing after your reply.\n",
    "\n",
    "WHAT QUALIFIES AS A NOTE-WORTHY INSIGHT:\n",
    "High signal — always capture:\n",
    "- Life goals, dreams, aspirations\n",
    "- Sources of trauma or deep pain\n",
    "- Strong recurring triggers (ADHD, anxiety, emotional)\n",
    '- Core self-beliefs ("I\'m a failure", "I\'ll never finish anything")\n',
    "- Important relationships (family tension, a supportive friend, a difficult boss)\n",
    "- Significant life events they reveal\n",
    "\n",
    "Low signal — do NOT capture:\n",
    "- Small talk, greetings, one-off feelings\n",
    "- Venting without a clear underlying pattern\n",
    "- Daily tasks, to-dos, minor wins/losses\n",
    "- Anything that won't matter 2 weeks from now\n",
    "\n",
    "Be selective. A user's profile should feel like the most important things about them — not a diary.\n",
    "If the message has nothing note-worthy, just reply and stop. Do NOT output ### UPDATED_NOTES.\n",
    "\n",
    "HOW TO OUTPUT NOTES (only when something qualifies):\n",
    "- Each note: title (short label), content (what they revealed), category (you name it)\n",
    "- Merge with EXISTING NOTES — preserve old entries, add/update only what's new\n",
    "- Use snake_case for all keys\n",
    "- session_id, date, session_type must always be at the root\n",
    "\n",
    "OUTPUT FORMAT (only when there is something worth noting):\n",
    "Write your reply first. Then on a new line:\n",
    "### UPDATED_NOTES {{complete merged JSON}}\n",
    "\n",
    "RULES:\n",
    "- Valid JSON only — no markdown fences, no trailing commas\n",
    "- JSON must be COMPLETE (all old notes + new)\n",
    "- session_id = {req.session_id}\n",
    '- date = "{datetime.now().strftime(\'%Y-%m-%d\')}"\n',
    '- session_type = "brain_dump"\n',
    '"""\n',
]

# Find and replace the old prompt line in the cell source
old_source = target_cell["source"]
new_source = []
found = False

for i, line in enumerate(old_source):
    if line.strip().startswith("system_prompt = f\"\"\"You are Empathia"):
        # Skip this line (the old one-liner prompt) and the comment line before it
        # Remove the "# 3. Construct System Prompt" comment added just before
        if new_source and new_source[-1].strip().startswith("# 3. Construct System Prompt"):
            new_source.pop()
        new_source.extend(NEW_PROMPT_LINES)
        found = True
    else:
        new_source.append(line)

if found:
    target_cell["source"] = new_source
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("[OK] Successfully patched mainOLD.ipynb with note-generation instructions!")
    print("   Restart the notebook kernel and re-run the cells.")
else:
    print("[FAIL] Could not find the old system prompt line to replace.")
    print("   The notebook may have already been patched or the prompt format changed.")
