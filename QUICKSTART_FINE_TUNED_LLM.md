# Quick Start: Running Indy with Fine-Tuned LLM

## 🚀 Startup Sequence (in order)

### Step 1: NLP Scoring Server (Port 8001)
```powershell
cd c:\Projects\AI\therapii\therapy-inator\server\fineTunedModel
jupyter notebook test.ipynb
# In browser: Run Cell 1
```
Expected output: `INFO:     Uvicorn running on http://0.0.0.0:8001`

### Step 2: Main Chat Server (Port 8000) - NOW WITH FINE-TUNED LLM
```powershell
cd c:\Projects\AI\therapii\therapy-inator\server
jupyter notebook main.ipynb
# In browser: Run Cell 1 (pip install), then Cell 2, then Cell 5
```
Expected output: `INFO:     Uvicorn running on http://0.0.0.0:8000`

After "Fine-tuned LLM loaded successfully!", wait for NLP server to be ready.

### Step 3: Frontend (Port 3000)
```powershell
cd c:\Projects\AI\therapii\therapy-inator
npm run dev
```
Open browser: `http://localhost:3000`

---

## ✨ What Changed

| Before (Groq API) | After (Fine-Tuned LLM) |
|---|---|
| Required GROQ_API_KEY environment variable | No API key needed |
| Called Groq servers | Local inference (GPU-accelerated) |
| Generic responses | Therapy-focused, empathetic responses |
| Limited multilingual support | English-Hindi, English-Spanish code-switching |
| Paid per request | Free (hardware only) |

---

## 🔧 Model Details

**Base Model:** Qwen2.5-7B-Instruct (7B parameters)
**Fine-Tuned On:** EmpatheticDialogues dataset
**Adapter:** LoRA (Low-Rank Adaptation) - ~50 MB
**Location:** `server/fineTunedLLM/empathia-multilingual-therapy-bot/phase2_artifacts/lora_adapter/`

---

## 📊 Performance

| Hardware | Response Time |
|----------|---|
| NVIDIA GPU (A100) | 2-5 seconds |
| NVIDIA GPU (RTX 3090) | 3-8 seconds |
| CPU (i7/Ryzen 7) | 15-30 seconds |

---

## ✅ Checklist Before Running

- [ ] Server directory has `fineTunedLLM` folder
- [ ] `phase2_artifacts/lora_adapter` contains `adapter_model.safetensors`
- [ ] Port 8000, 8001, 3000 are free
- [ ] Enough RAM (28GB recommended for CPU, 16GB for GPU)
- [ ] Dependencies installed (`transformers`, `peft`, `torch`, `fastapi`)

---

## 🐛 Common Issues

**"Module not found: peft"**
→ Cell 1 in main.ipynb will install it

**"CUDA out of memory"**
→ Use CPU or reduce max_new_tokens to 512

**"FileNotFoundError: lora_adapter"**
→ Check path relative to server/ directory

**"Slow response (15+ seconds)"**
→ You're running on CPU (normal) or GPU not detected

---

## 📝 Testing

Send a test message in the chat:
- ✅ Should get an empathetic response (2 sentences max)
- ✅ Cognitive shift scores should appear in graph
- ✅ Server status indicator should be green
- ✅ Check console for generation time

---

## 🔐 Privacy

✅ All inference runs locally - no data sent to external APIs
✅ Chat history stored in `sessions/session_XXX.json` locally
✅ No API keys required
✅ Model weights never uploaded

---

## 📚 Full Documentation

See `FINE_TUNED_LLM_SETUP.md` for detailed troubleshooting, architecture, and performance benchmarks.

