# Methodology: Fine-Tuned Multilingual Therapy LLM with Cognitive Shift Analysis

## 1. System Architecture Overview

The proposed system consists of three main components:
1. **Fine-Tuned Language Model Server** (Qwen2.5-7B-Instruct with LoRA adapter)
2. **Cognitive Shift Scoring Module** (Fine-tuned sequence classifier)
3. **Session Management and RAG System** (Dynamic context retrieval)

### 1.1 System Data Flow

```
User Input (Multilingual)
    ↓
[Parallel Processing]
    ├─ Component A: Fine-Tuned LLM → Empathetic Response + Notes Extraction
    └─ Component B: NLP Scoring Server → Cognitive Shift Analysis (affective/cognitive/agency)
    ↓
[Response Assembly]
    ├─ Clean Response Text
    ├─ Cognitive Shift Scores
    └─ Extracted Session Notes
    ↓
[Session Persistence & RAG]
    ├─ Store in JSON session files
    ├─ Extract relevant past context
    └─ Feed back to LLM for next turn
    ↓
Frontend Visualization
```

---

## 2. Natural Language Processing Component

### 2.1 Multilingual Embedding and Language Detection

**Model**: XLM-RoBERTa-Base (Cross-lingual Language Model)

**Purpose**: Enable the system to understand and process code-switched text (mixed-language input such as English-Hindi "Hinglish" or English-Spanish).

**Justification**: 
- XLM-RoBERTa-Base is trained on 100+ languages with shared embedding space
- Handles code-switching naturally without separate language-specific models
- Enables therapy conversations in multilingual communities where language mixing is common

**Implementation Details**:
- Tokenizer: `sentencepiece` (BPE-based, handles subword tokens across languages)
- Vocabulary size: ~250K tokens across all supported languages
- Hidden dimension: 768 (balanced between expressiveness and efficiency)

**Example Supported Patterns**:
```
Input: "I feel bahut anxious kya karu"
Interpretation: English "I feel" + Hindi "bahut anxious" (very anxious) + Hindi "kya karu" (what to do)
Embedding: Unified cross-lingual representation captures semantic meaning across language boundaries
```

---

## 3. Fine-Tuned Generative Model

### 3.1 Base Model Selection

**Base Model**: Qwen2.5-7B-Instruct

**Justification for 7B over larger models**:
- 7 billion parameters provides sufficient capacity for therapy-focused tasks
- Requires only 16GB VRAM (vs 40GB+ for 70B models)
- Maintains reasonable inference latency (2-5s on GPU, 15-30s on CPU)
- Better fine-tuning efficiency with limited computational resources

**Model Architecture**:
- Transformer-based causal language model
- Grouped Query Attention (GQA) for efficiency
- RoPE (Rotary Position Embeddings) for position encoding
- Flash Attention for accelerated computation

### 3.2 Fine-Tuning Strategy: LoRA (Low-Rank Adaptation)

**Method**: Parameter-Efficient Fine-Tuning (PEFT) using LoRA

**Configuration**:
- Rank (r): 16
- Alpha (α): 32 (scaling factor)
- Dropout: 0.05
- Target modules: 
  - Attention projections: `q_proj`, `k_proj`, `v_proj`, `o_proj`
  - MLP/FFN layers: `gate_proj`, `up_proj`, `down_proj`
- Bias: None

**Why LoRA over Full Fine-Tuning**:
1. Trainable parameters: ~0.5% of total (10M vs 7B parameters)
2. Memory efficient: 8-bit quantization + gradient checkpointing → 20GB A100
3. Fast training: 3 epochs on 8K samples in ~2-3 hours
4. Easy deployment: ~50MB adapter vs 14GB full model
5. Preserves base model knowledge while specializing for therapy

**Training Configuration**:

| Parameter | Value | Justification |
|-----------|-------|---|
| Quantization | 8-bit | Reduces VRAM from 28GB → 10GB |
| Gradient Checkpointing | Enabled | Trades compute for memory |
| Batch Size (effective) | 32 | 4 per-device × 8 accumulation |
| Learning Rate | 2e-4 | Conservative for fine-tuning |
| Scheduler | Cosine with warmup | Smooth convergence |
| Epochs | 3 | Sufficient for dataset size (~8K samples) |
| Optimizer | paged_adamw_8bit | Memory-efficient Adam variant |

### 3.3 Training Data: EmpatheticDialogues Dataset

**Dataset**: EmpatheticDialogues (Rashkin et al., 2018)

**Characteristics**:
- Size: ~25K conversations
- Training subset: 8K samples (stratified by emotion)
- 1:1 User-Agent turn pairs
- Emotion labels: 32 emotion categories
- Context: Situation + emotion + dialogue

**Data Preparation**:
1. Extract user input and agent response
2. Clean prefixes ("Customer :", "Agent :")
3. Build conversation prompt using Qwen2.5 chat template
4. Mask prompt tokens during loss computation (only train on response)
5. Set labels to -100 for padding tokens

**Format Example**:
```
Input:  <|im_start|>system\nYou are a compassionate therapist...<|im_end|>
        <|im_start|>user\nI feel overwhelmed<|im_end|>
        <|im_start|>assistant\n
Target: I hear you. That's a heavy feeling...
Labels: [-100, -100, ..., -100, token1, token2, ...]
        (Loss only on response tokens)
```

---

## 4. Cognitive Shift Analysis Module

### 4.1 Three-Dimensional Emotional/Cognitive State Tracking

**Concept**: Track user's psychological state across three dimensions:
1. **Affective**: Emotional expression (feelings, moods, distress)
2. **Cognitive**: Thinking patterns (reasoning, reflection, understanding)
3. **Agency**: Sense of control (action, decision-making, autonomy)

**Model Architecture**: Fine-tuned Sequence Classifier

- Base: XLM-RoBERTa-Base
- Task: 3-class classification
- Output: Probability distribution over [affective, cognitive, agency]

### 4.2 Scoring Implementation

**Input**: User message text

**Processing**:
```python
tokenized = tokenizer(
    text, 
    max_length=512, 
    truncation=True, 
    padding=True
)
logits = model(input_ids, attention_mask)
scores = softmax(logits)  # Range: [0, 1] for each class
dominant = argmax(scores)  # Most likely category
```

**Output Format**:
```json
{
  "scores": {
    "affective": 0.7234,
    "cognitive": 0.1893,
    "agency": 0.0873
  },
  "dominant": "affective"
}
```

**Interpretation**:
- Affective-dominant (0.7+): User expressing emotions
- Cognitive-dominant (0.5+): User reasoning/reflecting
- Agency-dominant (0.5+): User taking/planning action

---

## 5. Response Generation with Dual-Tone Support

### 5.1 Tone Conditioning

The model supports two distinct personas:

#### 5.1.1 Therapist Tone
- Professional but warm
- Reflective questions
- Minimal advice unless requested
- Validates emotions
- Focuses on understanding

**System Prompt**:
```
You are a compassionate, licensed therapist.
Listen carefully, reflect feelings back,
ask one open-ended follow-up question.
Never give direct advice unless explicitly asked.
```

#### 5.1.2 Friend Tone
- Conversational and natural
- Supportive and genuine
- Casual language
- Empathetic but relatable
- May offer perspective gently

**System Prompt**:
```
You are a warm, supportive friend who genuinely cares.
Respond naturally and conversationally.
Acknowledge feelings first, then offer perspective if helpful.
Match the user's language and tone exactly.
```

### 5.2 Tone Selection Mechanism

**Method**: Concatenate tone identifier in system prompt before inference

```python
SYSTEM_PROMPT_TEMPLATE = """
You are Indy, an ADHD copilot. {TONE_DESCRIPTION}

EXISTING CONTEXT:
{PAST_NOTES}

YOUR RESPONSE SHOULD BE:
- Short (max 2 sentences)
- Kind and empathetic
- In the user's language

{TONE_SPECIFIC_INSTRUCTIONS}
"""

# Example for therapist tone
prompt = SYSTEM_PROMPT_TEMPLATE.format(
    TONE_DESCRIPTION=THERAPIST_DESCRIPTION,
    PAST_NOTES=context,
    TONE_SPECIFIC_INSTRUCTIONS=THERAPIST_RULES
)
```

---

## 6. Retrieval-Augmented Generation (RAG) for Context

### 6.1 Problem Statement

**Base Model Limitation**: 
- Qwen2.5-7B context window: 32K tokens (sufficient for single conversation)
- But each session accumulates 50-200+ messages over weeks
- Full history would exceed context window

**Solution**: Dynamic context retrieval

### 6.2 RAG Implementation

**Storage**: Session-based JSON persistence

```json
{
  "session_id": 1,
  "created": "2026-04-07T10:30:00",
  "conversation": [
    {
      "role": "user",
      "content": "I feel overwhelmed",
      "timestamp": "...",
      "cognitive_shift": {"scores": {...}, "dominant": "affective"}
    },
    {
      "role": "assistant",
      "content": "I hear you...",
      "notes": {
        "session_id": 1,
        "priorities": ["finish project"],
        "blockers": ["anxiety about failure"],
        "insights": {"core_fear": "I'm not good enough"}
      }
    }
  ],
  "cognitive_shifts": [...]
}
```

**Retrieval Strategy**:
1. Load previous 3-4 sessions (recency bias)
2. Extract high-signal notes (life goals, triggers, relationships)
3. Summarize: "Session N: Mood: [mood], Wins: [small_wins]"
4. Concatenate into system prompt

**Pseudo-code**:
```python
def get_past_context(session_id: int) -> str:
    context_blocks = []
    for i in range(session_id - 1, max(0, session_id - 4), -1):
        session_data = load_session(i)
        summary = f"Session {i}: Mood: {session_data.get('mood')}, Wins: {session_data.get('small_wins')}"
        context_blocks.append(summary)
    return "\n".join(context_blocks)
```

**Why This Works**:
- Recent sessions more relevant (temporal locality)
- Automatic summarization avoids token bloat
- Notes are pre-filtered by LLM judgment (high signal)
- Graceful degradation if notes are sparse

---

## 7. Intelligent Note Extraction

### 7.1 Selective Insight Capture

**Problem**: Not all user messages warrant note-taking. Random logging creates noise.

**Solution**: LLM-guided selective extraction

**High-Signal Notes** (Always capture):
- Life goals, dreams, aspirations
- Sources of trauma or deep pain
- Strong recurring triggers (ADHD, anxiety, emotional)
- Core self-beliefs ("I'm a failure")
- Important relationships (family tension, supportive friend)
- Significant life events

**Low-Signal Notes** (Skip):
- Small talk, greetings
- One-off venting without pattern
- Daily tasks, to-dos, minor wins/losses
- Anything that won't matter in 2 weeks

### 7.2 Implementation

**Mechanism**: Include extraction instructions in system prompt

```
OUTPUT FORMAT (only when something qualifies):
Write your reply first. Then on a new line:
### UPDATED_NOTES {complete merged JSON}

RULES:
- Valid JSON only
- JSON must be COMPLETE (all old notes + new)
- session_id, date, session_type must always be at the root
```

**Post-Processing**:
1. Check for `### UPDATED_NOTES` marker
2. Extract JSON payload
3. Parse and validate
4. Merge with existing session notes
5. Persist to disk

**Example Extraction**:
```
User: "My therapist thinks I should try meditation but I'm skeptical"

LLM Response:
"That's understandable. What appeals to you about it, or what concerns do you have?"

### UPDATED_NOTES
{
  "session_id": 1,
  "date": "2026-04-07",
  "session_type": "brain_dump",
  "relationships": {
    "therapist": "Recommends meditation, patient skeptical"
  },
  "mental_health_approaches": ["meditation (skeptical)"]
}
```

---

## 8. Model Training and Inference

### 8.1 Training Phase (Google Colab)

**Environment**: 
- GPU: NVIDIA A100 (20GB VRAM)
- Framework: Hugging Face Transformers + PEFT
- Training time: ~2-3 hours for 3 epochs

**Process**:
1. Load base model in 8-bit (10GB VRAM)
2. Attach LoRA adapters (minimal overhead)
3. Enable gradient checkpointing (memory ↓, compute ↑)
4. Train on 8K samples
5. Evaluate on validation set
6. Save adapter (~50MB) to disk

**Output**: `phase2_artifacts/lora_adapter/` directory

### 8.2 Inference Phase (Local or Server)

**Deployment Options**:

#### Option A: GPU Server (Recommended)
```python
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.float16,
    device_map="cuda"  # Full model in VRAM
)
model = PeftModel.from_pretrained(model, "./lora_adapter")
```
- Speed: 2-5 seconds per response
- Cost: Persistent GPU (e.g., AWS GPU instance)

#### Option B: CPU Fallback (Always Available)
```python
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.float32
).to("cpu")  # Uses system RAM
model = PeftModel.from_pretrained(model, "./lora_adapter")
```
- Speed: 15-30 seconds per response
- Cost: Server CPU (minimal)
- Graceful degradation if VRAM insufficient

### 8.3 Generation Parameters

**Sampling Strategy**: Controlled stochasticity

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `temperature` | 0.7 | Balance: not too random (0), not too deterministic (1) |
| `top_p` | 0.9 | Nucleus sampling: keep top 90% probability mass |
| `repetition_penalty` | 1.1 | Penalize repeated phrases |
| `max_new_tokens` | 1024 | Reasonable upper bound (can be truncated) |
| `do_sample` | True | Use sampling not greedy decoding |

---

## 9. Comparative Analysis: Groq API vs. Fine-Tuned LLM

### 9.1 Initial Implementation (Groq API)

**Model**: Groq's compound-mini (proprietary)

**Advantages**:
- No training required
- Fast inference (1-2 seconds)
- Minimal setup
- Offloads computational burden

**Disadvantages**:
- Generic responses (not therapy-focused)
- Limited multilingual/code-switching
- Per-request API costs
- Privacy concerns (data to external service)
- No customization capability

### 9.2 Revised Implementation (Fine-Tuned LLM)

**Model**: Qwen2.5-7B-Instruct with LoRA

**Advantages**:
- Therapy-specialized training
- Excellent multilingual/code-switching
- No API costs (local inference)
- Complete privacy (no external calls)
- Customizable parameters and tones
- Better long-term cost (amortized over sessions)

**Disadvantages**:
- Inference latency: 2-5s (GPU) vs 1-2s (Groq)
- Requires hardware (GPU for speed, CPU works)
- Training time and expertise
- Model complexity to maintain

**Cost Analysis** (over 1 year, 100 daily users, 2 messages/user/day):
- Groq: ~$7,300/year ($0.005 per request)
- Fine-Tuned: ~$2,000/year (GPU instance) + training
- Break-even: ~6-8 months

---

## 10. System Integration and API Design

### 10.1 Chat Endpoint

**Endpoint**: `POST /chat`

**Request**:
```json
{
  "session_id": 1,
  "message": "I feel overwhelmed with work",
  "history": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

**Processing Steps**:
1. Cognitive shift analysis (parallel, ~1s)
2. RAG retrieval (synchronous, ~10ms)
3. System prompt construction (~50ms)
4. LLM generation (~3-5s on GPU)
5. Note extraction (~100ms)
6. Session persistence (~50ms)

**Response**:
```json
{
  "reply": "That sounds really stressful...",
  "cognitive_shift": {
    "scores": {"affective": 0.72, "cognitive": 0.18, "agency": 0.10},
    "dominant": "affective"
  },
  "extracted_notes": {
    "session_id": 1,
    "date": "2026-04-07",
    "blockers": ["work overload"]
  }
}
```

**Total Response Time**: ~3.5-5.5 seconds (mostly LLM generation)

### 10.2 Cognitive Shifts Endpoint

**Endpoint**: `GET /sessions/{session_id}/cognitive-shifts`

**Purpose**: Retrieve historical cognitive shift data for visualization

**Response**:
```json
{
  "session_id": 1,
  "total_messages": 5,
  "shifts": [
    {
      "message_num": 1,
      "affective": 0.72,
      "cognitive": 0.18,
      "agency": 0.10,
      "dominant": "affective",
      "timestamp": "2026-04-07T10:30:00",
      "content": "I feel overwhelmed..."
    },
    ...
  ]
}
```

---

## 11. Frontend Visualization

### 11.1 Cognitive Shift Graph

**Chart Type**: Multi-line chart (3 series)

**Series**:
- Red line: Affective scores (emotional intensity)
- Teal line: Cognitive scores (thinking/reasoning)
- Yellow line: Agency scores (control/action orientation)

**X-axis**: Message sequence (1, 2, 3, ...)

**Y-axis**: Probability [0, 1]

**Visualization Logic**:
```javascript
fetch(`${SERVER}/sessions/${sessionId}/cognitive-shifts`)
  .then(data => {
    // Extract affective, cognitive, agency arrays
    const affectiveData = data.shifts.map(s => s.affective);
    const cognitiveData = data.shifts.map(s => s.cognitive);
    const agencyData = data.shifts.map(s => s.agency);
    
    // Plot 3-line chart with legend
    renderLineChart(affectiveData, cognitiveData, agencyData);
  });
```

**Interpretation for Therapist**:
- Rising affective: User becoming more emotionally engaged
- Rising cognitive: User reflecting/thinking critically
- Rising agency: User gaining sense of control

---

## 12. Evaluation Metrics and Validation

### 12.1 Quantitative Metrics

**For Fine-Tuning**:
- **Perplexity**: Measure on validation set
- **BLEU/ROUGE**: Compare generated responses to reference
- **Inference Latency**: Measure end-to-end response time

**For Cognitive Shift Scoring**:
- **Classification Accuracy**: Against manually labeled samples
- **F1 Score**: Per-class performance
- **Confusion Matrix**: Error analysis

### 12.2 Qualitative Evaluation

**Manual Review**:
1. Sample 50 responses across tones
2. Rate on:
   - Empathy (1-5)
   - Relevance to user input (1-5)
   - Appropriateness for therapy (1-5)
   - Language quality (1-5)
3. Calculate inter-rater agreement (Cohen's Kappa)

**Cognitive Shift Validation**:
1. Have domain expert label 100 messages
2. Compare model predictions to expert
3. Analyze disagreement patterns

### 12.3 A/B Testing (Future Work)

**Comparison Groups**:
- Therapist tone vs. Friend tone
- With notes vs. Without notes
- With RAG context vs. Without context

**Metrics**:
- User engagement (session length)
- Session completion rate
- User satisfaction (feedback)

---

## 13. Ethical Considerations and Limitations

### 13.1 Limitations

1. **Not a Replacement for Therapy**: Model is designed as a support tool, not clinical treatment
2. **Smaller Context Window**: 32K tokens limits long-term memory vs. human therapists
3. **No Real-Time Danger Detection**: Cannot assess acute mental health crises
4. **Cultural Specificity**: Trained primarily on English-centric EmpatheticDialogues
5. **Hallucination Risk**: May generate plausible-sounding but false information

### 13.2 Safeguards

1. **Explicit Disclaimer**: "I'm an AI support tool, not a therapist"
2. **Crisis Detection**: Flag keywords ("suicide", "harm") and direct to crisis hotline
3. **Regular Audits**: Monitor for harmful recommendations
4. **User Consent**: Clear data usage policies
5. **Local Processing**: No external data sharing without consent

---

## 14. Reproducibility and Implementation

### 14.1 Training Code

See: `phase2_finetuning.ipynb`
- Environment: Google Colab with A100 GPU
- Dataset: EmpatheticDialogues (Kaggle)
- Output: `phase2_artifacts/lora_adapter/`

### 14.2 Inference Code

See: `inference.ipynb` (Colab) and `main.ipynb` (local)
- Model loading: Transformers + PEFT
- API: FastAPI with Uvicorn
- Deployment: Docker + cloud instance

### 14.3 Hardware Requirements

| Component | GPU (Recommended) | CPU (Fallback) |
|-----------|---|---|
| RAM | 16GB VRAM | 28GB system RAM |
| CPU | Any modern | 8+ cores |
| Storage | 30GB (cache) | 30GB (cache) |
| Network | Minimal (local) | Minimal (local) |

---

## 15. Future Work

1. **Larger Context Windows**: Implement sparse attention or retrieval-based methods
2. **Fine-Grained Emotion Labels**: Extend from 3-class to 32-class emotion detection
3. **User Feedback Loop**: Incorporate user corrections to improve model
4. **Multi-Modal Input**: Add voice/video analysis
5. **Cross-Lingual Transfer**: Evaluate on non-English therapy conversations
6. **Explainability**: Generate explanations for note extraction decisions
7. **Personalization**: User-specific model adaptations via few-shot learning

---

## References

1. Qwen Team. Qwen2.5: A Large Language Model at Scale. Alibaba Cloud, 2024.
2. Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685.
3. Conneau, A., et al. (2019). Unsupervised Cross-lingual Representation Learning at Scale. ACL.
4. Rashkin, H., et al. (2018). Event Assisted Open Domain Question Answering. EMNLP.
5. Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS.
6. Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS.

---

**Author**: [Your Name]  
**Date**: April 7, 2026  
**Project**: Indy ADHD Copilot - Multilingual Therapy Support System

---

## Appendix A: JSON Schema Examples

### Session File Structure
```json
{
  "session_id": 1,
  "created": "2026-04-07T10:30:00.000Z",
  "conversation": [
    {
      "role": "user",
      "content": "I feel overwhelmed",
      "timestamp": "2026-04-07T10:30:05.000Z"
    },
    {
      "role": "assistant",
      "content": "That sounds really heavy.",
      "timestamp": "2026-04-07T10:30:10.000Z",
      "cognitive_shift": {
        "scores": {
          "affective": 0.8234,
          "cognitive": 0.1456,
          "agency": 0.0310
        },
        "dominant": "affective"
      },
      "notes": {
        "session_id": 1,
        "date": "2026-04-07",
        "session_type": "brain_dump",
        "blockers": ["work pressure"],
        "insights": []
      }
    }
  ],
  "cognitive_shifts": [
    {
      "message": "I feel overwhelmed",
      "analysis": {
        "scores": {...},
        "dominant": "affective"
      },
      "timestamp": "2026-04-07T10:30:10.000Z"
    }
  ]
}
```

### API Request/Response Format
```json
{
  "request": {
    "session_id": 1,
    "message": "I've been having trouble focusing",
    "history": [
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "..."}
    ]
  },
  "response": {
    "reply": "That's a common challenge. What's been distracting you lately?",
    "cognitive_shift": {
      "scores": {
        "affective": 0.45,
        "cognitive": 0.40,
        "agency": 0.15
      },
      "dominant": "affective"
    },
    "extracted_notes": {
      "session_id": 1,
      "date": "2026-04-07",
      "session_type": "brain_dump",
      "blockers": ["difficulty focusing"],
      "insights": []
    }
  }
}
```

---

## Appendix B: LoRA Configuration Details

**Qwen2.5-7B-Instruct + LoRA Specifications**

```python
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                          # Adapter rank
    lora_alpha=32,                 # Scaling factor
    lora_dropout=0.05,             # Regularization
    bias="none",                   # No bias terms
    target_modules=[               # Where to inject adapters
        "q_proj",                  # Query projection
        "k_proj",                  # Key projection
        "v_proj",                  # Value projection
        "o_proj",                  # Output projection
        "gate_proj",               # Gate in MLP
        "up_proj",                 # Up projection in MLP
        "down_proj",               # Down projection in MLP
    ],
    init_lora_weights="gaussian"   # Initialization method
)

# Expected results:
# - Base parameters: 7.06B
# - Trainable parameters: ~10.5M (0.15%)
# - Adapter size on disk: ~45-50MB
# - VRAM reduction: 28GB → ~12GB (with 8-bit quantization)
```

---

