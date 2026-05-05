# Hindi Instruction-Following Dataset and LLM Fine-Tuning

**Course:** Deep Learning — Individual Task 2  
**Language:** Hindi (हिन्दी) — a low-resource language in LLM research  
**Model:** Sarvam-1 fine-tuned with LoRA (Low-Rank Adaptation)  
**Dataset:** 1221 original Hindi instruction-response pairs across 6 domains  

---

## What This Project Does

Most large language models are trained primarily on English text. Even models that claim multilingual support tend to perform poorly on Hindi — giving short, incomplete, or sometimes nonsensical answers. This project builds an original Hindi instruction-following dataset from scratch and uses it to fine-tune Sarvam-1, a 2 billion parameter model built specifically for Indian languages.

The goal is simple: make the model better at answering Hindi questions in a natural, detailed and helpful way — and prove with numbers that it actually worked.

---

## Why Hindi?

Hindi is spoken by over 600 million people making it one of the most widely spoken languages in the world. Despite this, it is significantly underrepresented in instruction-tuned language models compared to English. Most open-source models either respond in English when asked in Hindi, give very short one-line answers, or produce grammatically awkward text.

This is what makes Hindi a strong and well-motivated choice for this project. There is a real and measurable gap between the language's global reach and its representation in AI systems. Fine-tuning a model on a carefully constructed Hindi dataset directly addresses this gap.

---

## Dataset

### Data Collection and Annotation Process

Creating a good dataset is the most important part of this project. Here is exactly how it was done:

**Step 1 — Domain planning**

Before writing a single example, six domains were chosen that represent everyday Hindi language use. The goal was to cover a wide variety of instruction types — factual questions, practical how-to instructions, mathematical reasoning, cultural knowledge, social conversation and science — so the model would learn to handle diverse queries rather than just one type of question.

**Step 2 — Example generation**

Examples were generated using two AI assistants — Claude and ChatGPT — with carefully designed prompts that enforced strict rules:
- Every response must be a minimum of 60 words in natural Hindi
- No English words mixed inside Hindi sentences
- The instruction and response must be a logical and complete pair
- Output must be in exact JSONL format with no extra text

Generating in batches of 60-100 examples per domain made the process manageable and consistent. A total of 15 batches were generated across the six domains.

**Step 3 — Manual review and annotation**

Every single example was read by a human reviewer. This is the annotation step. During review the following things were checked:
- Is the Hindi grammatically natural and not translation-sounding?
- Is the factual information in the response correct?
- Is the response actually answering the question asked?
- Are there any obvious hallucinations especially in history and science domains?

Examples that failed any of these checks were either corrected or removed entirely.

**Step 4 — Automated quality validation**

A Python script ran a final quality check on every line of the dataset file:

```python
# checks performed on every example
1. Valid JSON        — parseable with no syntax errors
2. Correct schema   — exactly 3 messages in system/user/assistant order
3. No empty fields  — all three roles have actual content
4. Minimum length   — assistant response at least 15 words
5. Language check   — Hindi character ratio to detect English contamination
```

This combination of human review and automated validation produced a clean, reliable dataset.

### Dataset Structure

Each example is stored as a single line in JSONL format (JSON Lines). Every line is one complete training example:

```json
{"messages": [
  {"role": "system", "content": "तुम एक सहायक हिन्दी भाषा के सहायक हो।"},
  {"role": "user", "content": "महात्मा गांधी के बारे में बताइए।"},
  {"role": "assistant", "content": "महात्मा गांधी भारत की स्वतंत्रता आंदोलन के प्रमुख नेता थे..."}
]}
```

The system prompt is fixed across all examples — it tells the model it is a helpful Hindi language assistant. The user field contains the instruction in Hindi. The assistant field contains the expected detailed Hindi response.

### Dataset Statistics

| Domain | Examples | Avg Response Length | Sample Topics |
|---|---|---|---|
| Indian culture and history | 204 | 87 words | Freedom fighters, Mughal empire, festivals, monuments, independence |
| General knowledge Q&A | 203 | 91 words | Science concepts, world geography, biology, physics, space |
| Everyday instructions | 203 | 94 words | Cooking recipes, household tasks, banking, health tips, travel |
| Simple reasoning and math | 204 | 78 words | Word problems, profit-loss, percentages, logic puzzles, arithmetic |
| Polite conversation and etiquette | 203 | 83 words | Greetings, conflict resolution, social situations, apologies |
| Indian geography and environment | 204 | 89 words | Rivers, mountains, states, climate zones, national parks, wildlife |
| **Total** | **1221** | **87 words avg** | **6 domains, all original** |

### Train / Validation / Test Split

| Split | Examples | Percentage | Purpose |
|---|---|---|---|
| Train | 976 | 80% | Used to update model weights during fine-tuning |
| Validation | 122 | 10% | Monitored during training to detect overfitting |
| Test | 123 | 10% | Held out completely — used only for final evaluation |

The split was done with a fixed random seed (42) for full reproducibility. The test set was never seen by the model at any point during training.

### Quality Validation Report

| Metric | Result |
|---|---|
| Total lines processed | 1234 |
| Valid JSON | 1221 (98.9%) |
| Correct schema | 1221 (100% of valid) |
| Short responses removed | 11 |
| English contamination detected | 2 (removed) |
| Final clean examples | 1221 |

### Why This Dataset Is Useful

Hindi speakers interact with AI assistants in their native language every day — asking about Indian history, following cooking instructions, solving everyday math problems, navigating social situations. Existing instruction-tuned models are not optimised for these use cases in Hindi.

This dataset is useful because it covers exactly these everyday scenarios with detailed, natural Hindi responses written at a level that a typical Hindi speaker would find helpful and understandable. It is entirely original — no examples were copied from existing benchmarks or datasets.

---

## Model

### Why Sarvam-1?

We tested several models before settling on Sarvam-1:

| Model | Size | Hindi quality | Outcome |
|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 0.5B | Poor | Confused रेगिस्तान (desert) with रेजिमेंट (regiment) |
| Qwen2.5-1.5B-Instruct | 1.5B | Poor | Same issues even in full precision without quantization |
| Sarvam-1 | 2B | Good | Coherent, grammatically correct Hindi out of the box |

Sarvam-1 is a 2 billion parameter causal language model — an LLM — built by Sarvam AI, an Indian AI research company. It was trained from scratch on a large multilingual Indian language corpus including Hindi, Tamil, Telugu, Kannada, Malayalam, Bengali, Gujarati, Marathi, Punjabi and Odia. Unlike general multilingual models that treat Hindi as a secondary language, Sarvam-1 was built with Indian languages as the primary objective — making it the right foundation for this fine-tuning task.

---

## Fine-Tuning Method — LoRA

### What LoRA Is

Training all 2 billion parameters of Sarvam-1 would require enormous GPU memory and many hours of compute. LoRA (Low-Rank Adaptation) is a smarter approach. Instead of changing the original model weights at all, it freezes them completely and injects small trainable adapter matrices alongside specific layers.

The mathematical idea: rather than updating a large weight matrix W directly, LoRA learns two small matrices A and B where their product A×B approximates the update needed. Because A and B are much smaller than W, only a tiny fraction of parameters need to be trained.

In our case 8.8 million parameters were trained out of 502 million total — just 1.75%. This made the entire fine-tuning process run in under 15 minutes on a T4 GPU.

### LoRA Hyperparameters — Plain Language Explanation

**Rank (r = 16)**
This is the most important LoRA setting. It controls the size of the adapter matrices — how many dimensions the low-rank approximation uses. Think of it as the adapter's learning capacity. We chose 16 because it gives the model enough room to learn from 1221 examples without memorising them. Too low (like 4 or 8) and the adapter cannot capture enough nuance. Too high (like 64) and it risks overfitting on a relatively small dataset.

**Alpha (lora_alpha = 32)**
This is a scaling factor that controls how strongly the adapter's learned changes influence the final model output. We set it to 32 which is exactly 2× the rank — this is the standard recommended ratio in LoRA research. Think of it as a volume knob: too low and fine-tuning has no effect, too high and it overwrites the model's existing Hindi language knowledge.

**Dropout (lora_dropout = 0.05)**
During each training step 5% of the adapter weights are randomly turned off. This is a regularisation technique that prevents the model from relying too heavily on any single connection. It is especially important with datasets of this size where the risk of overfitting is real.

**Target modules**
LoRA adapters were applied to all seven projection layers in the transformer — the four attention layers (q_proj, k_proj, v_proj, o_proj) which handle how the model understands relationships between words, and the three feed-forward layers (gate_proj, up_proj, down_proj) which handle how it transforms and generates text. Targeting all seven gives the adapter maximum coverage.

### Training Hyperparameters — Plain Language Explanation

**Learning rate (1e-4 with cosine scheduler)**
The learning rate controls how large a step the model takes when updating its weights after each batch. We used 1e-4 which is conservative and stable — a smaller step means less risk of the model forgetting its existing Hindi knowledge while it learns from our dataset. The cosine scheduler gradually reduces this rate as training progresses so the final updates are small and precise rather than overshooting.

**Effective batch size (2 per device × 8 gradient accumulation = 16)**
We could not fit 16 examples in GPU memory at once so we used gradient accumulation. The GPU processes 2 examples at a time, accumulates the gradients across 8 steps, then makes one weight update. The model behaves as if it saw 16 examples per update which gives stable and meaningful gradient estimates.

**Epochs (3)**
The model saw the entire training set of 976 examples 3 times, giving 183 total weight updates. Three epochs was the right balance — enough repetitions for the loss to decrease meaningfully (from 0.98 to 0.75) without overfitting.

**Gradient clipping (max_grad_norm = 0.3)**
Occasionally individual batches can produce very large gradient values that destabilise training. Setting a hard limit of 0.3 means any gradient that exceeds this is scaled down automatically. This kept training smooth especially in the early steps when the randomly initialised adapter weights were still settling.

**Warmup steps (20)**
For the first 20 steps the learning rate starts very small and gradually increases to 1e-4. This prevents large unstable updates at the very start of training when the adapter weights are random and unpredictable.

**Optimizer (adamw_8bit)**
AdamW adapts the learning rate individually for each parameter based on its recent history of updates — parameters that change a lot get smaller updates, parameters that change little get larger ones. The 8-bit version from bitsandbytes stores optimizer states in 8-bit precision reducing memory usage significantly with no meaningful quality loss.

### Full Training Configuration

| Parameter | Value |
|---|---|
| Base model | sarvamai/sarvam-1 |
| Framework | Unsloth + HuggingFace TRL |
| Hardware | Google Colab T4 GPU (paid compute) |
| Quantization | 4-bit NF4 with double quantization |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | q, k, v, o, gate, up, down proj |
| Trainable parameters | 8,798,208 of 502,830,976 (1.75%) |
| Batch size | 2 × 8 accumulation = 16 effective |
| Epochs | 3 |
| Total steps | 183 |
| Learning rate | 1e-4 |
| LR scheduler | Cosine |
| Warmup steps | 20 |
| Weight decay | 0.01 |
| Gradient clipping | 0.3 |
| Optimizer | adamw_8bit |
| Max sequence length | 512 tokens |
| Training time | ~15 minutes |

---

## Training Results

### Loss Curve

| Step | Training Loss | Validation Loss |
|---|---|---|
| 50 | 0.9795 | 0.9544 |
| 100 | 0.8547 | 0.8858 |
| 150 | 0.7587 | 0.8700 |
| 183 (final) | 0.7454 | 0.8680 |

Training loss decreased steadily from 0.98 to 0.75 across all 183 steps. The validation loss tracked closely without diverging which confirms the model was generalising to unseen examples rather than memorising the training data.

---

## Evaluation Results

### Quantitative Metrics (evaluated on 30 held-out test examples)

| Metric | Base Model | Fine-tuned Model |
|---|---|---|
| BLEU Score (avg) | 0.2042 | 0.2118 |
| BLEU Improvement | — | +3.7% |
| Avg response length (words) | 115.6 | 159.4 |
| Empty or incoherent responses | 0 | 0 |
| Test examples evaluated | 30 | 30 |

BLEU (Bilingual Evaluation Understudy) measures how much the generated text overlaps with the reference answer in terms of word sequences. Scores in the 0.2-0.3 range are normal and expected for open-ended generation tasks — the model is not copying the reference word for word, it is generating its own valid Hindi response. We used character-level BLEU which is more appropriate for Hindi due to its complex word morphology.

### Per-Domain Response Length (qualitative test prompts)

| Domain | Base Model | Fine-tuned | Change |
|---|---|---|---|
| Indian culture and history | 47 words | 179 words | +281% |
| General knowledge | 3 words | 205 words | +6733% |
| Everyday instructions | 203 words | 208 words | +2% |
| Math reasoning | 94 words | 185 words | +97% |
| Polite conversation | 62 words | 208 words | +235% |

### Qualitative Comparison

**Question: भारत का सबसे बड़ा रेगिस्तान कौन सा है?**

❌ **Base model:** `Sahara Desert.`

✅ **Fine-tuned model:** *भारत में थार रेगिस्तान देश का सबसे बड़ा रेगिस्तान माना जाता है जो राजस्थान राज्य में स्थित है। यहाँ गर्मियों में बेहद गर्म होती हैं जबकि सर्दियाँ बहुत ठंडी रहती हैं। यहाँ वनस्पति कम पाई जाती है लेकिन पशु जीवन भरपूर होता है जैसे ऊँट, भेड़ और बकरियाँ...*

---

**Question: महात्मा गांधी के बारे में बताइए।**

❌ **Base model:** Brief mixed Hindi-English mention of 1947 independence. Ends abruptly at 47 words.

✅ **Fine-tuned model:** 179-word structured response covering Gandhi's philosophy of ahimsa, role in the freedom struggle, principles of truth and service, global influence and significance in Indian culture — all in fluent natural Hindi.

---

**Question: दोस्त से झगड़ा हो जाए तो क्या करें?**

❌ **Base model:** Generic 3-point list in Hindi, shallow advice, no cultural context. 62 words.

✅ **Fine-tuned model:** Detailed 208-word response covering calm communication, respectful tone, giving time, rebuilding trust — directly reflects the polite conversation training domain.

---

## How the Dataset Influenced Model Behaviour

**1. Response completeness**
Every training example had a minimum 60-word response. The model learned this pattern. The base model frequently gave one-line answers — sometimes just three words. The fine-tuned model consistently produces detailed multi-paragraph responses averaging 159 words.

**2. Language consistency**
The training data had zero English contamination. The base model mixed Hindi and English frequently — answering a Hindi question about India's largest desert with "Sahara Desert" in English. The fine-tuned model responds entirely in Hindi.

**3. Cultural and domain accuracy**
For Indian culture, history and geography questions the fine-tuned model gives contextually accurate and relevant answers because 407 training examples specifically covered Indian history, festivals, freedom fighters and geography. The model learned to associate the right Hindi knowledge with the right type of question.

---

## Repository Structure

```
Task2_Hindi/
├── dataset_clean.jsonl          # 1221 clean Hindi instruction examples
├── Task2_Hindi_Finetune.ipynb   # Full training and evaluation notebook
├── sarvam_adapter/              # Saved LoRA adapter weights
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer files
└── README.md                    # This file
```

---

## Task Requirements Coverage

| Requirement | Where it is addressed |
|---|---|
| Original dataset not from benchmarks | Created from scratch — generation and manual review pipeline |
| Targets a non-dominant language | Hindi — 600M+ speakers but severely underrepresented in LLMs |
| Data collection process described | Semi-automated pipeline section above with all four steps |
| Dataset structure and size described | 1221 examples, 6 domains, JSONL format fully documented |
| Why dataset is useful | Covers everyday Hindi use cases not served by existing models |
| Fine-tune a pretrained LLM | Sarvam-1 2B fine-tuned with LoRA |
| Document training setup | Full hyperparameter table with plain-language explanations |
| Document fine-tuning method | LoRA explained with mathematical intuition and all parameters |
| Demonstrate on new test inputs | Notebook Cell 10 — clean inference cell ready for live demo |
| Explain dataset influence | Response length +38%, BLEU +3.7%, qualitative examples above |

---

## Key Technical Decisions

**Why Sarvam-1 and not Qwen or Llama?**
We tested Qwen2.5-0.5B and Qwen2.5-1.5B extensively. Both produced nonsensical Hindi — confusing रेगिस्तान (desert) with रेजिमेंट (regiment) even in full float16 precision without any quantization. Sarvam-1 was the only freely available model that produced coherent, grammatically correct Hindi out of the box, making it the correct base for our task.

**Why LoRA and not full fine-tuning?**
The entire 2B Sarvam-1 model in float16 uses approximately 4GB of VRAM just to load. Full fine-tuning requires storing gradients and optimizer states for all 2B parameters — far exceeding available GPU memory. LoRA reduced trainable parameters to 8.8M (1.75%) making training feasible in 15 minutes while still producing meaningful adaptation to Hindi instruction following.

**Why 4-bit quantization?**
Loading Sarvam-1 in float16 uses approximately 4GB of VRAM. With 4-bit NF4 double quantization this drops to approximately 1.2GB, freeing over 2GB of VRAM for gradient computation, batch storage and optimizer states during training.

**Why character-level BLEU?**
Hindi has complex word morphology — the same root word appears in many forms depending on gender, tense and grammatical case. Word-level BLEU would miss many valid matches between generated text and reference text. Character-level BLEU captures partial word matches and is more appropriate for morphologically rich languages like Hindi.

---

*Built as part of the Deep Learning course, Vilnius University.*
