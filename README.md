# Hindi Instruction-Following Dataset and LLM Fine-Tuning

**Course:** Deep Learning — Individual Task 2  
**Language:** Hindi (हिन्दी) — a non-dominant language in LLM research  
**Model:** Sarvam-1 fine-tuned with LoRA  
**Dataset:** 1200+ original Hindi instruction-response pairs  

---

## What This Project Does

Most large language models are trained primarily on English text. Even models that claim multilingual support tend to perform poorly on Hindi — giving short, incomplete, or sometimes nonsensical answers. This project creates an original Hindi instruction-following dataset from scratch and uses it to fine-tune Sarvam-1, an open-source model built specifically for Indian languages.

The goal is simple: make the model better at answering Hindi questions in a natural, detailed, and helpful way.

---

## Why Hindi?

Hindi is spoken by over 600 million people making it one of the most spoken languages in the world. Despite this, it is significantly underrepresented in instruction-tuned language models compared to English. Most open-source models either respond in English when asked in Hindi, give very short answers, or produce grammatically incorrect Hindi.

This makes Hindi a strong candidate for a low-resource fine-tuning project — there is a real gap between the language's prevalence and its representation in AI systems.

---

## Dataset

### How It Was Created

The dataset was created through a semi-automated annotation pipeline:

1. **Domain selection** — Five core domains were chosen that cover everyday Hindi language use and are genuinely useful for Hindi speakers
2. **Generation** — Examples were generated using Claude and ChatGPT with strict prompts requiring minimum 60-word responses in natural Hindi
3. **Review and validation** — Every example was manually reviewed for grammatical correctness, factual accuracy, and response quality
4. **Quality check** — A Python script validated every line for correct JSON format, proper schema, response length, and language consistency

### Dataset Structure

Each example follows the standard instruction-following JSONL format:

```json
{"messages": [
  {"role": "system", "content": "तुम एक सहायक हिन्दी भाषा के सहायक हो।"},
  {"role": "user", "content": "महात्मा गांधी के बारे में बताइए।"},
  {"role": "assistant", "content": "महात्मा गांधी भारत की स्वतंत्रता आंदोलन के प्रमुख नेता थे..."}
]}
```

### Dataset Statistics

| Domain | Examples | Description |
|---|---|---|
| Indian culture and history | 240 | Freedom fighters, historical events, monuments, festivals |
| General knowledge Q&A | 240 | Science, geography, world facts explained in Hindi |
| Everyday instructions | 240 | Cooking, household tasks, practical how-to guides |
| Simple reasoning and math | 240 | Word problems, logic puzzles, arithmetic in Hindi |
| Polite conversation and etiquette | 240 | Social situations, greetings, conflict resolution |
| **Total** | **1200+** | **All original, no benchmark reuse** |

### Quality Validation Results

| Metric | Result |
|---|---|
| Total examples | 1220 |
| Valid JSON | 100% |
| Correct schema | 100% |
| Short responses flagged | 2 (removed) |
| English contamination | 0% |
| Final clean examples | 1218 |

### Why This Dataset Is Useful

Hindi speakers interact with AI assistants in Hindi but most models are not optimized for this. This dataset specifically covers domains that Hindi speakers use daily — asking about Indian history and culture, following cooking or household instructions, solving everyday math problems, and navigating social situations. All of these require natural fluent Hindi which is exactly what this dataset provides.

---

## Model Fine-Tuning

### Base Model

**Sarvam-1** (`sarvamai/sarvam-1`) was chosen as the base model. It is a 2B parameter model built by Sarvam AI specifically for Indian languages, trained on a large corpus of Hindi, Tamil, Telugu, Kannada, Malayalam, Bengali, Gujarati, Marathi, Punjabi and Odia text. Unlike general multilingual models, Sarvam-1 already has strong Hindi language understanding — fine-tuning it gives us a meaningful improvement rather than trying to teach Hindi from scratch.

### Fine-Tuning Method — LoRA

Instead of retraining the entire 2B parameter model which would require expensive hardware, we used **LoRA (Low-Rank Adaptation)** — an efficient fine-tuning technique that freezes the original model weights and adds small trainable adapter matrices.

The mathematical idea behind LoRA: rather than updating a large weight matrix W directly, it learns two small matrices A and B where A×B approximates the update needed. This means only about 1.7% of parameters are actually trained.

**LoRA Configuration:**

| Parameter | Value | Reason |
|---|---|---|
| Rank (r) | 16 | Good balance of capacity vs overfitting |
| Alpha | 32 | Scaling factor = 2× rank |
| Dropout | 0.05 | Regularization for small dataset |
| Target modules | q, k, v, o, gate, up, down proj | Full transformer coverage |
| Trainable parameters | ~8.8M of 502M (1.75%) | Efficient adaptation |

### Training Setup

| Parameter | Value |
|---|---|
| Framework | Unsloth + HuggingFace TRL |
| Hardware | Google Colab T4 GPU (free tier) |
| Quantization | 4-bit NF4 with double quantization |
| Batch size | 2 per device × 8 accumulation = 16 effective |
| Epochs | 3 |
| Learning rate | 1e-4 with cosine scheduler |
| Warmup steps | 20 |
| Weight decay | 0.01 |
| Gradient clipping | max_grad_norm = 0.3 |
| Optimizer | adamw_8bit |
| Max sequence length | 512 tokens |
| Training time | ~15 minutes on T4 |

### Training Loss Curve

| Step | Training Loss | Validation Loss |
|---|---|---|
| 50 | 0.9795 | 0.9544 |
| 100 | 0.8547 | 0.8858 |
| 150 | 0.7587 | 0.8700 |
| 183 (final) | 0.7454 | 0.8680 |

The training loss decreased steadily from ~0.98 to ~0.75 showing the model was learning from the dataset. Validation loss tracked training loss closely indicating no overfitting.

---

## Results

### Quantitative Evaluation (BLEU Score on 30 held-out examples)

| Metric | Base Model | Fine-tuned Model |
|---|---|---|
| BLEU Score (avg) | 0.2042 | 0.2118 |
| BLEU Improvement | — | +3.7% |
| Avg response length (words) | 115.6 | 159.4 |
| Empty/incoherent responses | 0 | 0 |
| Test examples evaluated | 30 | 30 |

### Per-Domain Response Length Comparison

| Domain | Base Model (words) | Fine-tuned (words) | Change |
|---|---|---|---|
| Indian culture and history | 47 | 179 | +281% |
| General knowledge | 3 | 205 | +6733% |
| Everyday instructions | 203 | 208 | +2% |
| Math reasoning | 94 | 185 | +97% |
| Polite conversation | 62 | 208 | +235% |

The general knowledge domain shows the most dramatic improvement — the base model gave only 3 words ("Sahara Desert") while the fine-tuned model gave a detailed 205-word response about the Thar Desert including geography, climate, wildlife and cultural significance.

### Qualitative Comparison

**Question: भारत का सबसे बड़ा रेगिस्तान कौन सा है?**

❌ Base model: `Sahara Desert.`

✅ Fine-tuned model: *भारत में थार रेगिस्तान देश का सबसे बड़ा रेगिस्तान माना जाता है जो राजस्थान राज्य में स्थित है... यहाँ गर्मियों में बेहद गर्म होती हैं जबकि सर्दियाँ बहुत ठंडी रहती हैं इसलिए यहाँ वनस्पति कम पाई जाती है लेकिन पशु जीवन भरपूर होता है...*

---

**Question: महात्मा गांधी के बारे में बताइए।**

❌ Base model: Brief mention of independence in 1947, mixed Hindi-English response, ends abruptly.

✅ Fine-tuned model: Detailed structured response covering Gandhi's philosophy of ahimsa, his role in the freedom struggle, his principles of truth and service, his global influence, and his significance in Indian culture — all in fluent natural Hindi.

---

## How the Dataset Influenced Model Behavior

The fine-tuned model shows three clear improvements over the base model:

**1. Response completeness** — The base model frequently gave one-line answers or answered in English. After fine-tuning the model consistently gives detailed multi-paragraph Hindi responses because every training example had a minimum 60-word response.

**2. Language consistency** — The base model mixed Hindi and English frequently. The fine-tuned model responds entirely in Hindi because the training data had zero English contamination.

**3. Domain-specific knowledge** — For Indian culture and history questions the fine-tuned model gives culturally accurate and contextually relevant answers because the dataset specifically covered Indian history, festivals, freedom fighters and geography.

---

## How to Run

### Requirements
- Google Colab with T4 GPU (free tier is sufficient)
- Google Drive with dataset file uploaded

### Steps

```python
# 1. Install dependencies
!pip install -q unsloth transformers datasets trl peft accelerate bitsandbytes

# 2. Load fine-tuned model for inference
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "path/to/saved/adapter",
    max_seq_length = 512,
    dtype = torch.float16,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

# 3. Ask a question
def ask(question):
    text = (
        f"### System:\nतुम एक सहायक हिन्दी भाषा के सहायक हो।\n\n"
        f"### User:\n{question}\n\n"
        f"### Assistant:\n"
    )
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.3,
        repetition_penalty=1.15
    )
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## Repository Structure

```
Task2_Hindi/
├── dataset_clean.jsonl          # 1218 clean Hindi instruction examples
├── Task2_Hindi_Finetune.ipynb   # Full training notebook
├── sarvam_adapter/              # Saved LoRA adapter weights
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer files
└── README.md                    # This file
```

---

## Task Requirements Coverage

| Requirement | How it is addressed |
|---|---|
| Original dataset not from benchmarks | Created from scratch using structured generation and manual review |
| Targets a non-dominant language | Hindi — underrepresented despite 600M+ speakers |
| Data collection process described | Semi-automated pipeline with Claude/ChatGPT generation and manual validation |
| Dataset structure and size described | 1218 examples, 5 domains, JSONL format documented above |
| Why dataset is useful | Covers everyday Hindi use cases not well served by existing models |
| Fine-tune a pretrained LLM | Sarvam-1 2B fine-tuned with LoRA |
| Document training setup | Full hyperparameter table included above |
| Document fine-tuning method | LoRA explanation with mathematical intuition included |
| Demonstrate on new test inputs | Notebook Cell 10 — clean inference cell ready for live demo |
| Explain dataset influence | Response length +38%, BLEU +3.7%, qualitative examples above |

---

## Key Technical Decisions Explained

**Why Sarvam-1 and not other models?**  
We tested Qwen2.5-0.5B, Qwen2.5-1.5B and the original HuggingFace Qwen weights. All produced nonsensical Hindi — confusing रेगिस्तान (desert) with रेजिमेंट (regiment) even in full precision. Sarvam-1 was the only model that produced coherent Hindi out of the box, making it the right base for fine-tuning.

**Why LoRA and not full fine-tuning?**  
The entire 2B model in float16 requires ~4GB of VRAM just to load. Full fine-tuning would require storing gradients and optimizer states for all 2B parameters which exceeds the 14.5GB T4 limit. LoRA reduces trainable parameters to 8.8M — about 1.75% of the total — making training feasible on free hardware in under 15 minutes.

**Why 4-bit quantization?**  
Loading the 2B model in float16 uses ~4GB. With 4-bit NF4 double quantization this drops to ~1.2GB, freeing over 2GB of VRAM for the training process, batch storage and gradient computation.

---

*Built as part of the Deep Learning course at Vilnius University.*
