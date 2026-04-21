# Long Document Summarization using LED + LoRA

##  Overview
This project focuses on **abstractive summarization of long documents** using a combination of Long-Transformer architectures and parameter-efficient fine-tuning. We use the **LED (Longformer Encoder-Decoder)** model with **LoRA (Low-Rank Adaptation)** to efficiently handle long sequences while reducing training cost and memory usage.

The goal is to generate coherent and factual summaries from long scientific documents (arXiv papers) containing thousands of tokens.

---

##  Key Idea
Traditional Transformer models struggle with long documents due to quadratic attention complexity. This project addresses that by:

- Using **LED model** for sparse + global attention
- Applying **LoRA fine-tuning** to reduce trainable parameters
- Training on a large-scale scientific dataset
- Evaluating using standard summarization metrics (ROUGE)

---

##  Model Architecture
- Base Model: `allenai/led-base-16384`
- Fine-tuning method: LoRA
- Attention mechanism: Sparse + Global Attention
- Task: Sequence-to-Sequence Summarization

---

##  Dataset
We use the **arXiv summarization dataset**:
- Input: Full scientific article
- Target: Abstract summary
- Total samples used: 20,000
- Split:
  - Train: 16,000
  - Validation: 2,000
  - Test: 2,000

---

##  Features
- Long-document processing (up to 1024–16k tokens)
- Efficient fine-tuning using LoRA
- Global attention masking for LED
- Training + evaluation + inference pipeline
- ROUGE-based evaluation

---

##  Evaluation Metrics
We evaluate model performance using:
- ROUGE-1 -- 0.2813876886976951
- ROUGE-2 -- 0.07384127923517919
- ROUGE-L -- 0.19928041761885346
- Rouge-Sum -- 0.22905967734131022

Both quantitative and qualitative analysis are performed.

