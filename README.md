# 🏛️ PolicyGPT

### A Domain-Specific Decoder-Only Transformer for Indian Public Policy Text

PolicyGPT is an educational/research project that builds a **GPT-style decoder-only Transformer from scratch** and trains it on Indian public-policy-related text.

The goal of this project is not to use an existing LLM or fine-tune a commercial model, but to understand and implement the core components behind modern autoregressive language models — including **tokenization, embeddings, causal self-attention, multi-head attention, feed-forward networks, residual connections, LayerNorm, next-token prediction, and autoregressive text generation**.

> **Note:** PolicyGPT is an experimental educational language model and should not be treated as an authoritative source of government policy, legal, financial, or political advice.

---

## 🚀 Project Overview

Large Language Models such as GPT generate text by predicting the next token given the previous tokens.

PolicyGPT follows the same fundamental idea, but on a much smaller scale and with a domain-specific corpus focused on **Indian public policy**.

```text
Policy Documents
       ↓
Data Collection
       ↓
Text Cleaning & Preprocessing
       ↓
Tokenization
       ↓
Training Dataset
       ↓
Decoder-Only Transformer
       ↓
Next-Token Prediction
       ↓
Model Training
       ↓
Autoregressive Generation
       ↓
PolicyGPT
```

---

## 🎯 Objectives

The main objectives of this project are:

* Understand how GPT-style language models work internally.
* Implement a **decoder-only Transformer from scratch**.
* Train a model using **next-token prediction**.
* Understand causal/masked self-attention.
* Experiment with tokenization and context length.
* Train a domain-specific model on public-policy text.
* Evaluate the model using validation loss and perplexity.
* Implement autoregressive text generation.
* Build a simple interface for interacting with the trained model.
* Understand how a language model can be integrated into an application workflow.

---

## 🧠 Architecture

PolicyGPT uses a **decoder-only Transformer architecture**.

```text
Input Tokens
     ↓
Token Embeddings
     +
Positional Information
     ↓
┌──────────────────────────────┐
│     Transformer Block        │
│                              │
│  Layer Normalization         │
│          ↓                   │
│  Masked Multi-Head           │
│  Self-Attention              │
│          ↓                   │
│  Residual Connection         │
│          ↓                   │
│  Layer Normalization         │
│          ↓                   │
│  Feed Forward Network        │
│          ↓                   │
│  Residual Connection         │
└──────────────────────────────┘
              × N
     ↓
Linear Language Model Head
     ↓
Logits
     ↓
Softmax
     ↓
Next Token Prediction
```

### Core Components

The model is implemented around the following concepts:

* Token Embeddings
* Positional Embeddings
* Query, Key and Value projections
* Scaled Dot-Product Attention
* Causal/Masked Self-Attention
* Multi-Head Attention
* Feed-Forward Network
* Layer Normalization
* Residual Connections
* Transformer Decoder Blocks
* Linear Language Model Head
* Cross-Entropy Loss
* Autoregressive Generation

---

## 🔥 Why Decoder-Only?

PolicyGPT uses a decoder-only architecture because the task is **next-token prediction**.

For example:

```text
Input:
The government introduced a new

Target:
policy
```

The model learns:

```text
"The government introduced a new"
                         ↓
                      "policy"
```

During generation, the model repeatedly predicts the next token:

```text
The
 ↓
government
 ↓
introduced
 ↓
a
 ↓
new
 ↓
policy
 ↓
...
```

The causal attention mask ensures that a token cannot look at future tokens during training.

---

## 🔍 Attention Mechanism

For every token, the model generates:

```text
Query (Q)
Key   (K)
Value (V)
```

Attention scores are calculated using:

```text
Attention(Q,K,V)
=
softmax(QKᵀ / √dₖ)V
```

### Why divide by √dₖ?

As the dimensionality of the keys increases, the dot products can become large.

Dividing by:

```text
√dₖ
```

keeps the values in a more stable range before applying softmax.

### Causal Mask

PolicyGPT uses a lower-triangular causal mask:

```text
1  0  0  0
1  1  0  0
1  1  1  0
1  1  1  1
```

This prevents the model from accessing future tokens.

---

## 🧩 Tokenization

The text must first be converted into numerical tokens.

Conceptually:

```text
"public policy"
       ↓
["public", "policy"]
       ↓
[152, 781]
```

The project can experiment with tokenization approaches such as:

* Character-level tokenization
* Word-level tokenization
* Subword/BPE tokenization

BPE/subword tokenization is particularly useful because it provides a balance between vocabulary size and the ability to represent unknown or uncommon words.

---

## 📚 Dataset

The training corpus is based on publicly available Indian public-policy-related documents and text.

Potential sources include:

* Government policy documents
* Public reports
* Official publications
* Policy papers
* Parliamentary/public institutional documents
* Other legally accessible public-domain or openly available policy-related text

### Data Pipeline

```text
Raw Documents
      ↓
Text Extraction
      ↓
Cleaning
      ↓
Normalization
      ↓
Duplicate Removal
      ↓
Train / Validation Split
      ↓
Tokenization
      ↓
Training Sequences
```

Data quality is important because the model learns patterns directly from the training corpus.

---

## 🏗️ Model Configuration

The exact configuration may change depending on available hardware.

A small experimental configuration may look like:

| Parameter           |                  Example |
| ------------------- | -----------------------: |
| Architecture        | Decoder-only Transformer |
| Number of Layers    |                      4–6 |
| Attention Heads     |                      6–8 |
| Embedding Dimension |                     ~384 |
| Context Length      |                  256–512 |
| Objective           |    Next-token prediction |
| Loss                |            Cross-Entropy |
| Optimizer           |                    AdamW |

These values are intentionally small enough for experimentation on limited hardware.

---

## 🏋️ Training

PolicyGPT is trained using **next-token prediction**.

Given a sequence:

```text
The government introduced a new policy
```

the training examples can be viewed as:

```text
Input:
The government introduced a new

Target:
policy
```

The model predicts a probability distribution over the vocabulary.

The training objective is to minimize **cross-entropy loss**.

Conceptually:

```text
Tokens
  ↓
Transformer
  ↓
Logits
  ↓
Cross-Entropy Loss
  ↓
Backpropagation
  ↓
Parameter Update
```

---

## 📊 Evaluation

The primary evaluation metrics are:

### Training Loss

Measures how well the model predicts tokens from the training data.

### Validation Loss

Measures how well the model performs on unseen validation data.

This helps identify overfitting.

### Perplexity

Perplexity is derived from the language-model loss and provides an intuitive measure of how uncertain the model is when predicting the next token.

Lower perplexity generally indicates better predictive performance on the evaluation corpus.

> Perplexity should be compared between models evaluated on the same or comparable dataset and tokenization setup.

---

## ✍️ Text Generation

After training, PolicyGPT generates text autoregressively.

```text
Prompt
  ↓
Tokenize
  ↓
Transformer
  ↓
Predict next token
  ↓
Append token
  ↓
Repeat
  ↓
Generated text
```

Example:

```text
Prompt:
The government introduced a policy to

Generated:
The government introduced a policy to improve
public infrastructure and strengthen access to
essential services...
```

Generated output depends heavily on the training corpus and model size.

---

## 🎛️ Sampling Strategies

The generation process can experiment with:

### Temperature

Controls randomness.

```text
Low temperature
→ More predictable output

High temperature
→ More diverse output
```

### Top-K Sampling

Restricts the next-token selection to the top K most probable tokens.

### Top-P Sampling

Selects from the smallest group of tokens whose cumulative probability exceeds a chosen threshold.

These techniques allow experimentation with the trade-off between **deterministic and diverse generation**.

---

## 🖥️ Application

A lightweight interface can be built using Streamlit.

```text
┌────────────────────────────────────┐
│            PolicyGPT               │
├────────────────────────────────────┤
│                                    │
│  Enter a policy-related prompt     │
│                                    │
│  [____________________________]     │
│                                    │
│             Generate               │
│                                    │
│  Generated Text:                   │
│  ................................  │
│  ................................  │
│                                    │
└────────────────────────────────────┘
```

The UI is intended for experimentation and demonstration of the trained model.

---

## 📁 Project Structure

A possible project structure:

```text
PolicyGPT/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
│
├── tokenizer/
│   ├── tokenizer.py
│   └── vocab/
│
├── model/
│   ├── embeddings.py
│   ├── attention.py
│   ├── transformer.py
│   └── model.py
│
├── training/
│   ├── dataset.py
│   ├── train.py
│   └── evaluate.py
│
├── generation/
│   └── generate.py
│
├── app/
│   └── streamlit_app.py
│
├── checkpoints/
│
├── requirements.txt
├── config.py
├── README.md
└── .gitignore
```

The exact structure may differ depending on the implementation.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/PolicyGPT.git
cd PolicyGPT
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

For Git Bash:

```bash
source venv/Scripts/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Training

After preparing the dataset:

```bash
python training/train.py
```

The trained model/checkpoint can then be saved for later generation.

---

## 💬 Generate Text

```bash
python generation/generate.py
```

Provide a prompt and let the model generate text autoregressively.

---

## 🌐 Run the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

The application will open in your browser.

---

## 🔬 Experiments

One of the main purposes of PolicyGPT is experimentation.

Possible experiments include:

### Experiment 1 — Context Length

Compare:

```text
128 vs 256 vs 512
```

and observe the effect on:

* Validation loss
* Perplexity
* Training time
* Memory usage

### Experiment 2 — Model Size

Compare smaller and larger Transformer configurations.

### Experiment 3 — Tokenization

Compare:

```text
Character-level
        vs
Subword/BPE
```

### Experiment 4 — Domain Specificity

Compare training on:

```text
General text
      vs
Policy-specific text
```

and observe how domain-specific training affects generated output.

### Experiment 5 — Sampling

Compare different:

* Temperature values
* Top-K values
* Top-P values

---

## 🧠 What I Learned

This project focuses heavily on understanding the internal mechanics of Transformer language models.

Key concepts explored:

* How token embeddings represent text numerically
* How positional information is added
* How Q/K/V are generated
* How attention scores are calculated
* Why scaled dot-product attention is used
* How causal masking prevents information leakage
* Why multiple attention heads are useful
* How residual connections help deep networks
* Why LayerNorm is used
* How FFNs transform token representations
* How logits become token probabilities
* How cross-entropy trains the model
* How backpropagation updates Transformer parameters
* How autoregressive generation works
* How temperature/top-k/top-p affect generation
* How validation loss and perplexity evaluate language models

---

## 🛠️ Tech Stack

**Programming**

* Python

**Deep Learning**

* PyTorch

**Architecture**

* Decoder-Only Transformer
* GPT-style Language Model

**NLP**

* Tokenization
* BPE/Subword Tokenization
* Causal Self-Attention
* Next-Token Prediction

**Application**

* Streamlit

**Version Control**

* Git
* GitHub

---

## 📌 Limitations

PolicyGPT is intentionally a small educational model.

Therefore:

* It is not comparable to large commercial LLMs.
* The training dataset is significantly smaller.
* The model has limited contextual capacity.
* Generated content may be inaccurate or repetitive.
* It does not guarantee factual correctness.
* It should not be used for legal, financial, political, or governmental decision-making.
* Model quality depends strongly on dataset quality and size.

The purpose of the project is **understanding and experimentation**, not building a production-grade foundation model.

---

## 🔮 Future Improvements

Possible future improvements include:

* Better BPE tokenizer
* Larger and cleaner policy corpus
* More Transformer layers
* Larger context window
* Better data deduplication
* Learning-rate scheduling
* Weight tying
* Checkpoint/resume training
* Beam/sampling experiments
* Quantization
* GPU training
* Evaluation benchmarks
* Policy-document retrieval
* RAG-based factual grounding
* Improved Streamlit interface

---

## 📈 Project Roadmap

```text
[x] Understand Transformer architecture
[x] Implement basic attention
[x] Implement masked self-attention
[x] Implement multi-head attention
[ ] Implement complete decoder block
[ ] Build tokenizer
[ ] Prepare policy dataset
[ ] Train language model
[ ] Evaluate validation loss
[ ] Calculate perplexity
[ ] Implement text generation
[ ] Build Streamlit interface
[ ] Run model experiments
[ ] Document results
```

---

## ⚠️ Disclaimer

PolicyGPT is an **educational and research project** created to explore Transformer-based language modeling.

It does not represent the views of any government organization or institution, and its generated text should not be considered official policy information or professional advice.

---

## 👨‍💻 Author

**Aditya Magdum**

B.Tech — Computer Science & Engineering (Data Science)

Interested in:

* Machine Learning
* Deep Learning
* Generative AI
* Transformers
* NLP
* Full-Stack Development

---

## ⭐ If You Find This Project Interesting

Feel free to explore the implementation, experiment with the architecture, and improve the model.

The main goal of this project is simple:

> **Understand what happens inside a GPT-style language model by building one from the ground up.**
