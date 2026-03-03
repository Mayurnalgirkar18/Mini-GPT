# 🚀 Mini GPT using (TensorFlow)

A Decoder-Only Transformer (GPT-style) language model implemented completely from scratch using TensorFlow and Keras.

This project demonstrates a deep understanding of transformer architecture, causal self-attention, and autoregressive text generation without using high-level NLP libraries like HuggingFace.

---

##  Project Overview

This project implements a simplified version of the architecture introduced in **"Attention Is All You Need"** and later used in models like **GPT (Generative Pretrained Transformer)**.

The model is trained as a character-level language model and generates text autoregressively using causal masking.

---

##  What Is GPT?

GPT (Generative Pretrained Transformer) is a decoder-only transformer architecture that:

* Uses masked multi-head self-attention
* Predicts the next token given previous tokens
* Generates text autoregressively
* Applies positional encoding to retain sequence order

This implementation recreates those core ideas from scratch.

---

##  Architecture

The model includes:

* Token Embedding Layer
* Positional Embedding
* Multiple Decoder Blocks

  * Masked Multi-Head Self-Attention
  * Feed Forward Network (MLP)
  * Residual Connections
  * Layer Normalization
* Final Linear Projection Layer

##  Model Configuration

* Embedding Dimension: (e.g. 128 / 256)
* Number of Heads: (e.g. 4 / 8)
* Number of Decoder Layers: (e.g. 2 / 4)
* Sequence Length: (e.g. 100)
* Optimizer: Adam
* Loss: Sparse Categorical Crossentropy
* Training: Teacher Forcing

---

##  Features Implemented

✔ Character-level tokenization
✔ Custom causal attention mask
✔ Multi-head self-attention using Keras
✔ Decoder-only transformer blocks
✔ Residual connections + LayerNorm
✔ Autoregressive text generation
✔ Temperature sampling
✔ Loss curve visualization

---

## Training

The model is trained using next-token prediction:

Input:

```
The quick brown fox
```

Target:

```
he quick brown fox 
```

Training Objective:
Minimize cross-entropy loss between predicted and actual next tokens.

---

##  Text Generation

Text is generated token-by-token using temperature sampling:

```python
predictions = logits[:, -1, :] / temperature
predicted_id = tf.random.categorical(predictions, num_samples=1)
```

Temperature controls randomness:

* Low temperature → More deterministic
* High temperature → More creative

---

## 📈 Sample Output

Example generated text:

```
The king said that the war was not over,
and the people of the land began to gather...
```

(Note: Output quality depends on training dataset size and epochs.)

---

## Project Structure

```
mini-gpt-from-scratch-tensorflow/
│
├── mini_gpt_training.ipynb
├── README.md
├── requirements.txt
└── sample_outputs/
```

---

##  Installation

Clone the repository:

```bash
git clone https://github.com/your-username/mini-gpt-from-scratch-tensorflow.git
cd mini-gpt-from-scratch-tensorflow
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the notebook:

```bash
jupyter notebook mini_gpt_training.ipynb
```

---

##  Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib

---

##  Key Learnings

* Deep understanding of transformer internals
* Practical implementation of causal masking
* Training autoregressive language models
* Implementing decoder-only architecture
* Building NLP models without high-level libraries

---

##  Future Improvements

* Add learning rate scheduler
* Implement gradient clipping
* Add Top-k / Nucleus sampling
* Train on larger dataset
* Add validation perplexity tracking
* Convert to subword tokenization (BPE)

---


