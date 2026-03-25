<div align="center">
  
# 🧠 Natural Language Understanding (NLU) - Assignment 2

**Word2Vec from Dynamic Web Crawling & Character-Level Sequence Models for Name Generation.**

[![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch)]()
[![NumPy](https://img.shields.io/badge/NumPy-1.24-013243?style=flat-square&logo=numpy)]()
[![Gensim](https://img.shields.io/badge/Gensim-4.3-F15F22?style=flat-square)]()

*Course: Natural Language Understanding (NLU), Semester 6*  
*Roll Number: B23CM1054*

---

<img src="Problem1/outputs/wordcloud.png" width="800" alt="Corpus Wordcloud">

</div>

## 📑 Table of Contents

- [Overview](#-overview)
- [Problem 1: Word Representations on IITJ Corpus](#-problem-1-word-representations-on-iitj-corpus)
  - [1. Data Collection \& Preprocessing](#1-data-collection--preprocessing)
  - [2. Theoretical Background (CBOW \& Skip-Gram)](#2-theoretical-background-cbow--skip-gram)
  - [3. From-Scratch NumPy Implementation](#3-from-scratch-numpy-implementation)
  - [4. Semantic Analysis \& Visualizations](#4-semantic-analysis--visualizations)
- [Problem 2: Character-Level Name Generation](#-problem-2-character-level-name-generation)
  - [1. Vanilla RNN Architecture](#1-vanilla-rnn-architecture)
  - [2. Bidirectional LSTM Architecture](#2-bidirectional-lstm-architecture)
  - [3. RNN + Causal Self-Attention](#3-rnn--causal-self-attention)
  - [4. Quantitative \& Qualitative Evaluation](#4-quantitative--qualitative-evaluation)
- [Installation \& Usage](#-how-to-run)

---

## 📖 Overview

This repository contains two comprehensive, from-scratch NLP projects:
1. **Word2Vec on Custom Domain**: A full pipeline that web-scrapes a real-world institutional domain (`iitj.ac.in`), processes text from HTML and complex PDFs (generating a pristine academic corpus), and trains dense embedding models (both natively in NumPy and using Gensim) to capture rich academic semantic geometry.
2. **Character-Level RNN Generation**: A comparative exploration of sequential architectures (Vanilla RNN vs. Bidirectional LSTM vs. Causal Self-Attention RNN) designed fundamentally from scratch using atomic tensor operations to learn and generate highly realistic Indian names, highlighting the architectural struggle against mode collapse and vanishing gradients.

---

## 🏗️ Problem 1: Word Representations on IITJ Corpus

### 1. Data Collection & Preprocessing
To learn meaningful, localized embeddings, we scraped our own semantic environment: the IIT Jodhpur institutional framework.

**Data Sourcing (`scrape_data.py`):**
* **BFS Web Crawler**: Starts from ~110 hand-picked seed URLs mapping to all departments, labs, and offices. Harvested up to 250 HTML web pages.
* **Deep Document Mining**: ~50 high-value English PDFs were automatically streamed and text-stripped using `PyMuPDF`. Documents include *Academic Regulations (UG/PG)*, *Annual Reports (2008-2024)*, the *IIT Act (1961)*, and *NIRF rankings*.

**Cleaning Pipeline (`preprocess.py`):**
Raw text (`~809K words`) undergoes a strict regiment: Regex removal of URLs/emails, stripping remnant script tags, enforcing printable ASCII (blocking erroneous scraped Hindi text), tokenizing via NLTK, lowercasing, and domain-specific stopword removal (e.g. pruning terms like "iitj", "login", "menu" which clutter geometric proximity). Tokens under 3 characters are discarded.

> **Final Corpus Statistics:** `26,123` sentences | `450,793` tokens | `16,990` training vocabulary size (after `min_count=2` filtering).

### 2. Theoretical Background (CBOW & Skip-Gram)

We trained 6 Word2Vec configurations exploring dimensions (50, 100, 200) and window configurations across two fundamental architectures.

**Continuous Bag-of-Words (CBOW)**
CBOW predicts a target center word $w_t$ from its surrounding symmetric context. Given a window size $c$, context vectors are averaged and pushed through softmax. CBOW is known to train quickly and provide smooth representations.
$$P(w_t \mid \mathcal{C}) = \frac{\exp(\mathbf{u}_{w_t}^\top \hat{v})}{\sum_{w \in V} \exp(\mathbf{u}_w^\top \hat{v})}$$

**Skip-Gram with Negative Sampling (SGNS)**
Skip-gram inversely predicts contextual window words given a singular center word $w_t$. Rather than expensive multi-class softmax over the entire 17,000-word vocabulary, our model distinguishes real context from $k$ noise words drawn from a modified distribution:
$$\mathcal{L}_{\text{NS}} = \log \sigma(\mathbf{u}_{w_O}^\top \mathbf{v}_{w_I}) + \sum_{i=1}^{k} \mathbb{E}_{w_i \sim P_n(w)}\left[\log \sigma(-\mathbf{u}_{w_i}^\top \mathbf{v}_{w_I})\right]$$
*Where $P_n(w) \propto f(w)^{3/4}$ is utilized to heavily discount the raw frequency probabilities of dominant conjunctions and common noise verbs.*

### 3. From-Scratch NumPy Implementation

A parallel Word2Vec framework (`word2vec_scratch.py`) was fully written using low-level NumPy matrix operations to compare backpropagation math against Gensim's industrial C-backend.

**Mathematical Features Included:**
* **Subsampling of Frequent Words:** Common words are probabilistic dropped during training pairing using $P_{\text{keep}}(w) = \sqrt{t / f(w)}$ where $t = 10^{-5}$ as defined in Mikolov et al.
* **Pure Gradient SGD:** Implements discrete dot-product binary cross-entropy gradients per context target without hierarchical grouping shortcuts.
* **Linear Learning Rate Decay:** Standard $\alpha_t = \alpha_0 \cdot (1 - t/T)$ mapping down to 0 over identical epoch counts.

<div align="center">
  <img src="Problem1/outputs/gensim_vs_scratch_time.png" width="700" alt="Training Time Comparison">
  <p><i><b>Figure 1:</b> Gensim's C engine provides a staggering 60x-130x speed optimization. Our Python backpropagation loops every (center, context) pair distinctly, whereas Gensim utilizes batched multi-threading.</i></p>
</div>

### 4. Semantic Analysis & Visualizations

**Evaluation via Nearest Neighbours & 3CosAdd Analogies**
We query the geometry of the embedding space using Cosine Similarity ($\frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|\|\mathbf{b}\|}$). 

| Target Word | Gensim Skip-Gram Top-3 | Scratch Skip-Gram Top-3 | Takeaway |
| :--- | :--- | :--- | :--- |
| `research` | activities (0.71), materials (0.66), drdo (0.63) | areas (0.95), groups (0.95), applications (0.94) | Both correctly map to organizational R&D structure. |
| `student` | scholars (0.73), ably (0.71), rachit (0.69) | awarded (0.96), first (0.91), part (0.89) | Gensim begins overfitting noise (memorizing exact student names like "Rachit" from awards lists). |
| `exam` | entrance (0.76), syllabus (0.74), qualify (0.73) | download (0.99), finanical (0.99), trimester (0.99) | Gensim SG learns perfect exam associations. |

> **Hardest Learned Analogy:** The from-scratch Skip-Gram model perfectly solved `undergraduate` : `bachelor` :: `postgraduate` : **`master`** (similarity: `0.98`), proving raw gradient descent functions properly despite training time constraints.

<div align="center">
  <br>
  <img src="Problem1/outputs/tsne_cbow_vs_sgns.png" width="900" alt="t-SNE Embeddings">
  <p><i><b>Figure 2:</b> t-SNE plot categorizing academic terms. Notice how Skip-gram (right) forces sharper localized islands separating discrete administrative areas from standard syllabus structures compared to CBOW's generalized distribution.</i></p>
</div>

---

## 🧬 Problem 2: Character-Level Name Generation

### Objective & Architectures Built
We model 1,000 diverse Indian full names textually, treating strings as serialized character lists. Given a character prefix, models autoregressively output probability logits for the subsequent letter across a vocab of 52 characters.

### 1. Vanilla RNN Architecture
A 2-layer stacked Elman network computing hidden states through pure matrix dot products.
* **Equation:** $h_t = \tanh(W_{ih}x_t + b_{ih} + W_{hh}h_{t-1} + b_{hh})$
* **Result:** Successfully learns phonotactics and typical naming structures (first name + surname split), easily capturing specific morphological suffix boundaries.

### 2. Bidirectional LSTM Architecture
Features a heavily scaled Encoder-Decoder format mapping the sequence bidirectionally to drastically improve long-range dependencies in extensive multi-string names.
* **Cell Structure Base:** 
  * $f_t = \sigma(W_f \cdot [x_t, h_{t-1}] + b_f)$
  * $i_t = \sigma(W_i \cdot [x_t, h_{t-1}] + b_i)$
  * $h_t = o_t \odot \tanh(c_t \cdot f_t + i_t)$
* **Result:** Highly rigorous name formations heavily replicating South Indian surname length, but suffers a slight penalty in raw unique diversity by overfitting training structures.

### 3. RNN + Causal Self-Attention
Rather than using standard encoder alignment, this implementation injects a learned attention query targeting previous localized RNN hidden states.
* **Mechanism:** $Context = \text{softmax}\left(\text{Mask}\left(\frac{Q K^\top}{\sqrt{d_k}}\right)\right) \cdot V$
* Lower-triangular masking ensures the autoregressive prediction cannot hallucinate forward. 

### 4. Quantitative & Qualitative Evaluation

Tested dynamically via temperature-controlled sampling for 1000 generated sequences.

| Model | Novelty Rate *(Not in Train Set)* | Diversity *(Unique/Total)* |
| :--- | :---: | :---: |
| **Vanilla RNN** | 99.50% | 0.9990 |
| **Bidirectional LSTM** | 90.80% | 0.9830 |
| **RNN + Causal Attention** | **98.20%** | **0.9960** |

**Generated Samples Showcase:**
* **Attention (Flawless Structure):** *Pankaj Kumar, Karishma Vanikam, Saurabh Jaiswat, Onian Rajkumar, Jeraish Kumar.*
* **Vanilla RNN (Creative Morphologies):** *Dhanana Gowda, Sokar Bharthen, Mohanna Dhunder.*
* **BiLSTM (Complex Sequencing):** *Bhoonondath Sen, Chandrasekaran Rajeshnan, Arunachalam Tharsan.*

**Failure Modes Noted:**
* **Portmanteau Cross-Mapping:** Models occasionally collide entirely disjoint regional conventions creating valid sounding but wildly mixed strings (e.g. *Dhanasamban Kannasime*).
* **Vocabulary Leakage:** Certain occupational strings mapped within the raw training set strings (e.g. "Singer", "Actor") are sometimes hallucinated as standard surnames by the BiLSTM.

---

## 📂 Directory Structure

```text
Assignment_2/
├── Problem1/
│   ├── scrape_data.py          # BFS web crawler + Stream PDF downloader
│   ├── preprocess.py           # Text cleaning, NLTK constraints, ASCII filtering
│   ├── train_word2vec.py       # C-optimized multi-threaded Gensim deployment
│   ├── word2vec_scratch.py     # Pure Numpy discrete gradient processing
│   ├── visualize.py            # PCA, t-SNE scatter plots & Wordcloud rendering
│   └── corpus/                 # Preprocessed dataset (cleaned_corpus.txt, 450K tokens)
│
└── Problem2/
    ├── train.py                # Normalized Adam / Cross-Entropy trainer over all networks
    ├── evaluate.py             # Autoregressive sequence sampling + Diversity testing
    ├── model_vanilla_rnn.py    # Standard Elman cell structure logic
    ├── model_bilstm.py         # Advanced multi-gated Bidirectional module logic
    ├── model_rnn_attention.py  # RNN + Scaled Dot-Product Masked Attention integration
    └── TrainingNames.txt       # Raw curated dataset (1000 multiregional Indian entries)
```

---

## 🚀 How to Run

Ensure Python 3.10+ is available. We recommend keeping virtual environments localized.

### 1. Execute Problem 1 (Word2Vec)
```bash
cd Problem1
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run sequentially:
python scrape_data.py          # Harvest HTML/PDFs natively from IITJ domain
python preprocess.py           # Compile strings into structured, analytical corpus
python train_word2vec.py       # Render CBOW + SG Gensim representations rapidly
python word2vec_scratch.py     # Validate math manually via NumPy framework
python analysis.py             # Test nearest neighbours, 3CosAdd arithmetic
python visualize.py            # Generate high-resolution PCA / t-SNE images
```

### 2. Execute Problem 2 (Character-Level Prediction)
```bash
cd Problem2
python3 -m venv venv && source venv/bin/activate
pip install torch tqdm matplotlib

# Run Sequentially:
python train.py --epochs 100 --batch_size 64 --lr 0.003
python evaluate.py --samples 1000
```

---
<p align="center">
  <i>Maintained and authored for academic requirement validation.</i>
</p>
