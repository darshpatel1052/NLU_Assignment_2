# NLU Assignment 2 —

**Roll Number:** B23CM1054  
**Course:** Natural Language Understanding (NLU)

---

## Problem 1: Word2Vec on IIT Jodhpur Corpus

### Objective

Build a custom text corpus by scraping the IIT Jodhpur website (iitj.ac.in), and train Word2Vec models on it using both the Gensim library and a from-scratch NumPy implementation. Evaluate the learned word embeddings via nearest-neighbour queries, word analogies, and dimensionality-reduced visualizations (PCA, t-SNE).

### Approach

1. **Data Collection** — A BFS web crawler (`scrape_data.py`) starts from ~90 hand-picked seed URLs across all IIT Jodhpur departments, schools, offices, and institutional pages. It harvests HTML text from up to 250 pages and downloads ~50 curated PDFs (annual reports, academic regulations, NIRF data, placement brochures, etc.) using streaming downloads with size limits. Only English content is retained.

2. **Preprocessing** (`preprocess.py`) — Raw scraped text is cleaned through multiple regex passes: URL/email removal, HTML tag stripping, non-ASCII filtering, and navigation junk removal. The cleaned text is sentence-tokenized using NLTK, then word-tokenized with stopword removal (including domain-specific words like "iit", "http", "menu"). Tokens shorter than 3 characters or non-alphabetic are dropped. The final corpus has **~450K tokens**, **~26K sentences**, and a vocabulary of **~27K unique words** across **51 source documents** (~3.6 MB).

3. **Training with Gensim** (`train_word2vec.py`) — Six Word2Vec configurations are trained: 3 CBOW (with hierarchical softmax) and 3 Skip-Gram (with negative sampling), each at embedding dimensions 50, 100, and 200 with varying window sizes (3, 5, 7).

4. **Training from Scratch** (`word2vec_scratch.py`) — A pure NumPy implementation of Word2Vec with:
   - CBOW and Skip-Gram architectures
   - Negative sampling with noise distribution (freq^0.75)
   - Subsampling of frequent words
   - Linear learning rate decay
   - Manual SGD on the skip-gram/CBOW objectives

5. **Analysis** (`analysis.py`) — Nearest-neighbour search and 5 analogy experiments (e.g., "undergraduate : bachelor :: postgraduate : ?") run on all 6 Gensim models.

6. **Comparison** (`compare_models.py`) — Side-by-side comparison of Gensim vs scratch models on training time, neighbour quality, analogy accuracy, and cosine similarity spread. Gensim is ~50-70x faster due to C optimizations; scratch Skip-Gram (SG_C) achieves reasonable quality (e.g., phd → fellowship, research → areas).

7. **Visualization** (`visualize.py`) — PCA and t-SNE plots of grouped word embeddings (academic programmes, research terms, student life, departments) plus word cloud generation.

### Directory Structure

```
Problem1/
├── scrape_data.py          # BFS web crawler + PDF downloader for iitj.ac.in
├── preprocess.py           # Text cleaning, tokenization, corpus stats & wordcloud
├── train_word2vec.py       # Train 6 Word2Vec configs using Gensim
├── word2vec_scratch.py     # Word2Vec from scratch (CBOW + Skip-Gram, negative sampling)
├── analysis.py             # Nearest neighbours & analogy experiments (Gensim models)
├── compare_models.py       # Gensim vs scratch side-by-side comparison
├── visualize.py            # PCA, t-SNE scatter plots & wordcloud
├── requirements.txt        # Python dependencies
├── corpus/
│   ├── raw/                # Raw scraped text files (51 files)
│   └── cleaned_corpus.txt  # Final preprocessed corpus (~3.6 MB, one sentence per line)
├── models/                 # Saved trained models (.model for Gensim, .pkl for scratch)
├── outputs/                # Generated plots, stats, and analysis JSON files
└── report.pdf              # Detailed report with results and discussion
```

### How to Run

```bash
cd Problem1

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Step 1: Scrape data from iitj.ac.in (~15-20 min)
python scrape_data.py

# Step 2: Preprocess raw text into cleaned corpus
python preprocess.py

# Step 3: Train Word2Vec models using Gensim
python train_word2vec.py

# Step 4: Train Word2Vec from scratch using NumPy
python word2vec_scratch.py

# Step 5: Semantic analysis (neighbours + analogies)
python analysis.py

# Step 6: Compare Gensim vs scratch implementations
python compare_models.py

# Step 7: Generate PCA, t-SNE visualizations
python visualize.py
```

---

## Problem 2: Character-Level Name Generation

### Objective

Train character-level recurrent neural networks to generate human-like names. Compare three architectures — Vanilla RNN, Bidirectional LSTM, and RNN with causal self-attention — on generation quality metrics (novelty, diversity).

### Approach

1. **Dataset** — A list of ~900 training names (`TrainingNames.txt`). Each name is represented as a sequence of characters with special `<SOS>` (start), `<EOS>` (end), and `<PAD>` tokens. The vocabulary is all unique characters in the dataset (~55 tokens).

2. **Vanilla RNN** (`model_vanilla_rnn.py`) — A 2-layer stacked Elman RNN built from scratch (not using `nn.RNN`). Each `VanillaRNNCell` computes h_t = tanh(W_ih · x_t + W_hh · h_{t-1} + b). Features: character embedding layer (dim=64), hidden size 128, dropout between layers, and a final linear projection to vocabulary logits.

3. **Bidirectional LSTM** (`model_bilstm.py`) — A 2-layer BiLSTM with manually implemented `LSTMCell` (4-gate architecture: input, forget, candidate, output gates). Processes sequences both left-to-right and right-to-left, concatenating forward and backward hidden states. Uses cell state for long-range memory, addressing the vanishing gradient problem.

4. **RNN + Attention** (`model_rnn_attention.py`) — A 2-layer stacked RNN with causal (masked) self-attention on top. After the RNN produces hidden states for all timesteps, a learned query projection attends over past hidden states using scaled dot-product attention with a lower-triangular causal mask (no future peeking). The attention context is concatenated with RNN outputs before the final linear layer.

5. **Training** (`train.py`) — All models are trained with Adam optimizer (lr=0.003), cross-entropy loss (ignoring padding), gradient clipping (max_norm=5.0), and 100 epochs with batch size 64.

6. **Evaluation** (`evaluate.py`) — Autoregressive character-by-character generation starting from `<SOS>`, sampling from the output distribution with temperature scaling. Metrics: **novelty rate** (% of generated names not in training set) and **diversity** (unique names / total generated).

### Directory Structure

```
Problem2/
├── train.py               # Training loop for all three architectures
├── evaluate.py            # Name generation & evaluation metrics
├── model_vanilla_rnn.py   # 2-layer stacked Vanilla RNN (from-scratch cells)
├── model_bilstm.py        # 2-layer Bidirectional LSTM (from-scratch cells)
├── model_rnn_attention.py # RNN + causal self-attention
├── utils.py               # NameDataset class & collate function for batching
├── TrainingNames.txt      # Training data (list of names, one per line)
├── generated_samples.txt  # Example generated names from evaluation
├── checkpoints/           # Saved model weights (.pth files)
└── report.pdf             # Detailed report with architecture diagrams and results
```

### How to Run

```bash
cd Problem2

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch tqdm matplotlib

# Train all three models (Vanilla RNN, BiLSTM, RNN+Attention)
python train.py --epochs 100 --batch_size 64 --lr 0.003

# Evaluate: generate 1000 names per model & compute metrics
python evaluate.py --samples 1000
```

---

## Reports

- **Problem1/report.pdf** — Corpus curation pipeline, Word2Vec training (Gensim & scratch), semantic analysis, analogy experiments, and Gensim vs scratch comparison with visualizations.
- **Problem2/report.pdf** — Model architectures (with diagrams), training curves, generated name samples, and novelty/diversity comparison across RNN, BiLSTM, and Attention models.
