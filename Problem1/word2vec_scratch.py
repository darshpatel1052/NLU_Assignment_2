import os
import json
import time
import pickle
import numpy as np
from collections import Counter


# basic file paths
CORPUS_PATH = os.path.join("corpus", "cleaned_corpus.txt")
MODELS_DIR = "models"
RESULTS_PATH = os.path.join("outputs", "scratch_training_results.json")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# read corpus + build vocab
def load_corpus(path, min_count=2):
    print("Loading corpus...")

    sentences = []
    word_counts = Counter()

    with open(path, "r") as f:
        for line in f:
            tokens = line.strip().split()

            # skip super short lines (not useful for context)
            if len(tokens) >= 3:
                sentences.append(tokens)
                word_counts.update(tokens)

    # remove rare words (they just add noise)
    kept_words = [(w, c) for w, c in sorted(word_counts.items()) if c >= min_count]

    # assign id to each word
    vocab = {w: i for i, (w, _) in enumerate(kept_words)}

    # store counts in array form (needed later for sampling)
    word_freq = np.zeros(len(vocab))
    for w, c in kept_words:
        word_freq[vocab[w]] = c

    # filter sentences to only keep vocab words
    filtered = []
    for sent in sentences:
        filtered_sent = [w for w in sent if w in vocab]

        # again skip too short sequences
        if len(filtered_sent) >= 3:
            filtered.append(filtered_sent)

    idx_to_word = {i: w for w, i in vocab.items()}

    print(f"Sentences: {len(filtered)}, Vocab: {len(vocab)}")

    return filtered, vocab, idx_to_word, word_freq


# build distribution for negative sampling
# trick: freq^0.75 → reduces dominance of very common words
def build_noise_distribution(word_freq):
    powered = word_freq ** 0.75
    return powered / powered.sum()


# randomly drop very frequent words
# helps model not over-focus on words like "the"
def subsample_sentence(sentence, vocab, word_freq, total_tokens, threshold=1e-5):
    kept = []

    for w in sentence:
        idx = vocab[w]
        freq_ratio = word_freq[idx] / total_tokens

        # more frequent → lower chance of keeping
        keep_prob = min(1.0, np.sqrt(threshold / freq_ratio))

        if np.random.random() < keep_prob:
            kept.append(w)

    return kept


# CBOW: predict center from surrounding context
def get_cbow_pairs(sentence, vocab, window):
    pairs = []
    indices = [vocab[w] for w in sentence]

    for i in range(len(indices)):
        center = indices[i]

        # randomly shrink window (like original word2vec)
        w = np.random.randint(1, window + 1)

        start = max(0, i - w)
        end = min(len(indices), i + w + 1)

        # collect context words
        context = [indices[j] for j in range(start, end) if j != i]

        if context:
            pairs.append((context, center))

    return pairs


# Skip-gram: predict context from center
def get_skipgram_pairs(sentence, vocab, window):
    pairs = []
    indices = [vocab[w] for w in sentence]

    for i in range(len(indices)):
        center = indices[i]

        w = np.random.randint(1, window + 1)

        start = max(0, i - w)
        end = min(len(indices), i + w + 1)

        for j in range(start, end):
            if j != i:
                pairs.append((center, indices[j]))

    return pairs


# sigmoid with clipping (prevents overflow issues)
def sigmoid(x):
    x = np.clip(x, -10, 10)
    return 1.0 / (1.0 + np.exp(-x))


class CBOWScratch:
    def __init__(self, vocab_size, embed_dim):
        # input embeddings (what we usually call "word vectors")
        self.W_in = (np.random.rand(vocab_size, embed_dim) - 0.5) / embed_dim

        # output embeddings (used during training)
        self.W_out = np.zeros((vocab_size, embed_dim))

        self.vocab_size = vocab_size

    def train_pair_ns(self, context_indices, center_idx, neg_indices, lr):
        # average all context vectors → single representation
        h = np.mean(self.W_in[context_indices], axis=0)

        # positive example (real center word)
        score_pos = np.dot(self.W_out[center_idx], h)
        sig_pos = sigmoid(score_pos)

        # (sigmoid - 1) comes from gradient of log-sigmoid
        grad_pos = (sig_pos - 1.0) * lr

        # this accumulates gradient for input embeddings
        grad_in = grad_pos * self.W_out[center_idx]

        # update output embedding of correct word
        self.W_out[center_idx] -= grad_pos * h

        # negative samples (fake words)
        for neg_idx in neg_indices:
            score_neg = np.dot(self.W_out[neg_idx], h)
            sig_neg = sigmoid(score_neg)

            grad_neg = sig_neg * lr

            grad_in += grad_neg * self.W_out[neg_idx]
            self.W_out[neg_idx] -= grad_neg * h

        # distribute gradient back to all context words
        for c in context_indices:
            self.W_in[c] -= grad_in / len(context_indices)

    def get_embedding(self, word_idx):
        return self.W_in[word_idx]


class SkipGramScratch:
    def __init__(self, vocab_size, embed_dim):
        self.W_in = (np.random.rand(vocab_size, embed_dim) - 0.5) / embed_dim
        self.W_out = np.zeros((vocab_size, embed_dim))

        self.vocab_size = vocab_size

    def train_pair_ns(self, center_idx, context_idx, neg_indices, lr):
        # center word embedding
        h = self.W_in[center_idx]

        # positive context word
        score_pos = np.dot(self.W_out[context_idx], h)
        sig_pos = sigmoid(score_pos)

        grad_pos = (sig_pos - 1.0) * lr

        grad_in = grad_pos * self.W_out[context_idx]

        self.W_out[context_idx] -= grad_pos * h

        # negative samples
        for neg_idx in neg_indices:
            score_neg = np.dot(self.W_out[neg_idx], h)
            sig_neg = sigmoid(score_neg)

            grad_neg = sig_neg * lr

            grad_in += grad_neg * self.W_out[neg_idx]
            self.W_out[neg_idx] -= grad_neg * h

        # update center embedding
        self.W_in[center_idx] -= grad_in

    def get_embedding(self, word_idx):
        return self.W_in[word_idx]


# cosine similarity (standard way to compare embeddings)
def cosine_similarity(a, b):
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return np.dot(a, b) / norm


# find closest words in embedding space
def most_similar(model, word, vocab, idx_to_word, topn=5):
    if word not in vocab:
        return None

    word_idx = vocab[word]
    word_vec = model.get_embedding(word_idx)

    scores = []
    for idx in range(model.vocab_size):
        if idx == word_idx:
            continue

        sim = cosine_similarity(word_vec, model.get_embedding(idx))
        scores.append((idx, sim))

    scores.sort(key=lambda x: x[1], reverse=True)

    return [(idx_to_word[i], round(s, 4)) for i, s in scores[:topn]]


# analogy: a : b :: c : ?
def analogy(model, a, b, c, vocab, idx_to_word, topn=5):
    if any(w not in vocab for w in [a, b, c]):
        return "Missing words in vocab"

    # classic vector arithmetic
    vec = (model.get_embedding(vocab[b])
           - model.get_embedding(vocab[a])
           + model.get_embedding(vocab[c]))

    exclude = {vocab[a], vocab[b], vocab[c]}
    scores = []

    for idx in range(model.vocab_size):
        if idx in exclude:
            continue

        sim = cosine_similarity(vec, model.get_embedding(idx))
        scores.append((idx, sim))

    scores.sort(key=lambda x: x[1], reverse=True)

    return [(idx_to_word[i], round(s, 4)) for i, s in scores[:topn]]


# main training loop
def train_model(sentences, vocab, idx_to_word, word_freq, config):
    name = config["name"]
    model_type = config["type"]
    embed_dim = config["vector_size"]
    window = config["window"]
    negative = config["negative"]
    epochs = config["epochs"]
    lr_start = config.get("lr", 0.025)

    vocab_size = len(vocab)
    total_tokens = int(word_freq.sum())

    print(f"\nTraining {name} ({model_type})")

    # pick model
    model = CBOWScratch(vocab_size, embed_dim) if model_type == "cbow" else SkipGramScratch(vocab_size, embed_dim)

    noise_dist = build_noise_distribution(word_freq)

    start_time = time.time()

    for epoch in range(epochs):
        # simple linear decay
        lr = lr_start * (1.0 - epoch / epochs)
        lr = max(lr, lr_start * 0.0001)

        order = np.random.permutation(len(sentences))
        pair_count = 0

        for sent_idx in order:
            sent = subsample_sentence(sentences[sent_idx], vocab, word_freq, total_tokens)

            if len(sent) < 3:
                continue

            if model_type == "cbow":
                pairs = get_cbow_pairs(sent, vocab, window)
                for context_indices, center_idx in pairs:
                    neg_indices = np.random.choice(vocab_size, size=negative, replace=False, p=noise_dist)
                    model.train_pair_ns(context_indices, center_idx, neg_indices, lr)
                    pair_count += 1
            else:
                pairs = get_skipgram_pairs(sent, vocab, window)
                for center_idx, context_idx in pairs:
                    neg_indices = np.random.choice(vocab_size, size=negative, replace=False, p=noise_dist)
                    model.train_pair_ns(center_idx, context_idx, neg_indices, lr)
                    pair_count += 1

        print(f"Epoch {epoch+1}/{epochs} done | lr={lr:.5f} | pairs={pair_count}")

    total_time = time.time() - start_time
    print(f"Training done in {total_time:.2f}s")

    # save everything needed to reload later
    model_path = os.path.join(MODELS_DIR, f"{name}_scratch.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({
            "W_in": model.W_in,
            "W_out": model.W_out,
            "vocab": vocab,
            "idx_to_word": idx_to_word,
            "config": config,
        }, f)

    return model, total_time