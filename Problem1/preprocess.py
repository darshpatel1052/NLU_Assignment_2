# preprocess.py
# turns messy scraped text into something usable
# also dumps some stats + word cloud (mostly to see if things look sane)

import os
import re
import json
from collections import Counter

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib
matplotlib.use("Agg")  # needed if running without display (learned this the hard way)
import matplotlib.pyplot as plt


# download nltk stuff (sometimes this fails silently but usually fine)
for pkg in ["punkt", "punkt_tab", "stopwords", "averaged_perceptron_tagger"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass


# paths
RAW_DIR = os.path.join("corpus", "raw")
CORPUS_PATH = os.path.join("corpus", "cleaned_corpus.txt")
STATS_PATH = os.path.join("outputs", "stats.json")
WORDCLOUD_PATH = os.path.join("outputs", "wordcloud.png")

os.makedirs("outputs", exist_ok=True)


# basic stopwords + some extra junk that showed up in scraping
STOP_WORDS = set(stopwords.words("english"))

# honestly this list came from just noticing garbage tokens repeatedly
DOMAIN_STOP = {
    "iit", "jodhpur", "iitj", "http", "https", "www", "co",
    "com", "org", "edu", "php", "html", "page", "click",
    "home", "back", "next", "menu", "search", "skip",
    "source", "na", "nil", "log", "login", "view",
    "also", "may", "shall", "one", "two", "three",
}

STOP_WORDS.update(DOMAIN_STOP)


def clean_text(text):
    # this part is basically regex cleanup hell

    # urls
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # emails (not super common but still)
    text = re.sub(r"\S+@\S+", " ", text)

    # leftover html
    text = re.sub(r"<[^>]+>", " ", text)

    # things like &amp;
    text = re.sub(r"&[a-zA-Z]+;", " ", text)

    # drop non-ascii (yeah this removes hindi etc, but keeping it simple)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    # navigation junk from websites
    text = re.sub(
        r"\b(home|menu|nav|skip to|cookie|copyright|all rights reserved)\b",
        " ", text, flags=re.IGNORECASE
    )

    # keep only basic stuff
    text = re.sub(r"[^a-zA-Z0-9\s\.\,\!\?]", " ", text)

    # cleanup spaces (this always ends up needed)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_sentence(sentence):
    # lowercasing first (word2vec usually assumes this anyway)
    sentence = sentence.lower()

    tokens = word_tokenize(sentence)

    # filtering rules — tweaked these a bit after seeing bad vocab
    filtered = []
    for tok in tokens:
        if not tok.isalpha():
            continue
        if len(tok) < 3:
            continue
        if tok in STOP_WORDS:
            continue
        filtered.append(tok)

    return filtered


def process_file(filepath):
    # read file → clean → split → tokenize

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()

    cleaned = clean_text(raw)

    # sentence split (good enough, not perfect)
    sentences = sent_tokenize(cleaned)

    tokenised = []

    for sent in sentences:
        tokens = tokenize_sentence(sent)

        # skip tiny sentences (usually garbage anyway)
        if len(tokens) < 3:
            continue

        tokenised.append(tokens)

    return tokenised


def generate_wordcloud(word_freq, save_path):
    # mostly just to eyeball if preprocessing worked

    freq_filtered = {
        w: c for w, c in word_freq.items() if w not in STOP_WORDS
    }

    wc = WordCloud(
        width=1200,
        height=700,
        background_color="white",
        max_words=200,
        colormap="viridis",
    ).generate_from_frequencies(freq_filtered)

    plt.figure(figsize=(14, 8))
    plt.imshow(wc)
    plt.axis("off")

    plt.title("Most Frequent Words", fontsize=18)

    plt.savefig(save_path)
    plt.close()

    print(f"  wordcloud saved -> {save_path}")


def main():
    print("Preprocessing corpus...")

    raw_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".txt")]

    if not raw_files:
        print("No raw files found. Did you run scraper?")
        return

    print(f"Found {len(raw_files)} files")

    all_sentences = []
    doc_token_counts = {}

    for fname in sorted(raw_files):
        fpath = os.path.join(RAW_DIR, fname)

        print(f"\nProcessing {fname}")

        sentences = process_file(fpath)

        token_count = sum(len(s) for s in sentences)

        doc_token_counts[fname] = {
            "sentences": len(sentences),
            "tokens": token_count,
        }

        print(f"  {len(sentences)} sentences | {token_count} tokens")

        all_sentences.extend(sentences)

    # save cleaned corpus
    print(f"\nWriting corpus -> {CORPUS_PATH}")

    os.makedirs(os.path.dirname(CORPUS_PATH), exist_ok=True)

    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        for tokens in all_sentences:
            f.write(" ".join(tokens) + "\n")

    # flatten tokens
    all_tokens_flat = []
    for sent in all_sentences:
        all_tokens_flat.extend(sent)

    token_freq = Counter(all_tokens_flat)
    vocab = set(all_tokens_flat)

    stats = {
        "total_documents": len(raw_files),
        "total_sentences": len(all_sentences),
        "total_tokens": len(all_tokens_flat),
        "vocabulary_size": len(vocab),
        "top_50_words": token_freq.most_common(50),
        "per_document": doc_token_counts,
    }

    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)

    # quick print (not super formatted, just for quick look)
    print("\nStats:")
    print(f"  docs: {stats['total_documents']}")
    print(f"  sentences: {stats['total_sentences']}")
    print(f"  tokens: {stats['total_tokens']}")
    print(f"  vocab: {stats['vocabulary_size']}")

    print("\nTop words:")
    for word, cnt in token_freq.most_common(20):
        print(f"  {word}: {cnt}")

    # generate word cloud
    print("\nMaking wordcloud...")
    generate_wordcloud(token_freq, WORDCLOUD_PATH)

    print("\nDone.")


if __name__ == "__main__":
    main()