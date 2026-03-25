# train_word2vec.py
# trains a bunch of word2vec configs and logs basic stats
# mostly just experimenting with CBOW vs skip-gram

import os
import json
import time
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


# paths (keeping everything relative so it's easy to move project around)
CORPUS_PATH = os.path.join("corpus", "cleaned_corpus.txt")
MODELS_DIR = "models"
RESULTS_PATH = os.path.join("outputs", "training_results.json")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# 6 configs → just trying a few combinations instead of grid search
# sg=0 → CBOW, sg=1 → skip-gram
# CBOW uses hierarchical softmax here, skip-gram uses negative sampling
CONFIGS = [
    # smaller / faster ones
    {"name": "CBOW_A", "sg": 0, "vector_size": 50,  "window": 3, "negative": 0, "min_count": 2, "epochs": 15},
    {"name": "CBOW_B", "sg": 0, "vector_size": 100, "window": 5, "negative": 0, "min_count": 2, "epochs": 15},
    {"name": "CBOW_C", "sg": 0, "vector_size": 200, "window": 7, "negative": 0, "min_count": 2, "epochs": 20},

    # skip-gram tends to do better for rare words, so trying different k
    {"name": "SG_A",   "sg": 1, "vector_size": 50,  "window": 3, "negative": 5,  "min_count": 2, "epochs": 15},
    {"name": "SG_B",   "sg": 1, "vector_size": 100, "window": 5, "negative": 10, "min_count": 2, "epochs": 15},
    {"name": "SG_C",   "sg": 1, "vector_size": 200, "window": 7, "negative": 15, "min_count": 2, "epochs": 20},
]


# just a few words to sanity check vocab later
# if these are missing → something probably went wrong
PROBE_WORDS = [
    "research", "student", "phd", "exam", "faculty",
    "department", "course", "professor", "academic", "iitj"
]


def load_sentences(path):
    # gensim has this nice lazy loader → doesn't load whole file into memory
    # useful if corpus gets big
    return LineSentence(path)


def train_model(cfg):
    # train a single config and return stats
    print(f"\n  Training {cfg['name']} ...")

    # printing config helps later when comparing runs
    print(f"    sg={cfg['sg']}, dim={cfg['vector_size']}, "
          f"window={cfg['window']}, neg={cfg['negative']}, epochs={cfg['epochs']}")

    start = time.time()

    # core training call
    model = Word2Vec(
        sentences=load_sentences(CORPUS_PATH),

        # embedding size
        vector_size=cfg["vector_size"],

        # how far context looks
        window=cfg["window"],

        # ignore rare words below this frequency
        min_count=cfg["min_count"],

        # 0 = CBOW, 1 = skip-gram
        sg=cfg["sg"],

        # negative sampling only for skip-gram
        negative=cfg["negative"] if cfg["sg"] == 1 else 0,

        # hierarchical softmax for CBOW (instead of negative sampling)
        hs=0 if cfg["sg"] == 1 else 1,

        # parallel workers (can tweak depending on CPU)
        workers=4,

        # number of passes over corpus
        epochs=cfg["epochs"],

        # fixed seed → reproducibility (kinda)
        seed=42,
    )

    elapsed = time.time() - start

    # save model so we don't have to retrain every time
    model_path = os.path.join(MODELS_DIR, f"{cfg['name']}.model")
    model.save(model_path)

    # check vocab size
    vocab_size = len(model.wv.key_to_index)

    # quick check: which important words survived min_count filtering
    available_probes = [w for w in PROBE_WORDS if w in model.wv]

    result = {
        "name": cfg["name"],
        "architecture": "CBOW" if cfg["sg"] == 0 else "Skip-gram",
        "vector_size": cfg["vector_size"],
        "window": cfg["window"],
        "negative_samples": cfg["negative"],
        "epochs": cfg["epochs"],
        "vocab_size": vocab_size,
        "training_time_sec": round(elapsed, 2),
        "probe_words_in_vocab": available_probes,
        "model_path": model_path,
    }

    # printing this is actually helpful when debugging configs
    print(f"    Done in {elapsed:.2f}s | vocab={vocab_size} | probes: {available_probes}")

    return result


def print_results_table(results):
    # quick table view instead of opening json every time
    print(f"\n{'='*90}")
    print(f"{'Name':<10} {'Arch':<12} {'Dim':>5} {'Win':>5} {'Neg':>5} "
          f"{'Epochs':>7} {'Vocab':>8} {'Time(s)':>9}")
    print(f"{'='*90}")

    for r in results:
        print(
            f"{r['name']:<10} {r['architecture']:<12} "
            f"{r['vector_size']:>5} {r['window']:>5} {r['negative_samples']:>5} "
            f"{r['epochs']:>7} {r['vocab_size']:>8} {r['training_time_sec']:>9.2f}"
        )

    print(f"{'='*90}")


def main():
    print("=" * 60)
    print("Training Word2Vec Models")
    print("=" * 60)

    # basic check before starting
    if not os.path.exists(CORPUS_PATH):
        print(f"Corpus not found at {CORPUS_PATH}. Run preprocess.py first.")
        return

    # just counting lines for reference (not strictly needed)
    with open(CORPUS_PATH) as f:
        n_lines = sum(1 for _ in f)

    print(f"\nCorpus: {CORPUS_PATH}  ({n_lines} sentences)")
    print(f"Training {len(CONFIGS)} models...")

    all_results = []

    # loop through configs one by one
    for cfg in CONFIGS:
        result = train_model(cfg)
        all_results.append(result)

    # print summary
    print_results_table(all_results)

    # save results → useful for plotting later
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {RESULTS_PATH}")
    print(f"Models saved to {MODELS_DIR}/")


if __name__ == "__main__":
    main()