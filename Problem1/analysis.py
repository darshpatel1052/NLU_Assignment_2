# analysis.py
# loads all 6 trained word2vec models and runs:
# 1. nearest neighbour search for some probe words
# 2. analogy experiments (a:b :: c:?)
# saves results to outputs/semantic_analysis.json

import os
import json
from gensim.models import Word2Vec

# paths
MODELS_DIR = "models"
RESULTS_PATH = os.path.join("outputs", "semantic_analysis.json")
os.makedirs("outputs", exist_ok=True)

# words we want to find neighbours for
PROBE_WORDS = ["research", "student", "phd", "exam"]

# analogy experiments
# format: A:B :: C:? which means find the word closest to vec(B) - vec(A) + vec(C)
# gensim wants it as positive=[C, B] and negative=[A]
ANALOGIES = [
    {
        "label": "undergraduate : bachelor :: postgraduate : ?",
        "positive": ["postgraduate", "bachelor"],
        "negative": ["undergraduate"],
    },
    {
        "label": "professor : research :: student : ?",
        "positive": ["student", "research"],
        "negative": ["professor"],
    },
    {
        "label": "department : course :: faculty : ?",
        "positive": ["faculty", "course"],
        "negative": ["department"],
    },
    {
        "label": "examinations : grade :: semester : ?",
        "positive": ["semester", "grade"],
        "negative": ["examinations"],
    },
    {
        "label": "bachelor : undergraduate :: master : ?",
        "positive": ["master", "undergraduate"],
        "negative": ["bachelor"],
    },
]


def load_model(name):
    # load a saved model by name
    path = os.path.join(MODELS_DIR, f"{name}.model")
    if not os.path.exists(path):
        print(f"  Model not found: {path}")
        return None
    return Word2Vec.load(path)


def nearest_neighbours(model, word, topn=5):
    # get the topn most similar words
    # if the exact word isnt in vocab, try a partial match
    if word not in model.wv:
        for w in model.wv.key_to_index:
            if word in w:
                word = w
                break
        else:
            return None
    return model.wv.most_similar(word, topn=topn)


def run_analogy(model, positive, negative, topn=5):
    # run 3CosAdd analogy: positive - negative
    # returns list of (word, score) tuples or an error string
    missing = [w for w in positive + negative if w not in model.wv]
    if missing:
        return f"Words not in vocab: {missing}"
    try:
        results = model.wv.most_similar(
            positive=positive, negative=negative, topn=topn
        )
        return results
    except Exception as e:
        return str(e)


def analyse_model(model, model_name):
    # run all the analysis for one model
    arch = "CBOW" if "CBOW" in model_name else "Skip-gram"
    print(f"\n{'='*50}")
    print(f"  {model_name}  ({arch})")
    print(f"{'='*50}")

    result = {
        "model": model_name,
        "architecture": arch,
        "vector_size": model.wv.vector_size,
        "vocab_size": len(model.wv.key_to_index),
        "nearest_neighbours": {},
        "analogies": [],
    }

    # nearest neighbours for each probe word
    print("\n  Nearest Neighbours:")
    for word in PROBE_WORDS:
        nbrs = nearest_neighbours(model, word, topn=5)
        if nbrs:
            result["nearest_neighbours"][word] = [
                {"word": w, "similarity": round(float(s), 4)} for w, s in nbrs
            ]
            nbr_str = ", ".join(f"{w}({s:.3f})" for w, s in nbrs)
            print(f"    {word:<12} -> {nbr_str}")
        else:
            result["nearest_neighbours"][word] = None
            print(f"    {word:<12} -> [NOT IN VOCAB]")

    # analogy experiments
    print("\n  Analogies:")
    for ana in ANALOGIES:
        res = run_analogy(model, ana["positive"], ana["negative"], topn=5)
        entry = {"label": ana["label"], "results": []}
        if isinstance(res, list):
            entry["results"] = [
                {"word": w, "similarity": round(float(s), 4)} for w, s in res
            ]
            top5 = ", ".join(f"{w}({s:.3f})" for w, s in res[:5])
            print(f"    {ana['label']:<40} -> {top5}")
        else:
            entry["error"] = res
            print(f"    {ana['label']:<40} -> ERROR: {res}")
        result["analogies"].append(entry)

    return result


def main():
    print("=" * 60)
    print("Semantic Analysis")
    print("=" * 60)

    # find all saved models
    model_names = [f.replace(".model", "")
                   for f in os.listdir(MODELS_DIR) if f.endswith(".model")]

    if not model_names:
        print("No models found. Run train_word2vec.py first.")
        return

    model_names = sorted(model_names)
    print(f"\nFound {len(model_names)} models: {model_names}")

    all_results = []
    for name in model_names:
        model = load_model(name)
        if model is None:
            continue
        result = analyse_model(model, name)
        all_results.append(result)

    # save everything
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\nResults saved to {RESULTS_PATH}")

    # print a quick comparison of the two main models
    print("\n\n--- CBOW_B vs SG_B comparison ---")
    cbow_res = next((r for r in all_results if r["model"] == "CBOW_B"), None)
    sg_res = next((r for r in all_results if r["model"] == "SG_B"), None)
    if cbow_res and sg_res:
        for word in PROBE_WORDS:
            print(f"\n  '{word}'")
            cb = cbow_res["nearest_neighbours"].get(word) or []
            sg = sg_res["nearest_neighbours"].get(word) or []
            cb_str = [f"{x['word']}({x['similarity']:.3f})" for x in cb]
            sg_str = [f"{x['word']}({x['similarity']:.3f})" for x in sg]
            print(f"    CBOW:      {', '.join(cb_str) or 'N/A'}")
            print(f"    Skip-gram: {', '.join(sg_str) or 'N/A'}")


if __name__ == "__main__":
    main()
