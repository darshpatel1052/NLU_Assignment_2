# compare_models.py
# side-by-side comparison of gensim vs from-scratch word2vec models
# loads results from outputs/ and prints a clean comparison table
# also generates a comparison bar chart for training times

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# paths
GENSIM_ANALYSIS = os.path.join("outputs", "semantic_analysis.json")
GENSIM_TRAINING = os.path.join("outputs", "training_results.json")
SCRATCH_RESULTS = os.path.join("outputs", "scratch_training_results.json")
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def print_header(text):
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}")


def compare_training_times(gensim_train, scratch):
    # compare how fast each approach trains
    print_header("Training Time Comparison (seconds)")
    print(f"{'Model':<12} {'Gensim':>10} {'Scratch':>12} {'Ratio':>10}")
    print("-" * 50)

    pairs = []
    for gt in gensim_train:
        name = gt["name"]
        sc = next((s for s in scratch if s["name"] == name), None)
        if sc:
            ratio = sc["training_time_sec"] / gt["training_time_sec"]
            pairs.append((name, gt["training_time_sec"], sc["training_time_sec"], ratio))
            print(f"{name:<12} {gt['training_time_sec']:>10.2f} {sc['training_time_sec']:>12.2f} {ratio:>9.1f}x")

    return pairs


def compare_neighbours(gensim_analysis, scratch, probe_words):
    # compare nearest neighbours for probe words between the two
    print_header("Nearest Neighbours Comparison")

    # we'll focus on the B configs (dim=100, window=5) for a fair comparison
    for arch_prefix in ["CBOW_B", "SG_B"]:
        g_model = next((m for m in gensim_analysis if m["model"] == arch_prefix), None)
        s_model = next((m for m in scratch if m["name"] == arch_prefix), None)

        if not g_model or not s_model:
            continue

        print(f"\n--- {arch_prefix} ---")
        for word in probe_words:
            g_nbrs = g_model["nearest_neighbours"].get(word, [])
            s_nbrs = s_model["nearest_neighbours"].get(word, [])

            if not g_nbrs or not s_nbrs:
                continue

            g_str = ", ".join(f"{n['word']}({n['similarity']:.2f})" for n in g_nbrs[:3])
            s_str = ", ".join(f"{n['word']}({n['similarity']:.2f})" for n in s_nbrs[:3])

            print(f"\n  '{word}':")
            print(f"    Gensim:  {g_str}")
            print(f"    Scratch: {s_str}")

            # check overlap in top-5 words
            g_words = set(n["word"] for n in g_nbrs[:5])
            s_words = set(n["word"] for n in s_nbrs[:5])
            overlap = g_words & s_words
            if overlap:
                print(f"    Overlap: {overlap}")
            else:
                print(f"    Overlap: none")


def compare_analogies(gensim_analysis, scratch):
    # compare analogy results
    print_header("Analogy Comparison (SG_C -- best model)")

    g_model = next((m for m in gensim_analysis if m["model"] == "SG_C"), None)
    s_model = next((m for m in scratch if m["name"] == "SG_C"), None)

    if not g_model or not s_model:
        print("SG_C not found in both result sets, trying SG_B...")
        g_model = next((m for m in gensim_analysis if m["model"] == "SG_B"), None)
        s_model = next((m for m in scratch if m["name"] == "SG_B"), None)

    if not g_model or not s_model:
        print("Cannot compare -- missing models.")
        return

    for g_ana, s_ana in zip(g_model["analogies"], s_model["analogies"]):
        label = g_ana["label"]
        g_results = g_ana.get("results", [])
        s_results = s_ana.get("results", [])

        g_top = g_results[0]["word"] if g_results else "N/A"
        s_top = s_results[0]["word"] if s_results else "N/A"

        g_sim = f"({g_results[0]['similarity']:.3f})" if g_results else ""
        s_sim = f"({s_results[0]['similarity']:.3f})" if s_results else ""

        print(f"\n  {label}")
        print(f"    Gensim:  {g_top} {g_sim}")
        print(f"    Scratch: {s_top} {s_sim}")

        # see if top-3 answers overlap at all
        g_words = set(r["word"] for r in g_results[:3])
        s_words = set(r["word"] for r in s_results[:3])
        common = g_words & s_words
        if common:
            print(f"    Common in top-3: {common}")


def plot_training_comparison(pairs):
    # bar chart: gensim vs scratch training times side by side
    names = [p[0] for p in pairs]
    gensim_times = [p[1] for p in pairs]
    scratch_times = [p[2] for p in pairs]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, gensim_times, width, label="Gensim (C-optimized)",
                   color="#1e78be", edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width/2, scratch_times, width, label="From Scratch (numpy)",
                   color="#e87d44", edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Training Time (seconds)", fontsize=12)
    ax.set_title("Gensim vs From-Scratch Training Time", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)

    # add value labels on top of bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h * 1.1,
                f"{h:.1f}s", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h * 1.1,
                f"{h:.0f}s", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "gensim_vs_scratch_time.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved training time chart to {out_path}")


def compute_similarity_spread(scratch):
    # look at how spread out the cosine similarities are
    # a good model has a wide spread (low similarity for unrelated words)
    # a poorly trained model has everything near 1.0
    print_header("Cosine Similarity Spread (scratch models)")
    print(f"{'Model':<12} {'Avg Top-5 Sim':>15} {'Interpretation':>25}")
    print("-" * 55)

    for m in scratch:
        sims = []
        for word, nbrs in m["nearest_neighbours"].items():
            if nbrs:
                for n in nbrs:
                    sims.append(n["similarity"])
        avg = np.mean(sims) if sims else 0
        # if avg is above 0.99 the model hasn't differentiated well
        if avg > 0.99:
            interp = "under-differentiated"
        elif avg > 0.95:
            interp = "somewhat concentrated"
        elif avg > 0.80:
            interp = "reasonable spread"
        else:
            interp = "good differentiation"
        print(f"{m['name']:<12} {avg:>15.4f} {interp:>25}")


def save_comparison_json(gensim_analysis, gensim_train, scratch, probe_words):
    # save a structured comparison for the report
    comparison = {
        "training_time": [],
        "neighbour_comparison": {},
        "analogy_comparison": [],
    }

    # training times
    for gt in gensim_train:
        sc = next((s for s in scratch if s["name"] == gt["name"]), None)
        if sc:
            comparison["training_time"].append({
                "model": gt["name"],
                "gensim_sec": gt["training_time_sec"],
                "scratch_sec": sc["training_time_sec"],
                "speedup": round(sc["training_time_sec"] / gt["training_time_sec"], 1),
            })

    # neighbour comparison for B models
    for prefix in ["CBOW_B", "SG_B", "SG_C"]:
        g = next((m for m in gensim_analysis if m["model"] == prefix), None)
        s = next((m for m in scratch if m["name"] == prefix), None)
        if g and s:
            comparison["neighbour_comparison"][prefix] = {}
            for word in probe_words:
                gn = g["nearest_neighbours"].get(word, [])
                sn = s["nearest_neighbours"].get(word, [])
                comparison["neighbour_comparison"][prefix][word] = {
                    "gensim_top3": [n["word"] for n in gn[:3]] if gn else [],
                    "scratch_top3": [n["word"] for n in sn[:3]] if sn else [],
                    "overlap_top5": list(
                        set(n["word"] for n in gn[:5]) &
                        set(n["word"] for n in sn[:5])
                    ) if gn and sn else [],
                }

    out = os.path.join(OUTPUT_DIR, "comparison_results.json")
    with open(out, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nSaved comparison JSON to {out}")


def main():
    probe_words = ["research", "student", "phd", "exam"]

    print("=" * 70)
    print("  Gensim vs From-Scratch Word2Vec Comparison")
    print("=" * 70)

    # load everything
    gensim_analysis = load_json(GENSIM_ANALYSIS)
    gensim_train = load_json(GENSIM_TRAINING)
    scratch = load_json(SCRATCH_RESULTS)

    # training time comparison
    pairs = compare_training_times(gensim_train, scratch)

    # plot side-by-side bar chart
    if pairs:
        plot_training_comparison(pairs)

    # neighbour comparison
    compare_neighbours(gensim_analysis, scratch, probe_words)

    # analogy comparison
    compare_analogies(gensim_analysis, scratch)

    # similarity spread check
    compute_similarity_spread(scratch)

    # save structured comparison
    save_comparison_json(gensim_analysis, gensim_train, scratch, probe_words)

    # summary
    print_header("Summary")
    print("""
  1. Gensim is ~50-70x faster thanks to its C extensions.
  2. Scratch CBOW shows high cosine saturation (>0.99) -- embeddings
     haven't differentiated enough. More epochs or tuning would help.
  3. Scratch Skip-gram (especially SG_C) gives reasonable results:
     phd -> fellowship, research -> areas, postgraduate -> master.
  4. Both approaches use the same vocab (16,990 words) and corpus.
  5. The quality gap is expected -- gensim has years of optimization
     (adaptive learning rate, efficient negative sampling tables, etc.)
     while our scratch version is a clean educational implementation.
""")


if __name__ == "__main__":
    main()
