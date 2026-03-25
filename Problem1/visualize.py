import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # important if running on server / no display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from wordcloud import WordCloud


# basic paths (keeping everything relative makes life easier)
MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"
CORPUS_PATH = os.path.join("corpus", "cleaned_corpus.txt")

os.makedirs(OUTPUTS_DIR, exist_ok=True)


# grouping words manually so plots are easier to read
# otherwise everything becomes a random cloud
WORD_GROUPS = {
    "Academic Programmes": [
        "btech", "mtech", "phd", "msc", "mba", "ug", "pg", "bachelor",
        "master", "doctoral", "undergraduate", "postgraduate",
    ],
    "Research & Faculty": [
        "research", "professor", "faculty", "laboratory", "publication",
        "journal", "conference", "thesis", "dissertation", "project",
    ],
    "Student Life": [
        "student", "hostel", "campus", "club", "festival", "sport",
        "library", "placement", "internship", "scholarship",
    ],
    "Academic Process": [
        "exam", "marks", "grade", "course", "semester", "credit",
        "attendance", "assignment", "lecture", "tutorial",
    ],
    "Departments": [
        "computer", "electrical", "mechanical", "chemical", "civil",
        "mathematics", "physics", "chemistry", "humanities",
    ],
}

# fixed colors → helps compare plots consistently
PALETTE = ["#E63946", "#2A9D8F", "#E9C46A", "#457B9D", "#F4A261"]


def load_model(name):
    # small helper, nothing fancy
    path = os.path.join(MODELS_DIR, f"{name}.model")

    if not os.path.exists(path):
        print(f"Model not found: {path}")
        return None

    # gensim handles loading internally
    return Word2Vec.load(path)


def get_word_vectors(model, groups):
    # collect vectors for only the words we care about
    # also keep group + color info so plotting is easier later

    vecs, words, labels, colors = [], [], [], []

    for group_name, color in zip(groups.keys(), PALETTE):
        for word in groups[group_name]:
            if word in model.wv:
                # normal case → word exists
                vecs.append(model.wv[word])
                words.append(word)
                labels.append(group_name)
                colors.append(color)
            else:
                # happens sometimes if word was filtered out during training
                pass

    return np.array(vecs), words, labels, colors


def scatter_plot(ax, coords, words, colors, labels, title):
    # basic scatter + annotations

    ax.scatter(
        coords[:, 0], coords[:, 1],
        c=colors,
        s=60,
        alpha=0.8,
        edgecolors="white",  # helps visibility
        linewidths=0.5
    )

    # label each point (can get messy if too many words)
    for i, (x, y) in enumerate(coords):
        ax.annotate(
            words[i],
            (x, y),
            fontsize=7.5,
            alpha=0.9,
            xytext=(3, 3),
            textcoords="offset points"
        )

    ax.set_title(title, fontsize=11, fontweight="bold")

    # hide axes → cleaner look
    ax.set_xticks([])
    ax.set_yticks([])

    # build legend manually (since colors are grouped)
    seen = {}
    for lbl, clr in zip(labels, colors):
        if lbl not in seen:
            seen[lbl] = clr

    patches = [mpatches.Patch(color=c, label=l) for l, c in seen.items()]
    ax.legend(handles=patches, fontsize=6.5, loc="lower right")


def plot_pca(cbow_model, sg_model, save_path):
    # PCA is linear → quick sanity check for structure

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    for ax, model, name in zip(
        axes,
        [cbow_model, sg_model],
        ["CBOW", "Skip-gram"]
    ):
        vecs, words, labels, colors = get_word_vectors(model, WORD_GROUPS)

        # PCA needs at least a few points
        if len(vecs) < 4:
            ax.text(0.5, 0.5, "Too few words", ha="center")
            continue

        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(vecs)

        # just showing how much variance is captured
        expl = pca.explained_variance_ratio_

        title = f"{name} | var: {expl[0]:.2f}, {expl[1]:.2f}"

        scatter_plot(ax, coords, words, colors, labels, title)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"PCA saved → {save_path}")


def plot_tsne(cbow_model, sg_model, save_path):
    # t-SNE → non-linear, usually better clusters
    # but less interpretable than PCA

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    for ax, model, name in zip(
        axes,
        [cbow_model, sg_model],
        ["CBOW", "Skip-gram"]
    ):
        vecs, words, labels, colors = get_word_vectors(model, WORD_GROUPS)

        if len(vecs) < 4:
            ax.text(0.5, 0.5, "Too few words", ha="center")
            continue

        # important: perplexity < num_samples
        perplexity = min(10, len(vecs) - 1)

        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            max_iter=2000,
            random_state=42,
            init="pca"
        )

        coords = tsne.fit_transform(vecs)

        scatter_plot(ax, coords, words, colors, labels, f"{name} (t-SNE)")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"t-SNE saved → {save_path}")


def plot_training_comparison(results_path, save_path):
    # simple comparison: vocab size + training time

    if not os.path.exists(results_path):
        print("results not found, skipping")
        return

    with open(results_path) as f:
        results = json.load(f)

    names = [r["name"] for r in results]
    vocabs = [r["vocab_size"] for r in results]
    times = [r["training_time_sec"] for r in results]

    x = np.arange(len(names))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # vocab plot
    axes[0].bar(x, vocabs)
    axes[0].set_title("Vocab Size")

    # time plot
    axes[1].bar(x, times)
    axes[1].set_title("Training Time")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"comparison saved → {save_path}")


def generate_wordcloud_from_corpus(corpus_path, save_path):
    # quick fallback word cloud (not super optimized)

    if not os.path.exists(corpus_path):
        return

    words = []

    with open(corpus_path) as f:
        for line in f:
            words.extend(line.strip().split())

    # simple frequency count
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1

    wc = WordCloud(width=1200, height=700).generate_from_frequencies(freq)

    plt.imshow(wc)
    plt.axis("off")
    plt.savefig(save_path)
    plt.close()

    print(f"wordcloud saved → {save_path}")


def main():
    print("Making plots...")

    # load models (assuming already trained)
    cbow_model = load_model("CBOW_B")
    sg_model = load_model("SG_B")

    if cbow_model is None or sg_model is None:
        print("Models missing, train first")
        return

    # run all visualizations
    plot_pca(cbow_model, sg_model,
             os.path.join(OUTPUTS_DIR, "pca.png"))

    plot_tsne(cbow_model, sg_model,
              os.path.join(OUTPUTS_DIR, "tsne.png"))

    plot_training_comparison(
        os.path.join(OUTPUTS_DIR, "training_results.json"),
        os.path.join(OUTPUTS_DIR, "comparison.png")
    )

    # generate word cloud if needed
    wc_path = os.path.join(OUTPUTS_DIR, "wordcloud.png")
    if not os.path.exists(wc_path):
        generate_wordcloud_from_corpus(CORPUS_PATH, wc_path)

    print("done")


if __name__ == "__main__":
    main()