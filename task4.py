"""
Task 4: NER using MLP over token embeddings

Supports:
✔ .pt embeddings (PyTorch)
✔ .npy embeddings
✔ Task-1 vocabulary
✔ OOV handling
✔ Accuracy + Macro-F1 reporting

Run:

python task_4_mlp_ner.py --emb glove.pt
python task_4_mlp_ner.py --emb svd.npy
"""

import numpy as np
import torch
import json
import argparse
from datasets import load_dataset
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from tqdm import tqdm


# ----------------------------------------------------
# LOAD VOCABULARY (Task 1)
# ----------------------------------------------------
with open("vocab_word2idx.json") as f:
    WORD2IDX = json.load(f)

VOCAB_SIZE = len(WORD2IDX)


# ----------------------------------------------------
# LOAD EMBEDDINGS (.pt or .npy)
# ----------------------------------------------------
def load_embeddings(path):

    if path.endswith(".npy"):
        emb = np.load(path)
        print("Loaded .npy embeddings:", emb.shape)

    elif path.endswith(".pt"):
        data = torch.load(path, map_location="cpu")

        if isinstance(data, torch.Tensor):
            emb = data.numpy()

        elif isinstance(data, dict):
            for key in ["embeddings", "weight", "vectors", "matrix"]:
                if key in data:
                    emb = data[key].cpu().numpy()
                    break
            else:
                raise ValueError("Could not find embeddings in .pt file")

        else:
            raise ValueError("Unsupported .pt format")

        print("Loaded .pt embeddings:", emb.shape)

    else:
        raise ValueError("Unsupported embedding format")

    # sanity check
    if emb.shape[0] != VOCAB_SIZE:
        raise ValueError(
            f"Embedding vocab size {emb.shape[0]} "
            f"!= vocab size {VOCAB_SIZE}"
        )

    return emb


# ----------------------------------------------------
# NER TAGS (CoNLL-2003)
# ----------------------------------------------------
NER_TAGS = [
    "O",
    "B-PER", "I-PER",
    "B-ORG", "I-ORG",
    "B-LOC", "I-LOC",
    "B-MISC", "I-MISC"
]


# ----------------------------------------------------
# TOKEN → EMBEDDING
# ----------------------------------------------------
def token_to_vec(token, embeddings):
    if token in WORD2IDX:
        return embeddings[WORD2IDX[token]]

    # OOV strategy: zero vector
    return np.zeros(embeddings.shape[1])


# ----------------------------------------------------
# BUILD DATASET
# ----------------------------------------------------
def build_dataset(split, embeddings):

    X = []
    y = []

    for tokens, tags in tqdm(zip(split["tokens"], split["ner_tags"]),
                             total=len(split["tokens"])):

        for token, tag in zip(tokens, tags):
            X.append(token_to_vec(token, embeddings))
            y.append(tag)

    return np.array(X), np.array(y)


# ----------------------------------------------------
# MAIN
# ----------------------------------------------------
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--emb", required=True,
                        help="Path to embedding file (.pt or .npy)")
    args = parser.parse_args()

    # load embeddings
    embeddings = load_embeddings(args.emb)

    # load dataset
    print("\nLoading CoNLL-2003 dataset...")
    dataset = load_dataset("conll2003")

    # build datasets
    print("\nBuilding training data...")
    X_train, y_train = build_dataset(dataset["train"], embeddings)

    print("\nBuilding test data...")
    X_test, y_test = build_dataset(dataset["test"], embeddings)

    print("\nTraining MLP classifier...")

    clf = MLPClassifier(
        hidden_layer_sizes=(256,256),
        activation="relu",
        solver="adam",
        batch_size=256,
        learning_rate_init=0.001,
        max_iter=40,
        verbose=True,
        random_state=42
    )

    clf.fit(X_train, y_train)

    print("\nEvaluating...")

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print("\nAccuracy:", acc)
    print("Macro F1:", macro_f1)

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=NER_TAGS))


if __name__ == "__main__":
    main()
