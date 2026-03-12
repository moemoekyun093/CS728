import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier

from task2 import (
    build_term_document_matrix,
    compute_svd_embeddings,
    load_vocab_mappings,
    load_vocab_document_dict,
    nearest_neighbors,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def save_json(obj: object, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def choose_words(vocab: Dict[str, int], k: int = 5) -> List[str]:
    words = list(vocab.keys())
    cap = [w for w in words if len(w) > 2 and w[:1].isupper() and w[1:].islower()]
    verb = [w for w in words if w.endswith("ing") or w.endswith("ed")]
    lower = [w for w in words if w.islower() and w.isalpha() and len(w) >= 4]

    selected = []
    for bucket in [cap, verb, lower]:
        for w in bucket:
            if w not in selected:
                selected.append(w)
                if len(selected) >= k:
                    return selected

    for w in words:
        if w not in selected:
            selected.append(w)
            if len(selected) >= k:
                break
    return selected


def token_to_vec(
    token: str,
    embeddings: np.ndarray,
    vocab: Dict[str, int],
    unk_strategy: str = "zero",
) -> np.ndarray:
    if token in vocab:
        return embeddings[vocab[token]]
    if unk_strategy == "avg":
        return embeddings.mean(axis=0)
    return np.zeros(embeddings.shape[1], dtype=np.float32)


def load_ner_split_features(split, embeddings: np.ndarray, vocab: Dict[str, int], unk_strategy: str):
    xs, ys = [], []
    for tokens, tags in zip(split["tokens"], split["ner_tags"]):
        for tok, tag in zip(tokens, tags):
            xs.append(token_to_vec(tok, embeddings, vocab, unk_strategy))
            ys.append(int(tag))
    x = np.asarray(xs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.int64)
    return x, y


def train_eval_mlp(
    embeddings: np.ndarray,
    vocab: Dict[str, int],
    hidden_layers: Tuple[int, ...],
    max_iter: int,
    batch_size: int,
    lr: float,
    unk_strategy: str,
    random_state: int,
    trust_remote_code: bool,
) -> Dict[str, float]:
    from datasets import load_dataset

    ds = load_dataset("conll2003", trust_remote_code=trust_remote_code)

    x_train, y_train = load_ner_split_features(ds["train"], embeddings, vocab, unk_strategy)
    x_test, y_test = load_ner_split_features(ds["test"], embeddings, vocab, unk_strategy)

    clf = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        batch_size=batch_size,
        learning_rate_init=lr,
        max_iter=max_iter,
        verbose=False,
        random_state=random_state,
    )
    clf.fit(x_train, y_train)
    preds = clf.predict(x_test)

    return {
        "token_accuracy": float(accuracy_score(y_test, preds)),
        "macro_f1": float(f1_score(y_test, preds, average="macro")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 5: TF-IDF + SVD boost")
    parser.add_argument("--input", type=str, required=True, help="Path to vocab->[(doc_id, passage)] json")
    parser.add_argument(
        "--vocab_word2idx",
        type=str,
        default="vocab_word2idx.json",
        help="Existing vocab word->idx JSON path (default: ./vocab_word2idx.json)",
    )
    parser.add_argument(
        "--vocab_idx2word",
        type=str,
        default="vocab_idx2word.json",
        help="Existing vocab idx->word JSON path (default: ./vocab_idx2word.json)",
    )
    parser.add_argument("--best_dim", type=int, required=True, help="Best SVD dimension from Task 4")
    parser.add_argument("--words", type=str, default="", help="Comma-separated words for neighbor comparison")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--out_dir", type=str, default=".")
    parser.add_argument("--run_ner", action="store_true", help="Train final MLP with TF-IDF SVD vectors")
    parser.add_argument("--train_raw_baseline", action="store_true", help="Also train raw SVD baseline MLP")
    parser.add_argument("--raw_task4_metrics", type=str, default="", help="Optional JSON with raw SVD metrics")
    parser.add_argument("--unk_strategy", type=str, choices=["zero", "avg"], default="zero")
    parser.add_argument(
        "--hidden_layers",
        type=str,
        default="256,256",
        help="Comma-separated hidden layer sizes (Task4 default: 256,256)",
    )
    parser.add_argument("--max_iter", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code=True when loading conll2003 via datasets",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    hidden_layers = tuple(int(x.strip()) for x in args.hidden_layers.split(",") if x.strip())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    data = load_vocab_document_dict(args.input)
    vocab, idx2word = load_vocab_mappings(args.vocab_word2idx, args.vocab_idx2word)
    matrix, _ = build_term_document_matrix(data, vocab)

    raw_emb = compute_svd_embeddings(matrix, args.best_dim, random_state=args.seed)
    tfidf = TfidfTransformer(norm=None, use_idf=True, smooth_idf=True, sublinear_tf=False)
    tfidf_matrix = tfidf.fit_transform(matrix.T).T
    tfidf_emb = compute_svd_embeddings(tfidf_matrix, args.best_dim, random_state=args.seed)

    np.save(out_dir / f"svd_raw_embeddings_d{args.best_dim}.npy", raw_emb)
    np.save(out_dir / f"svd_tfidf_embeddings_d{args.best_dim}.npy", tfidf_emb)

    words = [w.strip() for w in args.words.split(",") if w.strip()]
    if not words:
        words = choose_words(vocab, k=5)

    neighbors_report: Dict[str, Dict[str, List[Tuple[str, float]]]] = {}
    for w in words:
        neighbors_report[w] = {
            "raw_svd": nearest_neighbors(w, raw_emb, vocab, idx2word, top_k=args.top_k),
            "tfidf_svd": nearest_neighbors(w, tfidf_emb, vocab, idx2word, top_k=args.top_k),
        }
    save_json(neighbors_report, out_dir / "task5_neighbors_comparison.json")
    save_json(neighbors_report, out_dir / f"task5_neighbors_comparison_d{args.best_dim}.json")

    if args.run_ner:
        tfidf_metrics = train_eval_mlp(
            embeddings=tfidf_emb,
            vocab=vocab,
            hidden_layers=hidden_layers,
            max_iter=args.max_iter,
            batch_size=args.batch_size,
            lr=args.lr,
            unk_strategy=args.unk_strategy,
            random_state=args.seed,
            trust_remote_code=args.trust_remote_code,
        )
        result = {"tfidf_svd": tfidf_metrics}

        if args.raw_task4_metrics:
            with open(args.raw_task4_metrics, "r") as f:
                result["raw_svd"] = json.load(f)
        elif args.train_raw_baseline:
            raw_metrics = train_eval_mlp(
                embeddings=raw_emb,
                vocab=vocab,
                hidden_layers=hidden_layers,
                max_iter=args.max_iter,
                batch_size=args.batch_size,
                lr=args.lr,
                unk_strategy=args.unk_strategy,
                random_state=args.seed,
                trust_remote_code=args.trust_remote_code,
            )
            result["raw_svd"] = raw_metrics

        if "raw_svd" in result:
            result["improvement_macro_f1"] = result["tfidf_svd"]["macro_f1"] - result["raw_svd"]["macro_f1"]
            result["improvement_accuracy"] = (
                result["tfidf_svd"]["token_accuracy"] - result["raw_svd"]["token_accuracy"]
            )

        save_json(result, out_dir / f"task5_ner_metrics_d{args.best_dim}.json")

    print(f"Saved Task 5 outputs in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
