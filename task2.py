import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


def load_vocab_document_dict(path: str) -> Dict[str, List[List[object]]]:
    with open(path, "r") as f:
        data = json.load(f)
    return data


def load_vocab_mappings(word2idx_path: str, idx2word_path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    with open(word2idx_path, "r") as f:
        vocab = json.load(f)
    with open(idx2word_path, "r") as f:
        raw_idx2word = json.load(f)
    idx2word = {int(k): v for k, v in raw_idx2word.items()}
    return vocab, idx2word


def build_term_document_matrix(
    data: Dict[str, List[List[object]]],
    vocab: Dict[str, int],
) -> Tuple[csr_matrix, Dict[object, int]]:
    doc_ids = set()
    for entries in data.values():
        for doc_id, _ in entries:
            doc_ids.add(doc_id)

    try:
        sorted_doc_ids = sorted(doc_ids, key=lambda x: int(x))
    except Exception:
        sorted_doc_ids = sorted(doc_ids)

    doc2col = {doc_id: col for col, doc_id in enumerate(sorted_doc_ids)}

    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []

    for word, entries in data.items():
        if word not in vocab:
            continue
        row = vocab[word]
        for doc_id, passage in entries:
            count = 0
            for tok in str(passage).split():
                if tok == word:
                    count += 1
            if count > 0:
                rows.append(row)
                cols.append(doc2col[doc_id])
                vals.append(float(count))

    matrix = csr_matrix(
        (vals, (rows, cols)),
        shape=(len(vocab), len(doc2col)),
        dtype=np.float32,
    )
    return matrix, doc2col


def compute_svd_embeddings(matrix: csr_matrix, dim: int, random_state: int = 42) -> np.ndarray:
    svd = TruncatedSVD(n_components=dim, random_state=random_state)
    # term-document X = U S V^T, fit_transform(X) gives U S (token-wise vectors)
    emb = svd.fit_transform(matrix)
    return emb.astype(np.float32)


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, eps, None)


def nearest_neighbors(
    query_word: str,
    embeddings: np.ndarray,
    vocab: Dict[str, int],
    idx2word: Dict[int, str],
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    if query_word not in vocab:
        return []

    q_idx = vocab[query_word]
    norm_emb = l2_normalize(embeddings)
    sims = norm_emb @ norm_emb[q_idx]
    sims[q_idx] = -1.0
    k = min(top_k, len(sims) - 1)
    if k <= 0:
        return []
    top_idx = np.argpartition(-sims, range(k))[:k]
    top_idx = top_idx[np.argsort(-sims[top_idx])]
    return [(idx2word[int(i)], float(sims[int(i)])) for i in top_idx]


def parse_dims(text: str) -> List[int]:
    dims = []
    for p in text.split(","):
        p = p.strip()
        if p:
            dims.append(int(p))
    return dims


def save_json(obj: object, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 2: SVD pipeline")
    parser.add_argument("--input", type=str, required=True, help="Path to vocab->[(doc_id, passage)] json")
    parser.add_argument(
        "--vocab_word2idx",
        type=str,
        default="vocab_word2idx.json",
        help="Existing vocab word->idx JSON path",
    )
    parser.add_argument(
        "--vocab_idx2word",
        type=str,
        default="vocab_idx2word.json",
        help="Existing vocab idx->word JSON path",
    )
    parser.add_argument("--dims", type=str, default="50,100,200,300", help="Comma-separated dimensions")
    parser.add_argument("--out_dir", type=str, default=".", help="Output directory")
    parser.add_argument(
        "--neighbors_words",
        type=str,
        default="",
        help="Comma-separated words for nearest-neighbor report",
    )
    parser.add_argument("--top_k", type=int, default=5, help="Top-k neighbors")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_vocab_document_dict(args.input)
    vocab, idx2word = load_vocab_mappings(args.vocab_word2idx, args.vocab_idx2word)
    matrix, doc2col = build_term_document_matrix(data, vocab)

    save_json({str(k): int(v) for k, v in doc2col.items()}, out_dir / "svd_doc2col.json")

    dims = parse_dims(args.dims)
    report: Dict[str, object] = {
        "vocab_size": len(vocab),
        "num_docs": matrix.shape[1],
        "dims": dims,
    }

    query_words = [w.strip() for w in args.neighbors_words.split(",") if w.strip()]
    for d in dims:
        emb = compute_svd_embeddings(matrix, d)
        np.save(out_dir / f"svd_embeddings_d{d}.npy", emb)

        if query_words:
            dim_report = {}
            for w in query_words:
                dim_report[w] = nearest_neighbors(w, emb, vocab, idx2word, top_k=args.top_k)
            report[f"neighbors_d{d}"] = dim_report

    if query_words:
        save_json(report, out_dir / "task2_neighbors_report.json")

    print(f"Saved SVD outputs in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
