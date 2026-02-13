import json
import argparse
from collections import defaultdict
from tqdm import tqdm


# ----------------------------------------------------
# 1. BUILD VOCAB INDEX
# ----------------------------------------------------
def build_vocab_index(data):
    """
    data: dict[word] -> list of [passage_id, passage]

    Returns:
        vocab: word -> index
        idx2word: index -> word
    """
    vocab = {}
    idx2word = {}

    for idx, word in enumerate(data.keys()):
        vocab[word] = idx
        idx2word[idx] = word

    return vocab, idx2word


# ----------------------------------------------------
# 2. BUILD COOCCURRENCE MATRIX
# ----------------------------------------------------
def build_cooccurrence(data, vocab, window_size=5):

    cooc = defaultdict(float)

    print("Building co-occurrence matrix...")

    for word in tqdm(data.keys()):

        if word not in vocab:
            continue

        w_i = vocab[word]
        passages = data[word]

        for passage_id, passage in passages:

            tokens = passage.split()

            positions = [pos for pos, tok in enumerate(tokens) if tok == word]

            for pos in positions:

                start = max(0, pos - window_size)
                end = min(len(tokens), pos + window_size + 1)

                for j in range(start, end):
                    if j == pos:
                        continue

                    context_word = tokens[j]

                    if context_word not in vocab:
                        continue

                    w_j = vocab[context_word]

                    # Pure co-occurrence count
                    cooc[(w_i, w_j)] += 1.0

    return cooc


# ----------------------------------------------------
# 3. SAVE FILES
# ----------------------------------------------------
def dump_cooccurrence(cooc, output_path):

    print("Saving co-occurrence matrix...")

    triples = [
        [int(i), int(j), float(v)]
        for (i, j), v in cooc.items()
    ]

    with open(output_path, "w") as f:
        json.dump(triples, f)

    print(f"Saved {len(triples)} non-zero entries.")


def dump_vocab(vocab, idx2word):
    print("Saving vocab mappings...")

    with open("vocab_word2idx.json", "w") as f:
        json.dump(vocab, f)

    with open("vocab_idx2word.json", "w") as f:
        json.dump(idx2word, f)

    print("Saved vocab_word2idx.json and vocab_idx2word.json")


# ----------------------------------------------------
# 4. MAIN
# ----------------------------------------------------
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="Path to inverted index JSON file")
    parser.add_argument("--window", type=int, default=5,
                        help="Context window size")
    parser.add_argument("--output", type=str,
                        default="cooc_matrix.json",
                        help="Output cooc file")

    args = parser.parse_args()

    print("Loading dataset...")
    with open(args.input, "r") as f:
        data = json.load(f)

    print("Building vocabulary index...")
    vocab, idx2word = build_vocab_index(data)

    print("Vocabulary size:", len(vocab))

    dump_vocab(vocab, idx2word)

    cooc = build_cooccurrence(
        data=data,
        vocab=vocab,
        window_size=args.window
    )

    print("Number of non-zero entries:", len(cooc))

    dump_cooccurrence(cooc, args.output)

    print("Done.")


if __name__ == "__main__":
    main()
