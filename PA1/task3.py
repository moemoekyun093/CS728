"""
Task 3: CRF NER (Final Version)

Includes:
✔ Vocabulary constraint (Task 1 vocab)
✔ Longest prefix OOV handling
✔ Lexical features
✔ Shape features (hasCap, startsWithCap, digits, etc.)
✔ Prefix & suffix features (length 1–3)
✔ Context features

Run:
    python task_3_crf_final.py
"""

import json
from datasets import load_dataset
import sklearn_crfsuite
from seqeval.metrics import classification_report, f1_score
from tqdm import tqdm
from collections import Counter, defaultdict
import re



SCORE_PATTERN = re.compile(r'\d+([:-]\d+)+')
# ----------------------------------------------------
# LOAD TASK-1 VOCABULARY
# ----------------------------------------------------
with open("vocab_word2idx.json") as f:
    WORD2IDX = json.load(f)

VOCAB_SET = set(WORD2IDX.keys())


# ----------------------------------------------------
# OOV HANDLING
# ----------------------------------------------------
def longest_prefix_match(token):
    for i in range(len(token), 0, -1):
        prefix = token[:i]
        if prefix in VOCAB_SET:
            return prefix
    return None


def longest_suffix_match(token):
    for i in range(len(token)):
        suffix = token[i:]
        if suffix in VOCAB_SET:
            return suffix
    return None


def map_token(token):
    if token in VOCAB_SET:
        return token

    prefix = longest_prefix_match(token)
    if prefix:
        return prefix

    suffix = longest_suffix_match(token)
    if suffix:
        return suffix

    return "<UNK>"


# ----------------------------------------------------
# NER TAG MAPPING
# ----------------------------------------------------
NER_TAGS = [
    "O",
    "B-PER", "I-PER",
    "B-ORG", "I-ORG",
    "B-LOC", "I-LOC",
    "B-MISC", "I-MISC"
]

SPEECH_VERBS = {
    "said", "told", "announced", "stated",
    "added", "claimed", "explained", "reported", "lost", "won", "defeated", "beat", "trounced",
}

LOC_PREP = {"in", "at", "from", "to", "near"}








# ----------------------------------------------------
# SHAPE FEATURES
# ----------------------------------------------------
def shape_features(token):

    return {
        # "hasCap": any(c.isupper() for c in token),
        # "startsWithCap": token[:1].isupper(),
        # "allCaps": token.isupper(),

        "hasDigit": any(c.isdigit() for c in token),
        "allDigits": token.isdigit(),
        "fourDigits": token.isdigit() and len(token) == 4,
    }


# ----------------------------------------------------
# FEATURE ENGINEERING
# ----------------------------------------------------
def word2features(sentence, i):

    token = sentence[i]
    mapped = map_token(token)

    features = {
        # lexical
        "token": token,
        "lower": token.lower(),
        # "vocab": mapped,
        "nextIsSpeechVerb": sentence[i+1].lower() in SPEECH_VERBS if i < len(sentence)-1 else False,
        "prevIsLocationPrep": sentence[i-1].lower() in LOC_PREP if i > 0 else False,

        # shape features
        **shape_features(token),

        # prefix (1–3)
        "pref1": token[:1],
        "pref2": token[:2],
        "pref3": token[:3],

        # suffix (1–3)
        "suf1": token[-1:],
        "suf2": token[-2:],
        "suf3": token[-3:],

    }

    # previous token
    if i > 0:
        prev = sentence[i-1]
        if len(prev) > 1 or prev.isalpha():
            features.update({
                "-1:vocab": map_token(prev),
                # "-1:startsWithCap": prev[:1].isupper(),
                "-1:isScore": bool(SCORE_PATTERN.match(prev))
            })
    # else:
    #     features["BOS"] = True

    # next token
    if i < len(sentence) - 1:
        nxt = sentence[i+1]
        if len(nxt) > 1 or nxt.isalpha():
            features.update({
                "+1:vocab": map_token(nxt),
                # "+1:startsWithCap": nxt[:1].isupper(),
                "+1:isScore": bool(SCORE_PATTERN.match(nxt))
            })
    else:
        features["EOS"] = True

    if i < len(sentence)-1:
        features["token_bigram"] = token + "_" + sentence[i+1]
    else:
        features["token_bigram"] = token + "_<EOS>"

    return features


def sent2features(sentence):
    return [word2features(sentence, i) for i in range(len(sentence))]


def sent2labels(tags):
    return [NER_TAGS[tag] for tag in tags]


# ----------------------------------------------------
# LOAD DATASET
# ----------------------------------------------------
def load_conll():
    return load_dataset("conll2003")


# ----------------------------------------------------
# PREPARE DATA
# ----------------------------------------------------
def dump_to_jsonl(X, y, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for features_seq, labels_seq in zip(X, y):
            # Combine features and labels for each sentence
            record = {
                "pair":[(x,y) for x,y in zip(features_seq, labels_seq)]
            }
            f.write(json.dumps(record) + '\n')


def prepare_data(dataset):

    X_train = [sent2features(s) for s in tqdm(dataset["train"]["tokens"])]
    y_train = [sent2labels(s) for s in dataset["train"]["ner_tags"]]

    X_test = [sent2features(s) for s in tqdm(dataset["test"]["tokens"])]
    y_test = [sent2labels(s) for s in dataset["test"]["ner_tags"]]

    dump_to_jsonl(dataset["train"]["tokens"], y_train, 'train_data.jsonl')

    return X_train, y_train, X_test, y_test

# def prepare_data(dataset):

#     tag_counter = Counter()

#     X_train = []
#     y_train = []

#     for tokens, tags in tqdm(zip(dataset["train"]["tokens"],
#                                  dataset["train"]["ner_tags"]),
#                              total=len(dataset["train"]["tokens"])):

#         labels = sent2labels(tags)

#         X_train.append(sent2features(tokens))
#         y_train.append(labels)

#         tag_counter.update(labels)

#     print("\nTraining tag distribution:\n")
#     total = sum(tag_counter.values())

#     for tag, count in tag_counter.most_common():
#         pct = 100 * count / total
#         print(f"{tag:7s} : {count:6d}  ({pct:5.2f}%)")

#     dump_to_jsonl(dataset["train"]["tokens"], y_train, 'train_data.jsonl')
    

#     return X_train, y_train, 


# ----------------------------------------------------
# TRAIN CRF
# ----------------------------------------------------
def train_crf(X_train, y_train):

    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )

    crf.fit(X_train, y_train)
    return crf


# ----------------------------------------------------
# EVALUATE
# ----------------------------------------------------
def evaluate(crf, dataset, X_test, y_test):

    print("\nEvaluating model...")

    y_pred = crf.predict(X_test)

    print("\nF1 Score:", f1_score(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # dump incorrect predictions
    dump_errors(
        sentences=dataset["test"]["tokens"],
        y_true=y_test,
        y_pred=y_pred,
        crf=crf,
        output_file="crf_errors.txt"
    )

    return y_pred



# ----------------------------------------------------
# FEATURE IMPORTANCE (for report)
# ----------------------------------------------------
def show_top_features(crf, top_k=10):
    print("\nTop features influencing predictions:\n")

    label_features = defaultdict(list)

    # state_features_: {(feature, label): weight}
    for (feature, label), weight in crf.state_features_.items():
        label_features[label].append((feature, weight))

    for label in crf.classes_:
        print(f"\nLabel: {label}")

        if label not in label_features:
            print("  No strong features learned.")
            continue

        # sort by absolute weight
        top_feats = sorted(
            label_features[label],
            key=lambda x: abs(x[1]),
            reverse=True
        )[:top_k]

        for feat, weight in top_feats:
            print(f"  {feat:30s} {weight:.3f}")

def explain_prediction(crf, features, predicted_label, top_k=15):
    """
    Returns top feature contributions for a predicted label.
    """

    contributions = []

    for feat_name, feat_val in features.items():

        if isinstance(feat_val, bool):
            key = f"{feat_name}"
        else:
            key = f"{feat_name}:{feat_val}"

        if (key, predicted_label) in crf.state_features_:
            weight = crf.state_features_[(key, predicted_label)]
            contributions.append((key+":"+str(feat_val), weight))

    contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    return contributions[:top_k]


def dump_errors(sentences, y_true, y_pred, crf,
                output_file="crf_errors.txt",
                max_sentences=100):

    print("\nCollecting errors...")

    wrong_count = 0

    with open(output_file, "w") as f:

        for tokens, true_tags, pred_tags in zip(sentences, y_true, y_pred):

            if true_tags == pred_tags:
                continue

            wrong_count += 1

            f.write("Sentence:\n")
            f.write(" ".join(tokens) + "\n\n")

            # mapped tokens
            f.write("Mapped tokens:\n")
            f.write(" ".join(map_token(tok) for tok in tokens) + "\n\n")

            f.write("TRUE:\n")
            f.write(" ".join(f"{w}/{t}" for w, t in zip(tokens, true_tags)) + "\n\n")

            f.write("PRED:\n")
            f.write(" ".join(f"{w}/{t}" for w, t in zip(tokens, pred_tags)) + "\n\n")

            # explain incorrect tokens
            f.write("Incorrect token analysis:\n")

            for i, (tok, t_true, t_pred) in enumerate(zip(tokens, true_tags, pred_tags)):
                if t_true == t_pred:
                    continue

                f.write(f"\nToken: {tok}\n")
                f.write(f"TRUE: {t_true}   PRED: {t_pred}\n")

                feats = word2features(tokens, i)
                contribs = explain_prediction(crf, feats, t_pred)

                if contribs:
                    f.write("Top feature influences:\n")
                    for name, weight in contribs:
                        f.write(f"   {name:30s} {weight:+.3f}\n")
                else:
                    f.write("   (no strong features found)\n")

            f.write("\n" + "-"*60 + "\n\n")

            if wrong_count >= max_sentences:
                break

    print(f"Saved {wrong_count} error sentences → {output_file}")


def print_feature_weight(crf, feature_name):
    """
    Prints weights of a specific feature for all labels.
    """

    key = f"{feature_name}"

    print(f"\nWeights for feature: {key}\n")

    found = False

    with open("features.txt", "w") as f:
        for (feature, label), weight in sorted(
            crf.state_features_.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        ):
            f.write(f"{feature}\t{label}\t{weight:.6f}\n")

    print("Done.")

    for (feat, label), weight in crf.state_features_.items():
        if feat == key:
            print(f"{label:7s} : {weight:.4f}")
            found = True

    if not found:
        print("Feature not found in model.")




# ----------------------------------------------------
# MAIN
# ----------------------------------------------------
def main():

    print("Loading dataset...")
    dataset = load_conll()

    print("Preparing features...")
    X_train, y_train, X_test, y_test = prepare_data(dataset)

    print("Training CRF...")
    crf = train_crf(X_train, y_train)

    print("Evaluating...")
    evaluate(crf, dataset, X_test, y_test)

    print_feature_weight(crf, "+1:score")


    show_top_features(crf)


if __name__ == "__main__":
    main()
