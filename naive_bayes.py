"""
Naive Bayes SMS Spam Classifier (pure Python, no ML libraries)
==============================================================

This program trains a Naive Bayes classifier that labels SMS messages as
"spam" (unwanted advertising/scams) or "ham" (legitimate messages).

Dataset
-------
SMSSpamCollection: 5,574 messages, one per line, in the form

    label<TAB>message text

where label is "ham" or "spam" (about 13.4% of messages are spam).

The idea behind Naive Bayes
---------------------------
We want the class C (spam or ham) that is most probable given the words X
in a message. Bayes' rule gives

    P(C | X) = P(X | C) * P(C) / P(X)

    - P(C)      the prior: how common each class is before reading the message
    - P(X | C)  the likelihood: how likely these words are in that class
    - P(X)      the evidence: the same for both classes, so it does not change
                which class wins (we bring it back at the end to get a real
                probability)

P(X | C) is hard to estimate for a whole message, so we make the "naive"
assumption that words are conditionally independent given the class:

    P(X | C) ~= P(x1 | C) * P(x2 | C) * ... * P(xn | C)

Each P(word | class) is then just a count we can read off the training data.

How to run
----------
    python naive_bayes.py

Only the Python standard library is used (re, math, random, collections).
"""

import math
import random
import re
import sys
from collections import Counter

# Some messages contain characters (e.g. the pound sign) that a Windows
# console may not be able to display; print a replacement instead of crashing.
sys.stdout.reconfigure(errors="replace")

DATA_PATH = "SMSSpamCollection"
CLASSES = ("spam", "ham")


# =============================================================================
# STEP 1: LOAD THE DATA
# =============================================================================
# Each line of the file is "label<TAB>text". We split on the first tab only,
# because the message text itself could in principle contain a tab.

def load_data(path):
    """
    Read the SMS dataset from disk.

    Args:
        path: path to the SMSSpamCollection file.

    Returns:
        A list of (label, text) tuples, e.g. [("ham", "Ok lar..."), ...].
    """
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            label, text = line.split("\t", 1)
            data.append((label, text))
    return data


# =============================================================================
# STEP 2: TOKENIZE
# =============================================================================
# The classifier works on words, not raw text, so each message is turned into
# a list of lowercase words. The SAME function is used for training and for
# testing; if the two tokenized differently, the words would not match up.

def tokenize(text):
    """
    Split a message into a list of lowercase words.

    Lowercasing makes "FREE", "Free" and "free" the same word. Punctuation is
    dropped. Digits are kept on purpose: phone numbers and short codes such
    as "87121" are strong spam signals.

    Example:
        tokenize("WINNER!! Call now") -> ['winner', 'call', 'now']

    Args:
        text: the raw message string.

    Returns:
        A list of word strings.
    """
    # Runs of letters, digits and apostrophes ("don't" stays one word).
    return re.findall(r"[a-z0-9']+", text.lower())


# =============================================================================
# STEP 3: TRAIN / TEST SPLIT
# =============================================================================
# We hold out 20% of the messages for testing. The model never sees their
# words during training, so the test accuracy measures how well it handles
# messages it has never seen. Counting test words during training would be
# like grading a student on the exact questions they studied the answers to.
# A fixed seed makes the shuffle (and so the results) the same every run.

def split_data(data, ratio=0.8, seed=42):
    """
    Shuffle the data and split it into a training set and a test set.

    Args:
        data:  list of (label, text) tuples.
        ratio: fraction of messages that go into the training set.
        seed:  random seed so the split is reproducible.

    Returns:
        (train_data, test_data), two lists of (label, text) tuples.
    """
    shuffled = list(data)  # copy so the caller's list is not reordered
    random.Random(seed).shuffle(shuffled)
    cut = int(len(shuffled) * ratio)
    return shuffled[:cut], shuffled[cut:]


# =============================================================================
# STEP 4 + STEP 5: TRAINING (PRIORS P(C) AND WORD COUNTS)
# =============================================================================
# Training a Naive Bayes model is nothing more than counting.
#
# Step 4, priors: the fraction of training messages in each class.
#     P(spam) = (# spam messages) / (# training messages)
#     P(ham)  = (# ham messages)  / (# training messages)
#
# Step 5, word counts: for each class, how many times every word appears
# across all messages of that class, plus the total number of words in the
# class and the vocabulary (every distinct word seen in training). Every
# occurrence counts, so "free free free" adds 3 to count("free", spam); this
# is the multinomial version of Naive Bayes.

def train(train_data):
    """
    Learn the priors and per-class word counts from the training data.

    Formulas:
        P(c)       = (# messages in class c) / (# training messages)
        counts[c]  = Counter of word -> number of occurrences in class c
        totals[c]  = total number of words in class c
        vocab      = set of all distinct words in the training data

    Args:
        train_data: list of (label, text) tuples (training set ONLY).

    Returns:
        A model dict with keys "priors", "counts", "totals" and "vocab".
    """
    # Step 4: priors.
    num_messages = len(train_data)
    class_message_counts = Counter(label for label, _ in train_data)
    priors = {c: class_message_counts[c] / num_messages for c in CLASSES}

    # Step 5: word counts per class.
    counts = {c: Counter() for c in CLASSES}
    for label, text in train_data:
        counts[label].update(tokenize(text))

    totals = {c: sum(counts[c].values()) for c in CLASSES}
    vocab = set(counts["spam"]) | set(counts["ham"])

    return {"priors": priors, "counts": counts, "totals": totals, "vocab": vocab}


# =============================================================================
# STEP 6: LIKELIHOODS P(word | class) WITH LAPLACE SMOOTHING
# =============================================================================
# The plain maximum-likelihood estimate is
#     P(w | c) = count(w, c) / totals[c]
# but a word that never appeared in class c (e.g. a word only ever seen in
# ham) would get probability 0, and a single 0 wipes out the whole product
# P(x1|c) * P(x2|c) * ... no matter how strong the other evidence is.
#
# Laplace (add-one) smoothing pretends every vocabulary word was seen once
# more than it actually was:
#     P(w | c) = (count(w, c) + 1) / (totals[c] + |V|)
# The +1 on top means no probability is ever 0. The +|V| on the bottom keeps
# the probabilities summing to 1, because we added 1 for each of the |V|
# words in the vocabulary.

def word_prob(model, word, c):
    """
    Smoothed probability of seeing `word` in a message of class `c`.

    Formula:
        P(w | c) = (count(w, c) + 1) / (totals[c] + |V|)

    Args:
        model: the dict returned by train().
        word:  a single token.
        c:     "spam" or "ham".

    Returns:
        A float strictly between 0 and 1.
    """
    count = model["counts"][c][word]  # Counter returns 0 for unseen words
    return (count + 1) / (model["totals"][c] + len(model["vocab"]))


# =============================================================================
# STEP 7: CLASSIFY (LOG SCORES)
# STEP 8: TURN SCORES INTO A PROBABILITY (LOG-SUM-EXP)
# =============================================================================
# Step 7. The Naive Bayes rule says pick the class c that maximizes
#     P(c) * P(x1 | c) * P(x2 | c) * ... * P(xn | c)
# Multiplying many small numbers underflows to 0.0 in floating point (try
# 0.001 ** 200 in Python). Since log is increasing, maximizing the log gives
# the same winner, and the log turns the product into a sum:
#     score(c) = log P(c) + sum over words w of log P(w | c)
#
# Step 8. The scores are logs of the numerators of Bayes' rule; we skipped
# the denominator P(X). To get an actual probability back we normalize,
# because P(X) is just the sum of the numerators over both classes:
#     P(spam | X) = e^score_spam / (e^score_spam + e^score_ham)
# The scores are large negative numbers (often below -300), so e^score would
# underflow to 0 and we would divide 0 by 0. Subtracting the larger score
# from both first leaves the ratio unchanged (it multiplies top and bottom by
# the same constant) but keeps the numbers in a safe range. This is called
# the log-sum-exp trick.

def classify(model, text):
    """
    Predict whether a message is spam or ham.

    Formulas:
        score(c)    = log P(c) + sum_{w in message, w in vocab} log P(w | c)
        P(spam | X) = e^(s_spam - m) / (e^(s_spam - m) + e^(s_ham - m)),
                      where m = max(s_spam, s_ham)

    Args:
        model: the dict returned by train().
        text:  the raw message string.

    Returns:
        (label, p_spam): the predicted class and P(spam | message).
    """
    words = tokenize(text)

    # Step 7: log score for each class.
    scores = {}
    for c in CLASSES:
        score = math.log(model["priors"][c])
        for w in words:
            # Words never seen in training give no evidence for either
            # class, so they are skipped.
            if w in model["vocab"]:
                score += math.log(word_prob(model, w, c))
        scores[c] = score

    # Step 8: normalize the scores into P(spam | message).
    m = max(scores.values())  # subtract the max so exp() cannot underflow
    exp_spam = math.exp(scores["spam"] - m)
    exp_ham = math.exp(scores["ham"] - m)
    p_spam = exp_spam / (exp_spam + exp_ham)

    label = "spam" if scores["spam"] > scores["ham"] else "ham"
    return label, p_spam


# =============================================================================
# STEP 9: EVALUATE ON THE TEST SET
# =============================================================================
# Accuracy alone can be misleading here: since only ~13% of messages are
# spam, a "model" that always answers ham is already ~87% accurate. So we
# also report that baseline and a confusion matrix showing how many spam
# messages were caught or missed and how many ham messages were wrongly
# flagged.

def evaluate(model, test_data):
    """
    Measure the classifier's performance on held-out data.

    Formulas:
        accuracy = (# correct predictions) / (# test messages)
        baseline = (# ham messages) / (# test messages)   (always say "ham")

    Args:
        model:     the dict returned by train().
        test_data: list of (label, text) tuples the model was not trained on.

    Returns:
        A dict with "accuracy", "baseline" and the confusion matrix counts
        "tp" (spam caught), "fn" (spam missed), "fp" (ham flagged as spam)
        and "tn" (ham correctly passed).
    """
    tp = fn = fp = tn = 0
    for actual, text in test_data:
        predicted, _ = classify(model, text)
        if actual == "spam" and predicted == "spam":
            tp += 1
        elif actual == "spam":
            fn += 1
        elif predicted == "spam":
            fp += 1
        else:
            tn += 1

    total = len(test_data)
    return {
        "accuracy": (tp + tn) / total,
        "baseline": (tn + fp) / total,  # fraction of test messages that are ham
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
    }


# =============================================================================
# MAIN: RUN EVERY STEP AND PRINT A WALKTHROUGH
# =============================================================================

def shorten(text, width=60):
    """Trim a message to `width` characters for display."""
    return text if len(text) <= width else text[: width - 3] + "..."


def main():
    # --- Step 1: load ---
    print("--- Step 1: Load the data ---")
    data = load_data(DATA_PATH)
    num_spam = sum(1 for label, _ in data if label == "spam")
    print(f"Loaded {len(data)} messages: {num_spam} spam "
          f"({num_spam / len(data):.1%}), {len(data) - num_spam} ham")

    # --- Step 2: tokenize (show an example) ---
    print("\n--- Step 2: Tokenize ---")
    example = "WINNER!! Call now to claim your FREE prize"
    print(f'"{example}"\n  -> {tokenize(example)}')

    # --- Step 3: split ---
    print("\n--- Step 3: Train/test split ---")
    train_data, test_data = split_data(data)
    print(f"Training messages: {len(train_data)}   Test messages: {len(test_data)}")

    # --- Steps 4 and 5: train ---
    model = train(train_data)
    print("\n--- Step 4: Priors P(C) ---")
    for c in CLASSES:
        print(f"P({c}) = {model['priors'][c]:.4f}")

    print("\n--- Step 5: Word counts ---")
    print(f"Vocabulary size |V| = {len(model['vocab'])}")
    for c in CLASSES:
        top = ", ".join(w for w, _ in model["counts"][c].most_common(10))
        print(f"{c}: {model['totals'][c]} words total; most common: {top}")

    # --- Step 6: likelihoods for a few selected words ---
    print("\n--- Step 6: Likelihoods P(word | class) for selected words ---")
    # Sanity check: smoothed probabilities over the vocabulary sum to 1.
    for c in CLASSES:
        total_prob = sum(word_prob(model, w, c) for w in model["vocab"])
        print(f"Sum of P(w | {c}) over vocab = {total_prob:.6f}")
    print()
    selected = ["free", "win", "call", "txt", "prize", "claim",
                "urgent", "ok", "love", "home"]
    print(f"{'word':<8} {'P(w|spam)':>11} {'P(w|ham)':>11} {'spam/ham':>9}")
    for w in selected:
        ps, ph = word_prob(model, w, "spam"), word_prob(model, w, "ham")
        print(f"{w:<8} {ps:>11.6f} {ph:>11.6f} {ps / ph:>9.2f}")

    # --- Steps 7 and 8: classify some test messages ---
    print("\n--- Steps 7 & 8: Classify sample test messages ---")
    for actual, text in test_data[:10]:
        predicted, p_spam = classify(model, text)
        mark = "" if predicted == actual else "   <-- wrong"
        print(f'"{shorten(text)}"\n  -> {predicted.upper()} '
              f"(P(spam) = {p_spam:.3f}) [actual: {actual}]{mark}")

    # --- Step 9: evaluate ---
    print("\n--- Step 9: Evaluate on the test set ---")
    results = evaluate(model, test_data)
    print(f"Accuracy:                 {results['accuracy']:.2%}")
    print(f"Always-ham baseline:      {results['baseline']:.2%}")
    print("Confusion matrix:")
    print(f"                     predicted spam   predicted ham")
    print(f"  actual spam        {results['tp']:>14}   {results['fn']:>13}")
    print(f"  actual ham         {results['fp']:>14}   {results['tn']:>13}")
    print(f"Spam caught: {results['tp']}, spam missed: {results['fn']}, "
          f"ham wrongly flagged: {results['fp']}")

    # --- Custom messages ---
    print("\n--- Custom messages ---")
    custom = [
        "WINNER! You have been selected to receive a $1000 prize. Call 09061701461 now to claim",
        "URGENT! Your mobile number has won a free holiday. Txt WIN to 80086",
        "Hey, are we still meeting for lunch tomorrow?",
        "Ok I'll be home in 10 mins, love you",
        "Call me when you're free tonight",
        "Congrats on the new job! We should celebrate, drinks on me",
        "Thank you for applying to P&G. We’re excited you’re here! If you love P&G brands enough to want to work with us, why not get even closer to them? Join a P&G Brand Fan Club to discover special experiences, take on creative challenges and share what makes your favorite brands worth talking about.", 
        
    ]
    for text in custom:
        predicted, p_spam = classify(model, text)
        print(f'"{shorten(text)}"\n  -> {predicted.upper()} (P(spam) = {p_spam:.3f})')


if __name__ == "__main__":
    main()
