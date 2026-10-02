"""
src/intent_router.py
--------------------
Lightweight TF-IDF cosine similarity intent classifier.

Returns one of four intent labels and a confidence score:
  - IN_DOMAIN_ADMISSION   : Recognised admissions question
  - GREETING_OR_CHITCHAT  : Greetings, farewells, compliments, identity questions
  - OUT_OF_SCOPE          : Politics, coding, sports, system commands, etc.
  - VAGUE_AMBIGUOUS       : Single keyword / very short / uninterpretable input
"""

import math
import re
from collections import Counter


# ---------------------------------------------------------------------------
# Labelled seed corpus (keyword bags per intent class)
# ---------------------------------------------------------------------------

_SEED_CORPUS = {
    "IN_DOMAIN_ADMISSION": [
        "what is the minimum gpa for admission",
        "when is the application deadline",
        "how much is tuition fee",
        "do you offer scholarships financial aid",
        "is sat act required test optional",
        "what majors programs does the university offer",
        "is on campus housing guaranteed freshman",
        "how do i contact the admissions office email phone",
        "what is the fafsa school code financial aid",
        "what are the transfer credit evaluation policies",
        "when is the early action decision deadline november",
        "when is the regular decision deadline january",
        "what documents transcripts letters of recommendation required",
        "what is the application fee domestic international",
        "what toefl ielts english proficiency score accepted",
        "what merit scholarship amount awarded gpa",
        "what engineering computer science business arts programs",
        "when are decisions released notification march december",
        "what is the national candidate reply day deposit",
        "what is the css profile fafsa priority deadline",
        "how do i apply online portal common app",
        "can i visit campus tour information session",
        "what is the acceptance rate admission statistics",
        "what is the cost of attendance room board",
        "what is the work study program financial aid",
        "how do i submit my application materials",
        "what is the visa i-20 international student process",
        "how long does it take to receive admission decision",
        "what is early decision binding non binding",
        "do you accept duolingo english test score",
        "what gpa do i need for computer science engineering",
        "what is the graduate school application process",
        "how many letters of recommendation do i need",
        "what is the statement of purpose essay requirement",
        "is there a fee waiver for application",
        "admission requirements eligibility criteria",
        "tuition in state out of state cost",
        "housing dormitory residence hall freshman guarantee",
        "scholarship merit need based grant",
        "deadline early action regular decision",
        "any scholarships available offered",
        "freshman housing dorm guarantee",
        "transfer gpa requirements credits",
        "what majors offered programs",
    ],
    "GREETING_OR_CHITCHAT": [
        "hello hi hey there good morning good evening",
        "thank you thanks appreciate that helpful",
        "who are you what are you who made you",
        "what can you do help me",
        "goodbye bye see you later",
        "you are great awesome wonderful amazing",
        "how are you doing fine today",
        "nice to meet you pleasure talking",
        "what is your name",
        "are you a robot bot ai assistant",
        "tell me about yourself",
        "that is helpful thanks so much",
        "perfect great awesome thanks",
        "ok thanks alright sounds good",
        "cool interesting okay got it understood",
    ],
    "OUT_OF_SCOPE": [
        "who is the president prime minister politics government",
        "who is the president of the united states",
        "who is the prime minister of india",
        "what is python javascript programming code function",
        "how to cook bake recipe food",
        "how do i bake chocolate cake recipe ingredients",
        "how to make pasta bread dessert",
        "what is the weather temperature forecast today",
        "who won the match game cricket football sports",
        "who won the world cup soccer game yesterday",
        "write a poem story song lyrics",
        "write me a poem about love",
        "clear ur data reset delete remove",
        "what is the stock market price bitcoin crypto",
        "tell me a joke funny humor",
        "who is elon musk bill gates celebrity actor",
        "what is machine learning deep learning neural network model",
        "solve this math calculus equation algebra",
        "translate this text language french spanish german",
        "what is the capital city country world geography",
        "give me a code example algorithm sort python function",
        "explain quantum physics relativity science chemistry",
        "history world war ancient civilization roman",
        "recommend a movie book music song netflix",
        "what is the best phone laptop computer gaming",
        "health medical diagnosis symptoms treatment doctor",
        "what is the population of china india world",
        "who invented the telephone electricity telephone",
        "what is the speed of light physics constant",
        "how do planes fly aerodynamics lift thrust",
    ],
}

# ---------------------------------------------------------------------------
# Simple TF-IDF utilities (pure Python, no sklearn dependency)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _build_tf(tokens: list[str]) -> dict[str, float]:
    counts = Counter(tokens)
    total = max(len(tokens), 1)
    return {tok: count / total for tok, count in counts.items()}


def _build_idf(all_docs: list[list[str]]) -> dict[str, float]:
    n = len(all_docs)
    idf = {}
    all_vocab = set(tok for doc in all_docs for tok in doc)
    for word in all_vocab:
        df = sum(1 for doc in all_docs if word in doc)
        idf[word] = math.log((n + 1) / (df + 1)) + 1.0
    return idf


def _tfidf_vector(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    tf = _build_tf(tokens)
    return {tok: tf[tok] * idf.get(tok, 1.0) for tok in tf}


def _cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    common_keys = set(vec_a) & set(vec_b)
    if not common_keys:
        return 0.0
    dot = sum(vec_a[k] * vec_b[k] for k in common_keys)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Pre-build centroid vectors at import time
# ---------------------------------------------------------------------------

def _build_centroids():
    # Flatten each class into one big tokenised doc list and one centroid
    all_class_docs: dict[str, list[list[str]]] = {}
    for label, sentences in _SEED_CORPUS.items():
        all_class_docs[label] = [_tokenize(s) for s in sentences]

    # Build global IDF over the entire seed corpus
    flat_docs = [doc for docs in all_class_docs.values() for doc in docs]
    idf = _build_idf(flat_docs)

    # Build centroid TF-IDF vector per class (element-wise average)
    centroids = {}
    for label, docs in all_class_docs.items():
        vocab: dict[str, float] = {}
        for doc in docs:
            vec = _tfidf_vector(doc, idf)
            for k, v in vec.items():
                vocab[k] = vocab.get(k, 0.0) + v
        centroids[label] = {k: v / len(docs) for k, v in vocab.items()}

    return centroids, idf


_CENTROIDS, _IDF = _build_centroids()


# ---------------------------------------------------------------------------
# VAGUE / AMBIGUOUS heuristic thresholds
# ---------------------------------------------------------------------------

_VAGUE_MAX_TOKENS = 1       # only single tokens with no signal → VAGUE
_OOD_THRESHOLD   = 0.08    # if best admission sim < this → treat as OOD
_VAGUE_THRESHOLD = 0.08    # if best sim < this AND short → VAGUE


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class IntentResult:
    """Holds the intent classification result."""

    def __init__(self, intent: str, confidence: float, scores: dict[str, float]):
        self.intent     = intent
        self.confidence = round(confidence, 4)
        self.scores     = {k: round(v, 4) for k, v in scores.items()}

    def __repr__(self):
        return (
            f"IntentResult(intent='{self.intent}', "
            f"confidence={self.confidence}, scores={self.scores})"
        )


def classify_intent(text: str) -> IntentResult:
    """
    Classify the intent of the given user input string.

    Returns an IntentResult with:
      - intent     : one of IN_DOMAIN_ADMISSION, GREETING_OR_CHITCHAT,
                     OUT_OF_SCOPE, VAGUE_AMBIGUOUS
      - confidence : cosine similarity score to winning centroid (0-1)
      - scores     : similarity scores per class
    """
    tokens = _tokenize(text.strip())

    # --- Vague / ambiguous: very short with almost no signal ---------------
    if len(tokens) <= _VAGUE_MAX_TOKENS:
        return IntentResult("VAGUE_AMBIGUOUS", 0.0, {})

    query_vec = _tfidf_vector(tokens, _IDF)

    scores: dict[str, float] = {
        label: _cosine_similarity(query_vec, centroid)
        for label, centroid in _CENTROIDS.items()
    }

    best_label = max(scores, key=lambda k: scores[k])
    best_score = scores[best_label]

    # --- Short + very low confidence → still vague -------------------------
    if len(tokens) <= 4 and best_score < _VAGUE_THRESHOLD:
        return IntentResult("VAGUE_AMBIGUOUS", best_score, scores)

    return IntentResult(best_label, best_score, scores)
