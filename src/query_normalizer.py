"""
src/query_normalizer.py
-----------------------
Normalizes and expands malformed / shorthand user queries into canonical
admissions questions that align with the model's training distribution.

Steps applied (in order):
  1. Strip & lowercase
  2. Remove repeated punctuation, question marks, noise
  3. Fuzzy keyword synonym expansion (abbreviations, typos)
  4. Canonical question template matching (shorthand → full sentence)
  5. Return the best canonical question if a match is found, else return
     cleaned original text.
"""

import re


# ---------------------------------------------------------------------------
# 1. Synonym / Abbreviation map  (fragment -> canonical keyword)
# ---------------------------------------------------------------------------

_SYNONYM_MAP: dict[str, str] = {
    # GPA / grades
    "gpa":           "gpa minimum grade point average",
    "grades":        "gpa",
    "grade":         "gpa",

    # Deadlines
    "deadline":      "application deadline",
    "deadlines":     "application deadline",
    "due date":      "application deadline",
    "due":           "application deadline",
    "when to apply": "application deadline",

    # Tuition / cost
    "tuition":       "tuition fee cost",
    "cost":          "tuition cost",
    "fees":          "tuition fees cost",
    "fee":           "application fee",
    "price":         "tuition cost",
    "expensive":     "tuition cost",
    "how much":      "tuition cost",

    # Scholarships / aid
    "scholarship":   "merit scholarship financial aid",
    "scholarships":  "merit scholarship financial aid",
    "any scholarship":  "do you offer merit scholarship financial aid",
    "any scholarships": "do you offer merit scholarship financial aid",
    "aid":           "financial aid scholarship",
    "grant":         "need based grant financial aid",
    "fafsa":         "fafsa financial aid school code",
    "css":           "css profile financial aid",

    # Housing
    "housing":       "on campus housing guaranteed",
    "dorm":          "on campus housing dormitory",
    "dorms":         "on campus housing dormitory",
    "accommodation": "on campus housing",
    "hostel":        "on campus housing",
    "room":          "on campus housing room",

    # Tests / requirements
    "sat":           "sat act test optional required",
    "act":           "sat act test optional required",
    "test":          "sat act test optional required",
    "toefl":         "toefl ielts english proficiency accepted score",
    "ielts":         "toefl ielts english proficiency accepted score",
    "english test":  "toefl ielts english proficiency accepted score",

    # Programs / majors
    "majors":        "academic programs majors offered",
    "major":         "academic programs majors offered",
    "programs":      "academic programs majors offered",
    "courses":       "academic programs majors offered",
    "cs":            "computer science major program",
    "comp sci":      "computer science major program",

    # Contact
    "contact":       "contact admissions office email phone",
    "email":         "admissions office email contact",
    "phone":         "admissions office phone number contact",
    "number":        "admissions office phone number contact",
    "address":       "admissions office location address",

    # Documents
    "documents":     "required documents transcripts letters recommendation",
    "transcripts":   "required transcripts official documents",
    "lor":           "letters of recommendation required",
    "sop":           "statement of purpose essay required",
    "essay":         "statement of purpose essay application",

    # Deposit / reply
    "deposit":       "enrollment deposit national candidate reply day",
    "confirm":       "enrollment deposit confirm admission offer",

    # General
    "admission":     "admission requirements application",
    "apply":         "how to apply application process",
    "international": "international student admissions visa i-20",
    "transfer":      "transfer student credit evaluation",
    "visit":         "campus visit tour information session",
}


# ---------------------------------------------------------------------------
# 2. Canonical question templates
#    (keyword fingerprint → canonical full question)
# ---------------------------------------------------------------------------

_CANONICAL_TEMPLATES: list[tuple[frozenset[str], str]] = [
    # GPA requirements
    (frozenset({"gpa", "minimum", "admission"}),
     "what is the minimum gpa for admission?"),
    (frozenset({"gpa", "transfer"}),
     "what is the minimum gpa for transfer admission?"),
    (frozenset({"gpa"}),
     "what is the minimum gpa for admission?"),

    # Deadlines
    (frozenset({"early", "action", "deadline"}),
     "when is the early action deadline?"),
    (frozenset({"early", "decision", "deadline"}),
     "when is the early decision deadline?"),
    (frozenset({"regular", "decision", "deadline"}),
     "when is the regular decision deadline?"),
    (frozenset({"deadline", "application"}),
     "when is the regular decision deadline?"),
    (frozenset({"deadline"}),
     "when is the regular decision deadline?"),

    # Tuition / cost
    (frozenset({"tuition", "state", "in"}),
     "how much is tuition for in-state students?"),
    (frozenset({"tuition", "out"}),
     "how much is tuition for out-of-state students?"),
    (frozenset({"tuition"}),
     "how much is tuition?"),
    (frozenset({"application", "fee"}),
     "what is the application fee?"),
    (frozenset({"cost", "attendance"}),
     "what is the total cost of attendance?"),

    # Scholarships / financial aid
    (frozenset({"scholarship", "merit"}),
     "what merit scholarships are available?"),
    (frozenset({"scholarship"}),
     "do you offer scholarships?"),
    (frozenset({"financial", "aid"}),
     "do you offer financial aid and scholarships?"),
    (frozenset({"fafsa", "code"}),
     "what is the school code for fafsa?"),
    (frozenset({"fafsa"}),
     "what is the school code for fafsa?"),

    # Housing
    (frozenset({"housing", "guaranteed"}),
     "is on-campus housing guaranteed?"),
    (frozenset({"housing"}),
     "is on-campus housing guaranteed?"),
    (frozenset({"dorm"}),
     "is on-campus housing guaranteed?"),

    # Tests
    (frozenset({"sat", "act", "required"}),
     "is the sat or act required?"),
    (frozenset({"sat"}),
     "is the sat or act required?"),
    (frozenset({"act"}),
     "is the sat or act required?"),
    (frozenset({"test", "optional"}),
     "is the sat or act required?"),
    (frozenset({"toefl", "ielts"}),
     "what english test scores are accepted?"),
    (frozenset({"english", "test"}),
     "what english test scores are accepted?"),

    # Programs
    (frozenset({"computer", "science", "major"}),
     "tell me about the computer science major."),
    (frozenset({"majors", "programs"}),
     "what majors does the university offer?"),
    (frozenset({"major"}),
     "what majors does the university offer?"),

    # Contact
    (frozenset({"contact", "admissions"}),
     "how do i contact the admissions office?"),
    (frozenset({"email", "admissions"}),
     "how do i contact the admissions office?"),
    (frozenset({"phone", "admissions"}),
     "how do i contact the admissions office?"),
    (frozenset({"contact"}),
     "how do i contact the admissions office?"),

    # Documents
    (frozenset({"documents", "required"}),
     "what documents are required for the application?"),
    (frozenset({"letters", "recommendation"}),
     "how many letters of recommendation do i need?"),
    (frozenset({"statement", "purpose"}),
     "is a statement of purpose required?"),
    (frozenset({"essay", "required"}),
     "is a statement of purpose required?"),

    # Deposit / enrollment
    (frozenset({"deposit", "enrollment"}),
     "what is the enrollment deposit deadline?"),

    # International
    (frozenset({"international", "visa"}),
     "what is the visa process for international students?"),
    (frozenset({"international", "admission"}),
     "what are the admission requirements for international students?"),

    # Transfer
    (frozenset({"transfer", "credits"}),
     "how are transfer credits evaluated?"),
    (frozenset({"transfer", "admission"}),
     "what are the transfer admission requirements?"),

    # Apply / how to
    (frozenset({"apply", "application"}),
     "how do i apply to the university?"),
    (frozenset({"apply"}),
     "how do i apply to the university?"),

    # Visit
    (frozenset({"campus", "visit"}),
     "how can i visit the campus?"),
    (frozenset({"tour"}),
     "how can i visit the campus?"),
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """Lowercase, remove repeated punctuation, normalise whitespace."""
    text = text.lower().strip()
    # Remove repeated punctuation or noise characters
    text = re.sub(r"[?!.]{2,}", "?", text)
    # Remove unrecognised characters (keep alphanumerics, spaces, basic punct)
    text = re.sub(r"[^a-z0-9 '\-?.,]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _expand_synonyms(text: str) -> str:
    """Replace known abbreviations/fragments with expanded canonical words."""
    for abbr, expansion in _SYNONYM_MAP.items():
        # Whole-word match
        text = re.sub(r"\b" + re.escape(abbr) + r"\b", expansion, text)
    return text


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


def _match_canonical(tokens: set[str]) -> str | None:
    """Return the first canonical question whose keyword fingerprint is a subset of tokens."""
    best_match = None
    best_overlap = 0
    for keywords, question in _CANONICAL_TEMPLATES:
        overlap = len(keywords & tokens)
        if overlap == len(keywords) and overlap > best_overlap:
            best_match = question
            best_overlap = overlap
    return best_match


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_query(text: str) -> tuple[str, bool]:
    """
    Normalize and attempt to canonicalize the user query.

    Returns:
        (normalized_text, was_expanded)
        - normalized_text : canonical question string if matched, else cleaned text
        - was_expanded    : True if a canonical template was applied
    """
    cleaned = _clean_text(text)
    expanded = _expand_synonyms(cleaned)
    tokens = _tokenize(expanded)

    canonical = _match_canonical(tokens)
    if canonical:
        return canonical, True

    # No template match: return the cleaned version
    return cleaned, False
