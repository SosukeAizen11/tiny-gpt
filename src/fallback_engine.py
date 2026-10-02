"""
src/fallback_engine.py
----------------------
Handles all non-admissions interactions gracefully:
  - Chit-chat / greetings  → warm persona-aligned replies
  - Out-of-scope queries   → polite redirect back to admissions scope
  - Vague / ambiguous      → helpful disambiguation prompts
"""

import random
import re


# ---------------------------------------------------------------------------
# Response pools  (randomized to avoid robotic repetition)
# ---------------------------------------------------------------------------

_GREETING_RESPONSES = [
    (
        "Hello! I'm the University Admissions Assistant powered by TinyGPT. "
        "I'm here to help with application deadlines, tuition, scholarships, GPA requirements, "
        "housing, programs, and more. What can I help you with today?"
    ),
    (
        "Hi there! Welcome to the University Admissions Office virtual assistant. "
        "I can answer questions about deadlines, tuition & aid, GPA requirements, "
        "academic programs, and campus life. How can I assist you?"
    ),
    (
        "Hey! I'm your University Admissions Bot. Ask me anything about the application "
        "process — deadlines, fees, scholarships, test requirements, or housing. Ready when you are!"
    ),
]

_FAREWELL_RESPONSES = [
    "Thank you for visiting the Admissions Office. Best of luck with your college journey — we hope to see you on campus!",
    "Goodbye! Don't hesitate to reach out to us at admissions@university.edu if you have further questions. Good luck!",
    "Farewell! Best wishes with your application. Feel free to come back anytime — I'm here to help!",
]

_THANKS_RESPONSES = [
    "You're welcome! Is there anything else about the admissions process I can help you with?",
    "Happy to help! Feel free to ask any other admissions questions — I'm here for you.",
    "Of course! Let me know if you have any more questions about deadlines, financial aid, or programs.",
]

_IDENTITY_RESPONSES = [
    (
        "I'm the University Undergraduate Admissions Assistant, powered by TinyGPT — "
        "a small custom-trained Transformer language model built in-house. "
        "I specialise exclusively in university admissions information. "
        "Type `faq` to see what I know, or ask me any admissions question!"
    ),
    (
        "Great question! I'm TinyGPT — a neural language model specifically trained on university "
        "admissions knowledge. I can tell you about GPA requirements, application deadlines, "
        "tuition & scholarships, housing guarantees, and much more. How can I help?"
    ),
]

_OUT_OF_SCOPE_RESPONSES = [
    (
        "I'm specialised exclusively in University Admissions topics and cannot assist with that. "
        "I can help you with:\n"
        "  • Application deadlines (Early Action, Regular Decision)\n"
        "  • Tuition & financial aid (FAFSA, scholarships)\n"
        "  • GPA & test requirements\n"
        "  • Academic programs & majors\n"
        "  • Campus housing & life\n"
        "  • Contact information\n"
        "Feel free to ask an admissions question or type `faq` to see common topics!"
    ),
    (
        "That's outside my area of expertise! I'm trained only on university admissions information. "
        "Try asking me about deadlines, tuition, scholarships, GPA requirements, or housing. "
        "You can also type `faq` for a list of frequently asked questions."
    ),
    (
        "I can only assist with university admissions topics. "
        "Questions about politics, technology, general knowledge, or system commands "
        "are outside my scope. Ask me about application requirements, fees, "
        "financial aid, or academic programs — I'd love to help with those!"
    ),
]

_VAGUE_RESPONSES_WITH_SUGGESTIONS = [
    (
        "It looks like your query is a bit short. Could you be more specific? "
        "Here are some things I can help with:\n"
        "  • Type `gpa` → GPA requirements\n"
        "  • Type `deadline` → Application deadlines\n"
        "  • Type `tuition` → Cost of attendance\n"
        "  • Type `scholarship` → Financial aid options\n"
        "  • Type `faq` → See all frequently asked questions"
    ),
    (
        "I'd love to help, but I need a bit more context! Try asking a full question like:\n"
        "  • 'What is the minimum GPA for admission?'\n"
        "  • 'When is the application deadline?'\n"
        "  • 'How much is tuition for in-state students?'\n"
        "  • 'Do you offer merit scholarships?'\n"
        "Or type `faq` to browse common questions!"
    ),
]

# ---------------------------------------------------------------------------
# Keyword triggers for sub-categories of greetings
# ---------------------------------------------------------------------------

_FAREWELL_KEYWORDS = {"bye", "goodbye", "farewell", "see you", "later", "exit", "quit"}
_THANKS_KEYWORDS   = {"thank", "thanks", "thank you", "appreciate", "helpful", "great", "awesome", "perfect"}
_IDENTITY_KEYWORDS = {"who are you", "what are you", "your name", "who made you", "what is tinygpt",
                      "are you a bot", "are you ai", "are you robot", "tell me about yourself"}


def _lower(text: str) -> str:
    return text.lower().strip()


def _contains_any(text: str, keywords: set[str]) -> bool:
    t = _lower(text)
    return any(kw in t for kw in keywords)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def handle_greeting(user_text: str) -> str:
    """Return an appropriate greeting/chitchat response based on the user text."""
    if _contains_any(user_text, _FAREWELL_KEYWORDS):
        return random.choice(_FAREWELL_RESPONSES)
    if _contains_any(user_text, _THANKS_KEYWORDS):
        return random.choice(_THANKS_RESPONSES)
    if _contains_any(user_text, _IDENTITY_KEYWORDS):
        return random.choice(_IDENTITY_RESPONSES)
    return random.choice(_GREETING_RESPONSES)


def handle_out_of_scope(user_text: str) -> str:
    """Return a polite out-of-scope redirection message."""
    return random.choice(_OUT_OF_SCOPE_RESPONSES)


def handle_vague(user_text: str) -> str:
    """Return a helpful disambiguation prompt for very short / ambiguous inputs."""
    return random.choice(_VAGUE_RESPONSES_WITH_SUGGESTIONS)
