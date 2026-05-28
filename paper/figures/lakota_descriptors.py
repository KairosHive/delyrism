"""The Lakota Shape Kit descriptors used in the v1 paper.

Source: notebooks/symbol_ambiguity_clean.ipynb (cell 1).  Reproduced here so the
paper figures can be regenerated without the notebook.

Kite-authored; treat as community material.  Do not modify without consultation.
"""
from __future__ import annotations

SYMBOLS_TO_DESCRIPTORS = {
    "BAG": [
        "bag", "movement", "storage", "transport", "physical medicine",
        "spiritual medicine", "peace pipe", "healing", "necessity", "holding",
        "physical travel", "container", "physical protection", "carrying", "journey", "food",
    ],
    "CLOUDS": [
        "clouds", "rain", "spiritual connection", "vision", "spiritual knowledge",
        "spirit world", "portal", "transcendence", "wisdom", "openness of the plane", "vastness",
    ],
    "DRAGONFLY": [
        "dragonfly", "resilience", "ancient connection", "ancestral", "confirmation",
        "message", "spiritual travel", "flight", "astral travel", "lightly", "blessing",
        "patience", "survival",
    ],
    "EARTH": [
        "earth", "people", "animals", "living beings", "plants", "roots", "mother",
        "geography", "mountain", "earthly", "mundane", "physical world", "harmony",
        "life", "sustainability", "balance", "grounding", "anchor", "stability",
    ],
    "FEATHER": [
        "feather", "achievement", "honor", "bravery", "merit", "counting coup",
        "courage", "knowledge", "eagle", "bird", "ceremony", "humbleness", "nobility",
        "virtue", "valor", "strength",
    ],
    "HORSE TRACK": [
        "horse track", "wealth", "travelling", "movement", "path", "motion",
        "riding", "hunting", "abundance", "dowry", "generosity", "west", "mobility",
    ],
    "HOUSE": [
        "house", "nurturing", "protection", "security", "home", "community",
        "matriarchy", "camp", "family", "care", "council", "grandmother", "safety",
    ],
    "LIGHTNING": [
        "lightning", "power", "thunderbird", "sacred clown", "spirituality", "storm",
        "spiritual connection", "electricity", "west", "fear", "grandfather", "danger",
        "goes out of the eyes of the thunderbird",
    ],
    "STAR": [
        "star", "medicine wheel", "four directions", "north", "ancestral connection",
        "stardust in our bones", "elemental", "extraterrestrial", "star people", "soul",
        "source", "light", "outer space", "balance",
    ],
    "THUNDER": [
        "thunder", "storm", "sound", "cry", "rain", "alert", "spirits arrival",
        "humility", "reminder of humanness", "fragility", "vulnerability",
        "out of the mouth of the thunderbird",
    ],
}


# Anchor contexts from paper/PLAN.md §6.  Placeholder set Kite may swap.
CONTEXTS = {
    "C1":      "A storm gathers over the hills, the air thick with waiting.",
    "C2":      "Birds rise at first light, the river runs cold and steady.",
    "C3":      "Smoke curls upward toward stars; voices fall silent.",
    "C_A":     "An old wound surfaces in a new season.",
    "C_B":     "Rain washes the trail clean; footprints disappear.",
    "C_scene": "A thunderhead gathers above the river at dusk.",
}

# Short human labels for figures.  Keep these short — they live on plot titles.
CONTEXT_LABELS = {
    "C1":      "storm gathering",
    "C2":      "first light",
    "C3":      "smoke rising",
    "C_A":     "old wound",
    "C_B":     "rain washing",
    "C_scene": "thunderhead at dusk",
}
