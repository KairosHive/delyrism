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


# Anchor contexts from paper/PLAN.md §6.
#
# Working set drawn from the Great Vision in *Black Elk Speaks: Being the Life
# Story of a Holy Man of the Oglala Sioux as told through John G. Neihardt*
# (Neihardt 1932).  C1 / C2 / C3 are deliberately chosen to span DISTINCT
# registers of the vision so the per-context relational rewiring is visibly
# different in fig01:
#
#   C1 — cosmic wholeness    (the sacred hoop / highest mountain)
#   C2 — thunder & power     (the rumbling cloud, the bay horse, the west)
#   C3 — sorrow & dissolution (the dream's end at Wounded Knee)
#
# C_A and C_B are the morph endpoints in fig03 (vision-arrival ↔ vision-loss).
# C_scene is for fig05 cross-modal (text + audio + image probes of one scene).
#
# IMPORTANT: these are sacred-tradition fragments selected from a widely-
# cited source.  Co-author / knowledge-holder (Kite) review is required
# before publication.  Any of these may be swapped — the figures re-run
# in minutes against a new CONTEXTS dict.
CONTEXTS = {
    "C1": (
        "I was standing on the highest mountain of them all, "
        "and round about beneath me was the whole hoop of the world."
    ),
    "C2": (
        "Then the Voice that was in the rumbling cloud spoke to me, "
        "and there at the dawn of the morning was a beautiful bay horse."
    ),
    "C3": (
        "A people's dream died there. It was a beautiful dream. "
        "The nation's hoop is broken and scattered; the sacred tree is dead."
    ),
    "C_A": (
        "Behold a sacred voice is calling you; "
        "all over the sky a sacred voice is calling."
    ),
    "C_B": (
        "A people's dream died there. It was a beautiful dream. "
        "The nation's hoop is broken and scattered; the sacred tree is dead."
    ),
    "C_scene": (
        "At the dawn of the morning there was a beautiful bay horse standing, "
        "and a sacred voice came rolling out of the rumbling cloud."
    ),
}

# Short human labels for figures.  Keep these short — they live on plot titles.
CONTEXT_LABELS = {
    "C1":      "the sacred hoop",
    "C2":      "thunder voice & dawn horse",
    "C3":      "the dream's end",
    "C_A":     "sacred voice calling",
    "C_B":     "the dream's end",
    "C_scene": "horse at dawn (storm + light)",
}
