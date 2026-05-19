"""Tone presets for the story generator.

Ported from the old Streamlit `delyrism/app.py`.  Each named tone supplies
per-language `directives`, `avoid` and `lexicon` blocks that get appended to
the LLM prompt.  Without these the model only sees a bare label like
"tone=pynchon" and falls back to its training-set caricature of that author
— which (for Pynchon at least) is why every story was starting with
"As she navigates the labyrinthine...".
"""
from __future__ import annotations
from typing import Dict, List

# Literary-author presets — rich, multi-clause directives.
TONE_PRESETS: Dict[str, Dict[str, dict]] = {
    "pynchon": {
        "en": {
            "directives": (
                "Style: long, braided sentences with occasional sudden fragments; paranoid, satirical undertone; "
                "dense technical and historical vocabulary; quick zooms from street-level detail to systems theory. "
                "Favor appositives, parentheticals, sly asides in em-dashes, and conspiratorial hints. "
                "Weave motifs as if signals in a noisy network—interference, entropy, logistics, bureaucracy. "
                "Let humor flicker dryly; never explain the joke."
            ),
            "avoid": (
                "Avoid tidy morals, flat exposition, contemporary internet slang, and the cliché opening "
                '"As she/he navigates the labyrinthine..." — start somewhere unexpected.'
            ),
            "lexicon": ["entropy", "carrier wave", "paper trail", "solder", "ledger", "detritus", "ballistic", "telemetry", "archive"],
        },
        "fr": {
            "directives": (
                "Style : phrases longues et tressées, apartés ironiques ; humour sec ; vocabulaire technique et historique. "
                "Glisse de la micro-sensation au système global ; sous-texte paranoïaque, satirique."
            ),
            "avoid": "Évite les explications plates, l'argot web contemporain, et l'ouverture cliché « En traversant le dédale… ».",
            "lexicon": ["entropie", "onde porteuse", "registre", "soudure", "archives"],
        },
        "es": {
            "directives": (
                "Estilo: frases largas entretejidas, apartes irónicos; humor seco; léxico técnico e histórico. "
                "Saltos de lo microscópico a lo sistémico; subtexto paranoico y satírico."
            ),
            "avoid": "Evita moralejas claras, jerga de internet moderna, y la apertura cliché «Mientras navega por el laberíntico…».",
            "lexicon": ["entropía", "onda portadora", "archivo", "soldadura", "bitácora"],
        },
    },
    "blake": {
        "en": {
            "directives": (
                "Style: prophetic lyricism with visionary imagery; elevated diction; choral cadences; "
                "antinomies (Innocence/Experience, Fire/Water); occasional archaic turns; "
                "capitalized abstract Nouns as presences. Employ parallelism and anaphora."
            ),
            "avoid": "Avoid modern bureaucratic phrasing and pop-culture references.",
            "lexicon": ["Tyger", "Albion", "Urizen", "Eternity", "Lambent", "Firmament", "Vesture", "Anvil"],
        },
        "fr": {
            "directives": (
                "Style : lyrisme prophétique, images visionnaires ; diction élevée ; parallélismes et anaphores ; "
                "antinomies ; Noms abstraits capitalisés comme Présences."
            ),
            "avoid": "Évite le jargon administratif et les références pop.",
            "lexicon": ["Urizen", "Albion", "Firmament", "Vesture", "Éternité"],
        },
        "es": {
            "directives": (
                "Estilo: lirismo profético, imaginería visionaria; dicción elevada; paralelismos y anáforas; "
                "antinomias; Sustantivos abstractos en mayúscula como Presencias."
            ),
            "avoid": "Evita jerga administrativa y referencias pop.",
            "lexicon": ["Urizen", "Albion", "Firmamento", "Vestidura", "Eternidad"],
        },
    },
    "mystic-baroque": {
        "en": {
            "directives": (
                "Style: ornate, clause-rich sentences; sensuous concretes; theological shimmer; "
                "use asyndeton and periodic build-ups; switch between close tactile detail and cosmic scales."
            ),
            "avoid": "Avoid minimalism and corporate clichés.",
            "lexicon": ["thurible", "vellum", "nave", "coruscation", "meridian", "throne", "censorial"],
        },
        "fr": {
            "directives": "Style : baroque mystique ; phrases amples ; concret sensuel ; élans cosmiques.",
            "avoid": "Évite le minimalisme et le jargon d'entreprise.",
            "lexicon": ["encensoir", "vélin", "nef", "coruscation", "méridien"],
        },
        "es": {
            "directives": "Estilo: barroco místico; frases amplias; concreción sensual; amplitud cósmica.",
            "avoid": "Evita minimalismo y clichés corporativos.",
            "lexicon": ["incensario", "vitela", "nave", "coruscación", "meridiano"],
        },
    },
    "gnostic-techno": {
        "en": {
            "directives": (
                "Style: luminous cyber-gnostic register; terse sentences braided with sudden liturgical bursts; "
                "mix semiconductor jargon with apocryphal reverence. Let signal and revelation mirror each other."
            ),
            "avoid": "Avoid comic-book technobabble and over-explaining.",
            "lexicon": ["lattice", "gate", "angelic protocol", "firmware", "pleroma", "daemon", "checksum"],
        },
        "fr": {
            "directives": "Style : cyber-gnostique lumineux ; phrases brèves entrecoupées d'élans liturgiques.",
            "avoid": "Évite le techno-jargon caricatural.",
            "lexicon": ["trame", "passerelle", "pleroma", "daemon", "somme de contrôle"],
        },
        "es": {
            "directives": "Estilo: ciber-gnóstico luminoso; frases breves con irrupciones litúrgicas.",
            "avoid": "Evita tecnicismos caricaturescos.",
            "lexicon": ["retícula", "compuerta", "pleroma", "daemon", "suma de verificación"],
        },
    },
}

# Simple tone presets — just a short directive line, no avoid/lexicon.
SIMPLE_TONE_EXTRAS: Dict[str, Dict[str, str]] = {
    "dreamy": {
        "en": "soft focus, hypnagogic transitions, sensory synesthesia, ellipsis of motives, light anaphora.",
        "fr": "flou doux, transitions hypnagogiques, synesthésie sensorielle, ellipses de motifs.",
        "es": "foco suave, transiciones hipnagógicas, sinestesia sensorial, elipsis de motivos.",
    },
    "eerie": {
        "en": "quiet dread, negative space, mundane objects made numinous, withheld explanations, sparse adjectives.",
        "fr": "crainte silencieuse, vides, banal devenu numineux, explications retenues.",
        "es": "temor silencioso, espacios vacíos, lo banal hecho numinoso, explicaciones retenidas.",
    },
    "warm": {
        "en": "humane, grounded, gentle precision; small physical details carrying emotion; no irony.",
        "fr": "humain, ancré, précision douce ; petits détails physiques porteurs d'émotion.",
        "es": "humano, anclado, precisión amable; pequeños detalles físicos cargados de emoción.",
    },
}


def build_tone_extras(tone: str, lang_code: str) -> List[str]:
    """Return a list of extra prompt fragments for the chosen tone.

    Empty list if the tone has no preset — the prompt then just carries the
    bare 'tone=<name>' label, which is fine for a freeform tone the user
    types in but loses style guidance for the named-author presets.
    """
    out: List[str] = []
    preset = TONE_PRESETS.get(tone)
    if preset:
        loc = preset.get(lang_code, preset.get("en", {}))
        if loc.get("directives"):
            out.append(f"Style directives: {loc['directives']}")
        if loc.get("avoid"):
            out.append(f"Avoid: {loc['avoid']}")
        if loc.get("lexicon"):
            lex = ", ".join(map(str, loc["lexicon"][:10]))
            out.append(f"Suggested lexicon: {lex}")
        return out
    simple = SIMPLE_TONE_EXTRAS.get(tone, {})
    txt = simple.get(lang_code) or simple.get("en")
    if txt:
        out.append(f"Style directives: {txt}")
    return out
