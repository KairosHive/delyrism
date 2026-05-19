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

    # ---- 10 new tones ----

    "borges": {
        "en": {
            "directives": (
                "Style: labyrinths of references, dim libraries, recursive mirrors, philosophical "
                "paradoxes; treat motifs as cross-references — one image folds into another that "
                "negates it. Quiet erudition over melodrama. Ficciones-shaped: tight, lapidary, "
                "footnote-haunted. Each sentence implies an unwritten treatise."
            ),
            "avoid": "Avoid emotional outbursts, naive fantasy, and lush adjectival pile-ups.",
            "lexicon": ["catalogue", "library", "mirror", "labyrinth", "infinite", "treatise", "manuscript", "apocryphal", "tigre"],
        },
        "fr": {
            "directives": "Style : labyrinthes de références, bibliothèques sombres, miroirs récursifs, paradoxes philosophiques. Forme courte et érudite, ombre de la note de bas de page.",
            "avoid": "Évite les éclats émotionnels et la fantaisie naïve.",
            "lexicon": ["catalogue", "tigre", "bibliothèque", "miroir", "labyrinthe", "infini", "manuscrit"],
        },
        "es": {
            "directives": "Estilo: laberintos de referencias, bibliotecas oscuras, espejos recursivos, paradojas filosóficas. Concisión erudita; la sombra de una nota al pie.",
            "avoid": "Evita los arrebatos emocionales y la fantasía ingenua.",
            "lexicon": ["catálogo", "tigre", "biblioteca", "espejo", "laberinto", "infinito", "manuscrito"],
        },
    },

    "calvino": {
        "en": {
            "directives": (
                "Style: playful geometry, fable lightness, Invisible-Cities cataloguing — each "
                "motif becomes a city or a citizen; logic is precise but suspended like a kite. "
                "Warm intelligence, light touch, no irony of distance. Keep the camera high "
                "and humane; resist heavy psychology."
            ),
            "avoid": "Avoid heavy interiority and modern slang — keep the elevation.",
            "lexicon": ["city", "kite", "scaffold", "rooftop", "cartographer", "atlas", "pendulum", "geometry"],
        },
        "fr": {
            "directives": "Style : géométrie joueuse, légèreté de fable, catalogues à la 'Villes invisibles' ; logique précise mais suspendue ; chaleur sans ironie.",
            "avoid": "Évite la psychologie lourde ; garde l'élévation et la chaleur.",
            "lexicon": ["cerf-volant", "cité", "atlas", "cartographe", "pendule", "géométrie"],
        },
        "es": {
            "directives": "Estilo: geometría juguetona, ligereza de fábula, catalogación tipo 'Ciudades invisibles'; lógica precisa pero suspendida; calidez sin ironía.",
            "avoid": "Evita la interioridad psicológica densa; mantén la cámara alta y humana.",
            "lexicon": ["ciudad", "cometa", "atlas", "cartógrafo", "péndulo", "geometría"],
        },
    },

    "angela-carter": {
        "en": {
            "directives": (
                "Style: dark fairy-tale with baroque sensuality; transformed feminine archetypes; "
                "blood-red satin, abandoned manors, gothic furniture treated with sly irony; "
                "bodies metamorphose without warning; the moral is sharp and never tidy. Mix "
                "explicit and lapidary registers — fable meets boudoir."
            ),
            "avoid": "Avoid sanitized fairy-tale niceness, modern slang, and conventional moralizing.",
            "lexicon": ["bloom", "velvet", "thorn", "wolf", "mirror", "candlewax", "viscera", "satin", "moonblood"],
        },
        "fr": {
            "directives": "Style : conte sombre, sensualité baroque ; archétypes féminins transformés ; gothique ironique ; corps qui se métamorphosent ; la morale est tranchante.",
            "avoid": "Évite la mièvrerie des contes édulcorés et l'argot moderne.",
            "lexicon": ["velours", "épine", "loup", "miroir", "cire", "satin", "lune"],
        },
        "es": {
            "directives": "Estilo: cuento oscuro, sensualidad barroca; arquetipos femeninos transformados; gótico irónico; cuerpos que mutan; la moraleja afilada.",
            "avoid": "Evita la dulzura de cuento de hadas y la jerga moderna.",
            "lexicon": ["terciopelo", "espina", "lobo", "espejo", "cera", "satén", "luna"],
        },
    },

    "murakami": {
        "en": {
            "directives": (
                "Style: quiet surreal modern myth; magical-realist understatement; lonely interiors; "
                "the strange enters without fanfare. Cool flat sentences carry strange events. "
                "Ordinary objects — a record, a well, a cat, an elevator — function as ritual "
                "implements. Let the weird be mundane."
            ),
            "avoid": "Avoid magical-realist excess or comic-book strangeness — make the weird ordinary.",
            "lexicon": ["well", "stairwell", "jazz record", "cat", "elevator", "highway", "midnight", "kitchen"],
        },
        "fr": {
            "directives": "Style : surréalisme silencieux ; mythe moderne ; intérieurs solitaires ; l'étrange entre sans fanfare ; phrases plates et fraîches ; le bizarre devient ordinaire.",
            "avoid": "Évite l'excès magique ; rends le bizarre ordinaire.",
            "lexicon": ["puits", "escalier", "disque de jazz", "chat", "ascenseur", "minuit", "cuisine"],
        },
        "es": {
            "directives": "Estilo: surrealismo silencioso; mito moderno; interiores solitarios; lo extraño entra sin fanfarrias; frases planas; lo raro se vuelve ordinario.",
            "avoid": "Evita el exceso mágico; haz mundano lo extraño.",
            "lexicon": ["pozo", "escalera", "disco de jazz", "gato", "ascensor", "medianoche", "cocina"],
        },
    },

    "tarkovsky": {
        "en": {
            "directives": (
                "Style: cinematic stillness rendered in prose; religious imagery treated profanely "
                "and reverently at once; slow time; long takes as paragraphs; water dripping, ruins, "
                "candles, horses in a field at dawn. Faces hold longer than they should. Silence "
                "is a character with weight."
            ),
            "avoid": "Avoid quick cuts, jokes, snappy resolution, and contemporary cultural reference.",
            "lexicon": ["candle", "icon", "rain", "ruin", "horse", "well", "ash", "lamp", "milk"],
        },
        "fr": {
            "directives": "Style : immobilité cinématographique ; iconographie religieuse profanée ; temps lent ; eau, ruines, bougies, chevaux ; le silence comme personnage.",
            "avoid": "Évite les coupes rapides, les blagues, la résolution.",
            "lexicon": ["bougie", "icône", "pluie", "ruine", "cheval", "cendre", "lampe"],
        },
        "es": {
            "directives": "Estilo: quietud cinematográfica; iconografía religiosa profanada; tiempo lento; agua, ruinas, velas, caballos; el silencio como personaje.",
            "avoid": "Evita los cortes rápidos, los chistes y la resolución.",
            "lexicon": ["vela", "icono", "lluvia", "ruina", "caballo", "ceniza", "lámpara"],
        },
    },

    "homeric": {
        "en": {
            "directives": (
                "Style: oral-epic cadence; formulaic epithets ('grey-eyed', 'wine-dark', "
                "'cloud-gathering'); ringing catalogues; gods and weather interleaved with human "
                "gesture; the shadow of hexameter under the prose. Treat motifs as omens, gifts, "
                "or trophies. Names ring like bronze."
            ),
            "avoid": "Avoid modern interiority, irony, and psychological subtlety.",
            "lexicon": ["bronze", "wine-dark", "shield-bearing", "swift", "spear", "ash-shaft", "loom", "hearth", "ox"],
        },
        "fr": {
            "directives": "Style : cadence épique orale ; épithètes formulaires ; catalogues sonores ; dieux et météo entrelacés ; ombre de l'hexamètre ; les noms sonnent comme du bronze.",
            "avoid": "Évite l'intériorité moderne et l'ironie.",
            "lexicon": ["bronze", "lance", "frêne", "métier à tisser", "foyer", "bœuf"],
        },
        "es": {
            "directives": "Estilo: cadencia épica oral; epítetos formulares; catálogos sonoros; dioses y clima entrelazados; sombra del hexámetro; los nombres suenan a bronce.",
            "avoid": "Evita la interioridad moderna y la ironía.",
            "lexicon": ["bronce", "lanza", "fresno", "telar", "hogar", "buey"],
        },
    },

    "kafkaesque": {
        "en": {
            "directives": (
                "Style: bureaucratic dread without anger; opaque rules applied with patient "
                "courtesy; the body as institution; long sub-clauses that pile qualification on "
                "qualification; doors, corridors, forms, ledgers, summons. The protagonist treats "
                "absurdity as ordinary, even courteous."
            ),
            "avoid": "Avoid open emotional outbursts, explicit moral commentary, and resolution.",
            "lexicon": ["corridor", "ledger", "form", "summons", "doorman", "office", "stamp", "petitioner", "court"],
        },
        "fr": {
            "directives": "Style : effroi bureaucratique courtois ; règles opaques sans colère ; le corps comme institution ; longues subordonnées ; couloirs, portes, formulaires.",
            "avoid": "Évite les éclats émotionnels et le commentaire moral explicite.",
            "lexicon": ["couloir", "registre", "formulaire", "huissier", "bureau", "tampon", "tribunal"],
        },
        "es": {
            "directives": "Estilo: pavor burocrático cortés; reglas opacas sin enfado; cuerpo como institución; subordinadas largas; pasillos, puertas, formularios.",
            "avoid": "Evita los estallidos emocionales y el comentario moral explícito.",
            "lexicon": ["pasillo", "registro", "formulario", "ujier", "oficina", "sello", "tribunal"],
        },
    },

    "cosmic-horror": {
        "en": {
            "directives": (
                "Style: Lovecraftian indifference; deep time; non-Euclidean geometry as moral "
                "category; the protagonist as small witness to something that does not see them. "
                "Latinate abstract nouns; refusals to name. The horror is scale, not gore — the "
                "wrongness of proportion, not the wound."
            ),
            "avoid": "Avoid jump-scares, monsters described in detail, gore, and contemporary slang.",
            "lexicon": ["aeon", "lattice", "umbra", "abyss", "geometries", "indifferent", "lichen", "obsidian", "antediluvian"],
        },
        "fr": {
            "directives": "Style : indifférence lovecraftienne ; temps profond ; géométries non-euclidiennes ; témoin minuscule ; abstractions latinisantes ; refus de nommer.",
            "avoid": "Évite les sursauts, les monstres décrits en détail, le gore.",
            "lexicon": ["éon", "abîme", "umbra", "indifférent", "obsidienne", "antédiluvien"],
        },
        "es": {
            "directives": "Estilo: indiferencia lovecraftiana; tiempo profundo; geometrías no-euclidianas; testigo diminuto; abstracciones latinizadas; rechazo a nombrar.",
            "avoid": "Evita los sustos, los monstruos descritos en detalle y el gore.",
            "lexicon": ["eón", "abismo", "umbra", "indiferente", "obsidiana", "antediluviano"],
        },
    },

    "psalmic": {
        "en": {
            "directives": (
                "Style: liturgical repetition; second-person address ('You who…', 'O thou that…'); "
                "biblical cadence; parallel clauses; the world is invoked rather than described. "
                "Each motif is named and blessed. Use anaphora; let breath govern line length. "
                "Praise and lament intertwine."
            ),
            "avoid": "Avoid plot mechanics, contemporary cultural reference, and explanatory exposition.",
            "lexicon": ["O", "behold", "blessed", "stranger", "name", "shadow", "vessel", "covenant", "breath"],
        },
        "fr": {
            "directives": "Style : répétition liturgique ; adresse à la deuxième personne ; cadence biblique ; parallélismes ; le monde est invoqué plutôt que décrit ; anaphores ; louange et lamentation entrelacées.",
            "avoid": "Évite la mécanique du récit et les références contemporaines.",
            "lexicon": ["béni", "voici", "souffle", "alliance", "ombre", "vase", "nom"],
        },
        "es": {
            "directives": "Estilo: repetición litúrgica; segunda persona ('Tú que…'); cadencia bíblica; paralelismos; el mundo invocado más que descrito; anáforas; alabanza y lamento entrelazados.",
            "avoid": "Evita la mecánica del relato y las referencias contemporáneas.",
            "lexicon": ["he aquí", "bendito", "aliento", "alianza", "sombra", "vasija", "nombre"],
        },
    },

    "garcia-marquez": {
        "en": {
            "directives": (
                "Style: magical realism as ordinary register; hyperbolic family chronicle; long "
                "sentences that accumulate generations in a single arc; rain that lasts years; "
                "yellow butterflies, ice, mirrors. The miraculous and the trivial share a "
                "paragraph — and the same matter-of-fact tone."
            ),
            "avoid": "Avoid treating wonder as wonder — keep the tone matter-of-fact about miracles.",
            "lexicon": ["butterfly", "ice", "patriarch", "almond tree", "letter", "rain", "amnesia", "century"],
        },
        "fr": {
            "directives": "Style : réalisme magique comme registre ordinaire ; chronique familiale hyperbolique ; longues phrases qui accumulent des générations ; pluie qui dure des années ; papillons jaunes, glace.",
            "avoid": "Évite l'émerveillement traité comme émerveillement — ton matter-of-fact.",
            "lexicon": ["papillon", "glace", "patriarche", "amandier", "lettre", "pluie", "amnésie", "siècle"],
        },
        "es": {
            "directives": "Estilo: realismo mágico como registro ordinario; crónica familiar hiperbólica; oraciones largas que acumulan generaciones; lluvia de años; mariposas amarillas, hielo.",
            "avoid": "Evita tratar la maravilla como maravilla; mantén el tono matter-of-fact.",
            "lexicon": ["mariposa", "hielo", "patriarca", "almendro", "carta", "lluvia", "amnesia", "siglo"],
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
}


# --------------- Form presets ----------------
# The "Form" parameter controls the SHAPE of the output (one paragraph,
# poem with stanzas, myth, ritual incantation, etc.) — separate from
# Tone, which controls register/style.

FORM_PRESETS: Dict[str, Dict[str, str]] = {
    "prose": {
        "en": "Form: a single cohesive paragraph of prose. No lists, no headings, no line breaks.",
        "fr": "Forme : un seul paragraphe de prose cohérent. Pas de liste, pas de titre, pas de retour à la ligne.",
        "es": "Forma: un solo párrafo de prosa cohesivo. Sin listas, sin títulos, sin saltos de línea.",
    },
    "short-story": {
        "en": "Form: a short story across 2–4 short paragraphs — a small scene establishes; something turns; the close lands on a resonant image. Beginning, middle, end.",
        "fr": "Forme : une nouvelle en 2 à 4 brefs paragraphes — une petite scène s'installe, quelque chose bascule, la clôture tombe sur une image marquante. Début, milieu, fin.",
        "es": "Forma: un cuento breve en 2 a 4 párrafos cortos — una pequeña escena se instala, algo cambia, el cierre cae en una imagen resonante. Principio, medio, final.",
    },
    "poem": {
        "en": "Form: a poem with line breaks and (where useful) stanzas. Free verse — no fixed meter. Imagery and rhythm over narrative; each motif anchors a line or stanza.",
        "fr": "Forme : un poème avec sauts de ligne et (si utile) strophes. Vers libre, sans mètre fixe. Images et rythme plutôt que récit ; chaque motif ancre une ligne ou une strophe.",
        "es": "Forma: un poema con saltos de línea y (si conviene) estrofas. Verso libre, sin métrica fija. Imágenes y ritmo más que narración; cada motivo ancla una línea o estrofa.",
    },
    "myth": {
        "en": "Form: a myth — cosmic, etiological, 'in the time before time' register. Forces and figures are named like principles. No psychological interiority. The piece explains how something came to be.",
        "fr": "Forme : un mythe — registre cosmique et étiologique, « au temps d'avant le temps ». Forces et figures nommées comme des principes. Pas d'intériorité psychologique. Le texte raconte comment quelque chose advint.",
        "es": "Forma: un mito — registro cósmico, etiológico, «en el tiempo antes del tiempo». Fuerzas y figuras nombradas como principios. Sin interioridad psicológica. La pieza cuenta cómo algo llegó a ser.",
    },
    "incantation": {
        "en": "Form: an incantation — ritual repetition, second-person address ('You who…', 'Come now…'), performative. The text DOES rather than describes. Heavy anaphora; let phrases echo and accumulate.",
        "fr": "Forme : une incantation — répétition rituelle, adresse à la deuxième personne, performative. Le texte FAIT au lieu de décrire. Anaphores marquées ; laisse les phrases résonner et s'accumuler.",
        "es": "Forma: una incantación — repetición ritual, segunda persona ('Tú que…'), performativa. El texto HACE en vez de describir. Anáforas marcadas; deja que las frases resuenen y se acumulen.",
    },
    "vignette": {
        "en": "Form: a single vignette — one scene, one moment, one held image. No plot, no resolution. Sensory and exact; the motifs shape the visible field.",
        "fr": "Forme : une vignette — une scène, un instant, une image tenue. Pas d'intrigue, pas de résolution. Sensoriel et précis ; les motifs façonnent le champ visuel.",
        "es": "Forma: una viñeta — una escena, un momento, una imagen sostenida. Sin trama, sin resolución. Sensorial y exacto; los motivos dan forma al campo visual.",
    },
}


def build_form_directive(form: str, lang_code: str) -> str:
    """Return the form constraint line for the chosen output shape, or '' if
    unknown (fallback to free prose)."""
    preset = FORM_PRESETS.get(form)
    if not preset:
        return ""
    return preset.get(lang_code, preset.get("en", ""))


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
