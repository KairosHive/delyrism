// Shared visual constants — section palette + structure display names.
// Mirrors the colors from the old Streamlit app (`delyrism/app.py` ~line 1148).

export const SECTION_COLORS = {
  symbolic:   "#3498db", // data / structure       — blue
  embedding:  "#9b59b6", // model                  — purple
  context:    "#2ecc71", // sentence / weights     — green
  map:        "#f1c40f", // semantic map           — yellow
  ranking:    "#e67e22", // proposal               — orange
  subgraph:   "#e74c3c", // contextual subgraph    — red
  delta:      "#1abc9c", // Δ graph                — teal
} as const;

// Friendly names + glyphs for the built-in JSON presets. Falls back to the raw
// filename for any preset that isn't pre-mapped here.
export const STRUCTURE_DISPLAY: Record<string, string> = {
  elements:        "🜂  Four Elements",
  planets:         "☉  Celestial Planets",
  jungian:         "🜏  Jungian Archetypes",
  lakota:          "🪶  Lakota Dream Symbols",
  chakras:         "◉  Chakra System",
  chinese_zodiac:  "🐉  Chinese Zodiac",
  mayan:           "𐊗  Mayan Calendar",
  musical:         "♪  Musical Modes",
  architecture:    "🏛  Sacred Architecture",
  seasons_life:    "🌱  Seasons of Life",
};

export function displayPreset(name: string | null | undefined): string {
  if (!name) return "(custom)";
  return STRUCTURE_DISPLAY[name] ?? name;
}
