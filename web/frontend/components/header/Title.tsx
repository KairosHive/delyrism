"use client";
import * as React from "react";

/**
 * Big centered DELYRISM wordmark with a column of runic glyphs dripping below.
 * Port of the Streamlit `delyrism-header` block (delyrism/app.py L1083-1280).
 */
export function Title() {
  return (
    <header className="delyrism-header">
      <div className="delyrism-title-container">
        <h1 className="delyrism-title">DELYRISM</h1>
        <div className="drip-container" aria-hidden>
          {DRIPS.map((g, i) => (
            <span key={i} className={`drip drip-${i}`}>{g}</span>
          ))}
        </div>
      </div>
      <div className="delyrism-subtitle">🧭 Archetype Explorer</div>
    </header>
  );
}

const DRIPS = ["ᛞ", "ᛖ", "ᛚ", "ʏ", "ᚱ", "ɪ", "ꜱ", "ᛗ", "◌", "∿"];
