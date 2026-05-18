import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        // Deep, painterly palette — echoes the existing Streamlit aesthetic
        ink: {
          50: "#f5f7fa",
          100: "#e8edf3",
          200: "#cad4e0",
          300: "#9fadc1",
          400: "#6e7e95",
          500: "#4d5a6f",
          600: "#3a4458",
          700: "#2a3142",
          800: "#1b212e",
          900: "#10131c",
          950: "#070912",
        },
        accent: {
          // Nord-ish teal that matches the engine's default palette
          50: "#eafaf9",
          100: "#caf0ec",
          200: "#95e0d8",
          300: "#5fcfc4",
          400: "#3bbdb0",
          500: "#26a195",
          600: "#1e8278",
          700: "#1b675f",
          800: "#1a524b",
          900: "#0f3933",
        },
        warmth: "#d08770",
        warning: "#ebcb8b",
        danger: "#bf616a",
      },
      fontFamily: {
        sans: [
          "InterVariable",
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "Roboto",
          "sans-serif",
        ],
        mono: ["JetBrains Mono", "ui-monospace", "Menlo", "Consolas", "monospace"],
        display: ["InterDisplay", "Inter", "ui-sans-serif", "system-ui"],
      },
      boxShadow: {
        soft: "0 1px 0 rgba(255,255,255,0.04) inset, 0 1px 2px rgba(0,0,0,0.4)",
        glow: "0 0 0 1px rgba(59,189,176,0.25), 0 8px 24px -8px rgba(59,189,176,0.35)",
      },
    },
  },
  plugins: [],
};

export default config;
