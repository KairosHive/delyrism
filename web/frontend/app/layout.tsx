import "./globals.css";
import type { Metadata, Viewport } from "next";
import { Providers } from "./providers";

export const metadata: Metadata = {
  title: "Delyrism — Symbolic Archetype Explorer",
  description:
    "Context-conditioned embeddings, graph diffusion, and attention over symbolic spaces.",
};

// Mobile rendering needs an explicit viewport — without this, mobile
// Safari/Chrome assume a 980px viewport and shrink the whole UI to fit.
export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  // Allow zoom — the matrix heatmaps benefit from pinch-zoom on phones.
  maximumScale: 5,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
