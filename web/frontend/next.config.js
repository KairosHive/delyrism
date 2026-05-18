/** @type {import('next').NextConfig} */
const nextConfig = {
  // Static export so the FastAPI backend can serve the build at the same
  // origin — no separate Next.js Node server, no CORS, one Railway service.
  output: "export",
  reactStrictMode: true,
  images: { unoptimized: true }, // static export incompatible with image optimizer
  // Default to empty so fetch() falls back to relative URLs (same-origin).
  // In local dev, set NEXT_PUBLIC_API_BASE=http://localhost:8000 to point at
  // the standalone uvicorn server.
  env: {
    NEXT_PUBLIC_API_BASE: process.env.NEXT_PUBLIC_API_BASE || "",
  },
};
module.exports = nextConfig;
