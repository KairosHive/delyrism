# delyrism / frontend

Next.js 14 (App Router), Tailwind, TanStack Query, Zustand.  Talks to the
FastAPI backend in `../backend`.

## Dev

```bash
npm install
NEXT_PUBLIC_API_BASE=http://localhost:8000 npm run dev
```

## Deploy

Any platform that runs `npm run build && npm start` works (Vercel, Railway
nixpacks, Fly, etc.). Set `NEXT_PUBLIC_API_BASE` at build time to the public
URL of the FastAPI backend.
