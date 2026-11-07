import { defineConfig, devices } from "@playwright/test";
import path from "path";

// Browser tests of the website as it is deployed: the static export
// (`npm run build` → out/) served by the real API process, with the
// deterministic stand-in embedder (DELYRISM_TEST_EMBEDDER=1), so no network
// and no model is needed.  Build first, then `npm run e2e`.
//   PYTHON   the interpreter that has delyrism installed (default: python)
//   E2E_URL  test a running site instead (e.g. the Railway URL): no local server

const PORT = 8765;
const ROOT = path.resolve(__dirname, "..", "..");
const PY = process.env.PYTHON || "python";
const external = process.env.E2E_URL;

export default defineConfig({
  testDir: "./e2e",
  timeout: 60_000,
  expect: { timeout: 20_000 },
  fullyParallel: false,
  workers: 1,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [["github"], ["list"]] : "list",
  use: {
    baseURL: external || `http://127.0.0.1:${PORT}`,
    trace: "retain-on-failure",
    ...devices["Desktop Chrome"],
  },
  webServer: external
    ? undefined
    : {
        command: `"${PY}" -m delyrism.server --host 127.0.0.1 --port ${PORT} --static "${path.join(__dirname, "out")}"`,
        cwd: ROOT,
        url: `http://127.0.0.1:${PORT}/healthz`,
        env: { DELYRISM_TEST_EMBEDDER: "1", DELYRISM_OSC: "0", DELYRISM_SPEAK: "0", DELYRISM_MEDIA: "0" },
        reuseExistingServer: false,
        timeout: 120_000,
      },
});
