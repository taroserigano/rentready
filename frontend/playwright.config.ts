import { defineConfig, devices } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

// Absolute, OS-native paths so the webServer commands are safe under Windows
// cmd.exe (a command starting with "../" is parsed as the program name `..`).
const CONFIG_DIR = path.dirname(fileURLToPath(import.meta.url));
const REPO_DIR = path.resolve(CONFIG_DIR, "..");
const BACKEND_DIR = path.resolve(REPO_DIR, "backend");
const VENV_PY = path.resolve(REPO_DIR, ".venv", "Scripts", "python.exe");

// E2E runs on a DEDICATED backend (port 8100) with a throwaway copy of the DB,
// so tests that create/load applicants never mutate the real rentready.db.
// The copy is seeded from the real DB (its 4 real applicants) at config load —
// this runs in Node before the webServer starts, avoiding any race.
const E2E_PORT_BACKEND = 8100;
const E2E_PORT_FRONTEND = 5199;
const E2E_API = `http://localhost:${E2E_PORT_BACKEND}`;
const BASE_URL = `http://localhost:${E2E_PORT_FRONTEND}`;
const REAL_DB = path.resolve(REPO_DIR, "rentready.db");
const E2E_DB = path.resolve(REPO_DIR, "rentready.e2e.db");
try {
  if (fs.existsSync(REAL_DB)) fs.copyFileSync(REAL_DB, E2E_DB);
} catch {
  /* if the copy fails the backend just starts with a fresh (empty) e2e DB */
}

/**
 * E2E config for RentReady.
 *
 * Uses the *system* Google Chrome (`channel: "chrome"`) so we never download a
 * Playwright browser bundle — the corporate TLS proxy blocks that download, and
 * Chrome is already installed on this machine.
 *
 * `webServer` boots both tiers fresh on dedicated ports (backend 8100 with the
 * throwaway e2e DB, frontend Vite 5199 with `--strictPort`) so the run is fully
 * isolated from the dev servers on 8000/5173 and never touches real data.
 */
export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  workers: 1,
  forbidOnly: !!process.env.CI,
  retries: 0,
  reporter: [["list"]],
  timeout: 60_000,
  expect: { timeout: 15_000 },
  use: {
    baseURL: BASE_URL,
    channel: "chrome",
    headless: true,
    viewport: { width: 1440, height: 1200 },
    actionTimeout: 15_000,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [
    {
      name: "chrome",
      use: { ...devices["Desktop Chrome"], channel: "chrome" },
    },
  ],
  webServer: [
    {
      command: `"${VENV_PY}" -m uvicorn main:app --port ${E2E_PORT_BACKEND}`,
      cwd: BACKEND_DIR,
      env: { EMBEDDING_BACKEND: "hash", RENTREADY_DB: E2E_DB },
      url: `${E2E_API}/health`,
      reuseExistingServer: false,
      timeout: 180_000,
      stdout: "ignore",
      stderr: "pipe",
    },
    {
      command: `npm run dev -- --port ${E2E_PORT_FRONTEND} --strictPort`,
      env: { VITE_API_URL: E2E_API },
      url: BASE_URL,
      reuseExistingServer: false,
      timeout: 120_000,
      stdout: "ignore",
      stderr: "pipe",
    },
  ],
});
