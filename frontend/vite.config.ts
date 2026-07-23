/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: { port: 5173 },
  test: {
    globals: true,
    environment: "jsdom",
    setupFiles: "./src/test/setup.ts",
    // Vitest owns the unit tests under src/; the e2e/ Playwright specs are run
    // by Playwright, not Vitest, so keep them out of this suite.
    include: ["src/**/*.{test,spec}.{ts,tsx}"],
  },
});
