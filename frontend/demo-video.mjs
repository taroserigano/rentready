// Standalone Playwright script that records an interaction-rich video
// walkthrough of every RentReady page — typing into forms, running the
// eligibility pipeline, booking a tour end-to-end, chatting with each
// assistant, dragging the What-If risk sliders, and filtering/sorting tables.
//
// Runs against the already-running dev servers (frontend :5173 -> backend
// :8000). Uses system Chrome (channel: "chrome") so it never downloads a
// Playwright browser bundle (blocked by the corporate TLS proxy). Node
// resolves @playwright/test from ./node_modules because this file lives in
// frontend/.
//
//   node frontend/demo-video.mjs
//
// Output: frontend/demo/rentready-demo.webm  (path printed at the end)

import { chromium } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const BASE = process.env.DEMO_BASE_URL ?? "http://localhost:5173";
const OUT_DIR = path.resolve(HERE, "demo");
const SIZE = { width: 1600, height: 900 };

fs.mkdirSync(OUT_DIR, { recursive: true });

/** Dwell so the recording has time to show each state. */
const pause = (ms) => new Promise((r) => setTimeout(r, ms));

/** Run a step; never let one bad selector abort the whole recording. */
async function step(name, fn) {
  process.stdout.write(`• ${name}\n`);
  try {
    await fn();
  } catch (err) {
    process.stdout.write(`  ! skipped (${(err?.message ?? err).toString().split("\n")[0]})\n`);
  }
}

async function main() {
  const browser = await chromium.launch({ channel: "chrome", headless: true });
  const context = await browser.newContext({
    viewport: SIZE,
    recordVideo: { dir: OUT_DIR, size: SIZE },
    baseURL: BASE,
  });
  const page = await context.newPage();
  page.setDefaultTimeout(30_000);
  // Never let a stray native dialog hang the run.
  page.on("dialog", (d) => d.accept().catch(() => {}));

  const goto = async (hash) => {
    await page.goto(`${BASE}/${hash}`, { waitUntil: "networkidle" }).catch(() => {});
    await pause(600);
  };

  /** Visible, human-paced typing for the marquee text fields. */
  const type = async (locator, text, delay = 45) => {
    await locator.click();
    await locator.fill("");
    await locator.pressSequentially(text, { delay });
  };

  /** Scroll top→y→top so long pages reveal their content on camera. */
  const scrollTour = async (y = 1100, dwell = 900) => {
    await page.evaluate((yy) => window.scrollTo({ top: yy, behavior: "smooth" }), y);
    await pause(dwell);
    await page.evaluate(() => window.scrollTo({ top: 0, behavior: "smooth" }));
    await pause(500);
  };

  // ── 1. Apply — manual data entry → run the eligibility pipeline ───────────
  await step("Apply — fill the form by hand and check eligibility", async () => {
    await goto("");
    await page.getByRole("heading", { name: "Apply", level: 1 }).waitFor();
    await pause(700);

    await type(page.getByLabel("Name", { exact: true }), "Jordan Rivera");
    await type(page.getByLabel("Monthly income $", { exact: true }), "6800", 60);
    await type(page.getByLabel("Desired rent $", { exact: true }), "1950", 60);
    const credit = page.getByLabel("Credit score (optional)");
    if (await credit.count()) await type(credit, "728", 70);
    await pause(400);

    // Toggle an amenity chip.
    const gym = page.getByRole("button", { name: "+ Gym" });
    if (await gym.count()) {
      await gym.click();
      await pause(500);
    }
    // Expand the Employment section and add an employer.
    const employment = page.getByText("Employment", { exact: true });
    if (await employment.count()) {
      await employment.click();
      await pause(500);
      const employer = page.getByLabel("Employer");
      if (await employer.count()) await type(employer, "Acme Corp", 45);
    }
    await pause(500);

    // Run the pipeline: extract → eligibility → recommendations.
    await page.getByRole("button", { name: "Check my eligibility" }).click();
    await page
      .getByRole("heading", { name: /Eligibility/ })
      .waitFor({ timeout: 90_000 });
    await pause(1600);
    await scrollTour(1400, 1200);
    // Show the recommended properties section if it rendered.
    const recs = page.getByRole("heading", { name: /Recommended propert/i });
    if (await recs.count()) {
      await recs.scrollIntoViewIfNeeded();
      await pause(1600);
      await page.evaluate(() => window.scrollTo({ top: 0, behavior: "smooth" }));
      await pause(500);
    }
  });

  // ── 2. Applicants — open a saved applicant ────────────────────────────────
  await step("Applicants — open a saved applicant's profile", async () => {
    await goto("#/applicants");
    await page.getByRole("heading", { name: "Saved applicants", level: 1 }).waitFor();
    const rows = page.locator("table.table tbody tr");
    await rows.first().waitFor({ timeout: 15_000 });
    await pause(1200);
    await rows.first().click();
    await page.getByRole("heading", { name: /Eligibility/ }).waitFor({ timeout: 30_000 });
    await pause(1400);
    await scrollTour(900);
  });

  // ── 3. Properties — filter, sort, save, open a listing ────────────────────
  await step("Properties — filter, sort, save, open a listing", async () => {
    await goto("#/properties");
    await page.getByRole("heading", { name: "Browse homes", level: 1 }).waitFor();
    await page.locator(".prop-card").first().waitFor({ timeout: 15_000 });
    await pause(900);

    // Max-rent filter (debounced chip).
    const maxRent = page.getByPlaceholder("Any").first();
    if (await maxRent.count()) {
      await type(maxRent, "1800", 70);
      await pause(1600);
    }
    // Reveal advanced filters.
    const adv = page.getByRole("button", { name: /Advanced search/ });
    if (await adv.count()) {
      await adv.click();
      await pause(1200);
      const hide = page.getByRole("button", { name: /Hide advanced/ });
      if (await hide.count()) await hide.click();
      await pause(400);
    }
    // Sort high → low.
    const sort = page.locator("select").filter({ hasText: "Price: low to high" });
    if (await sort.count()) {
      await sort.selectOption({ label: "Price: high to low" });
      await pause(1200);
    }
    // Save the first home, filter to saved-only, then clear.
    await page.locator(".prop-card").first().locator("button").first().click();
    await pause(700);
    const savedOnly = page.getByRole("checkbox", { name: "Saved only" });
    if (await savedOnly.count()) {
      await savedOnly.check();
      await pause(1200);
      await savedOnly.uncheck();
      await pause(500);
    }
    // Open a listing detail dialog.
    await page.locator(".prop-card").first().click();
    const dialog = page.getByRole("dialog");
    if (await dialog.count()) {
      await dialog.waitFor({ timeout: 10_000 });
      await pause(1800);
      await page.keyboard.press("Escape");
      await pause(600);
    }
  });

  // ── 4. Tours — chat, propose slots, book end-to-end ───────────────────────
  await step("Tours — chat to find a slot and book a tour", async () => {
    await goto("#/tours");
    await page.getByRole("heading", { name: "Tour Scheduler", level: 1 }).waitFor();
    await pause(700);
    const select = page.locator("select").first();
    await select.locator("option").nth(1).waitFor({ state: "attached", timeout: 10_000 });
    const opts = await select.locator("option").allTextContents();
    const real = opts.find((o) => o.trim() !== "Select a property…");
    if (!real) return;
    await select.selectOption({ label: real });
    await page.locator(".cal-week").waitFor({ timeout: 15_000 });
    await pause(1200);

    const sched = page.locator(".card", { hasText: "Schedule a tour" });
    // Ask in plain language.
    const chatInput = sched.locator(".chat-form input");
    if (await chatInput.count()) {
      await type(chatInput, "I'd like to tour this week, an afternoon works best", 35);
      await sched.locator(".chat-form button[type=submit]").click();
    } else {
      // Fall back to a starter chip.
      await sched.locator(".chip").first().click();
    }
    // Wait for the assistant to propose bookable slots.
    const slotBtn = sched.locator('button[aria-label^="Book "]');
    await slotBtn.first().waitFor({ timeout: 30_000 });
    await pause(1400);
    await slotBtn.first().click();

    // Contact form appears — fill it and confirm the booking.
    const nameField = sched.getByPlaceholder("Jane Doe");
    await nameField.waitFor({ timeout: 15_000 });
    await type(nameField, "Jordan Rivera", 35);
    await type(sched.getByPlaceholder("(512) 555-0100"), "(512) 555-0142", 25);
    await type(sched.getByPlaceholder("jane@example.com"), "jordan.rivera@example.com", 20);
    await pause(600);
    await sched.getByRole("button", { name: /Confirm tour/ }).click();
    // Booking confirmed → contact form disappears, tour lands in the list.
    await nameField.waitFor({ state: "detached", timeout: 30_000 }).catch(() => {});
    await pause(1800);
    await scrollTour(900);
  });

  // ── 5. Ask — a multi-turn concierge conversation ──────────────────────────
  await step("Ask — a two-turn concierge conversation", async () => {
    await goto("#/ask");
    await page.getByLabel("Scope to a property").waitFor();
    await pause(600);
    const select = page.getByLabel("Scope to a property");
    await select.locator("option").nth(1).waitFor({ state: "attached", timeout: 10_000 });
    const opts = await select.locator("option").allTextContents();
    const real = opts.find((o) => o.trim() !== "Select a property…");
    if (!real) return;
    await select.selectOption({ label: real });
    await pause(700);

    const input = page.getByPlaceholder("e.g. What's the pet policy?");
    const ask = page.locator(".chat-form").getByRole("button", { name: "Ask" });
    await type(input, "What's the monthly rent, and is the deposit refundable?", 32);
    await ask.click();
    await page.locator(".chat-msg.bot").first().waitFor({ timeout: 30_000 });
    await pause(1600);

    // Follow-up turn.
    await type(input, "Great — are pets allowed, and is there a fee?", 32);
    await ask.click();
    await page.locator(".chat-msg.bot").nth(1).waitFor({ timeout: 30_000 });
    await pause(1600);
    await scrollTour(700);
  });

  // ── 6. Dashboard — KPIs + charts ──────────────────────────────────────────
  await step("Dashboard — KPI tiles + charts", async () => {
    await goto("#/dashboard");
    await page.getByRole("heading", { name: "Dashboard", level: 1 }).waitFor();
    await page.locator(".recharts-wrapper").first().waitFor({ timeout: 15_000 }).catch(() => {});
    await pause(1200);
    await scrollTour(1500, 1200);
    const refresh = page.getByRole("button", { name: "Refresh" }).first();
    if (await refresh.count()) {
      await refresh.click();
      await pause(1000);
    }
  });

  // ── 7. Risk — filter, drill in, What-If sliders, ask the assistant ────────
  await step("Risk — filter, drill in, drag What-If, ask the assistant", async () => {
    await goto("#/risk");
    await page.getByRole("heading", { name: "Late-Payment Risk", level: 1 }).waitFor();
    await page.locator("table.table tbody tr").first().waitFor({ timeout: 15_000 });
    await pause(900);

    // Search then clear.
    const search = page.getByPlaceholder("Search applicants…");
    if (await search.count()) {
      await type(search, "Sam", 60);
      await pause(1200);
      await search.fill("");
      await pause(500);
    }
    // Drill into the top-ranked applicant.
    await page.locator("table.table tbody tr").first().click();
    await pause(1400);
    await scrollTour(1000, 1000);

    // What-If: open, then drag the credit-score slider up and watch it re-score.
    const explore = page.getByRole("button", { name: /Explore factors/ });
    if (await explore.count()) {
      await explore.click();
      await pause(900);
      const creditSlider = page.locator('input.slider[aria-label="Credit score"]');
      if (await creditSlider.count()) {
        await creditSlider.scrollIntoViewIfNeeded();
        await creditSlider.focus();
        for (let i = 0; i < 12; i++) {
          await page.keyboard.press("ArrowRight");
          await pause(110);
        }
        await pause(1600); // let the debounced re-score + gauge settle
      }
    }

    // Ask the risk assistant about the selected applicant.
    const riskChat = page.locator(".card", { hasText: "Ask About Risk" });
    if (await riskChat.count()) {
      await riskChat.scrollIntoViewIfNeeded();
      const rInput = riskChat.locator(".chat-form input");
      await type(rInput, "Why is this applicant scored this way?", 32);
      await riskChat.locator(".chat-form").getByRole("button", { name: "Ask" }).click();
      await riskChat.locator(".chat-msg.bot, .risk-chat-artifact").first()
        .waitFor({ timeout: 30_000 }).catch(() => {});
      await pause(2000);
      await scrollTour(900);
    }
  });

  // ── 8. Residents — portfolio forecast chat, then a resident drill-in ──────
  await step("Residents — forecast chat, score a property, drill into a resident", async () => {
    await goto("#/residents");
    await page.getByRole("heading", { name: "Residents", level: 1 }).waitFor();
    await page.locator(".res-prop-card").first().waitFor({ timeout: 15_000 });
    await pause(1000);

    // Portfolio-scoped forecast question (signature feature).
    const resChat = page.locator(".card", { hasText: "Ask about residents" });
    if (await resChat.count()) {
      const rInput = resChat.locator(".chat-form input");
      await type(rInput, "How many late payments could we anticipate next quarter?", 30);
      await resChat.locator(".chat-form button[type=submit]").click();
      await resChat.locator(".chat-msg.bot").last().waitFor({ timeout: 30_000 });
      await pause(2200);
    }

    // Score a property's residents.
    await page.locator(".res-prop-card").first().click();
    const rows = page.locator("table.table tbody tr");
    await rows.first().waitFor({ timeout: 30_000 });
    await pause(1200);

    // Search within the loaded table, then clear.
    const resSearch = page.getByPlaceholder("Search unit or resident…");
    if (await resSearch.count()) {
      const firstName = (await rows.first().locator("td").first().textContent())?.trim() ?? "";
      if (firstName) {
        await type(resSearch, firstName.split(" ")[0], 60);
        await pause(1200);
        await resSearch.fill("");
        await pause(500);
      }
    }
    await scrollTour(1000, 1000);

    // Drill into a resident: prediction detail + ledger + forecast charts.
    await rows.first().click();
    await page.getByRole("heading", { name: /Payment history/ }).waitFor({ timeout: 30_000 });
    await pause(1600);
    await scrollTour(1500, 1400);
  });

  await pause(800);

  const video = page.video();
  await context.close(); // flushes the .webm to disk
  await browser.close();

  const recorded = video ? await video.path() : null;
  if (recorded) {
    const finalPath = path.resolve(OUT_DIR, "rentready-demo.webm");
    try {
      if (fs.existsSync(finalPath) && finalPath !== recorded) fs.rmSync(finalPath);
      fs.renameSync(recorded, finalPath);
      process.stdout.write(`\nVIDEO: ${finalPath}\n`);
    } catch {
      process.stdout.write(`\nVIDEO: ${recorded}\n`);
    }
  } else {
    process.stdout.write(`\nNo video was recorded.\n`);
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
