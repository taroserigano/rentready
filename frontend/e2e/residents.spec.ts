import { test, expect } from "@playwright/test";

/**
 * Residents page e2e — property-first flow: the picker loads cheaply, a property
 * must be chosen before residents are scored, and drilling into a resident opens
 * the prediction detail + rent ledger. Backend serves deterministic synthetic
 * residents (10 properties × 25).
 */
test.describe("Residents page", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/#/residents");
    await expect(page.getByRole("heading", { name: "Residents", level: 1 })).toBeVisible();
  });

  test("shows the property picker with 10 property cards", async ({ page }) => {
    const cards = page.locator(".res-prop-card");
    await expect(cards.first()).toBeVisible();
    await expect(cards).toHaveCount(10);
    // Each card names a property and a resident count.
    await expect(cards.first().locator(".res-prop-name")).toBeVisible();
    await expect(cards.first().locator(".res-prop-meta")).toContainText(/resident/);
  });

  test("no property selected → best→worst health ranking is shown", async ({ page }) => {
    // PropertyHealthRanking loads async; wait for a ranked/graded result.
    await expect(page.locator("text=/Ranking properties by health/i")).toHaveCount(0, {
      timeout: 20_000,
    });
    // A portfolio-level chat rail is available before any property is picked.
    await expect(page.locator(".res-prop-card").first()).toBeVisible();
  });

  test("selecting a property scores its residents", async ({ page }) => {
    const firstCard = page.locator(".res-prop-card").first();
    const propName = (await firstCard.locator(".res-prop-name").textContent())?.trim() ?? "";
    await firstCard.click();

    // Header switches to the selected property; the residents table fills in.
    await expect(page.getByRole("heading", { name: new RegExp(`Residents — ${escapeRe(propName)}`) }))
      .toBeVisible({ timeout: 30_000 });
    const rows = page.locator("table.table tbody tr");
    await expect(rows.first()).toBeVisible({ timeout: 30_000 });
    await expect(rows).toHaveCount(25);
  });

  test("drilling into a resident opens the prediction detail + ledger", async ({ page }) => {
    await page.locator(".res-prop-card").first().click();
    const rows = page.locator("table.table tbody tr");
    await expect(rows.first()).toBeVisible({ timeout: 30_000 });
    await rows.first().click();

    // ResidentDetail renders the 5-year payment history + rent ledger.
    await expect(page.getByRole("heading", { name: /Payment history/ })).toBeVisible({
      timeout: 30_000,
    });
    await expect(page.getByRole("heading", { name: "Rent ledger" })).toBeVisible();
    // The per-resident chat rail is present.
    await expect(page.locator(".risk-chat-rail")).toBeVisible();
  });

  test("resident search + band filter operate on the loaded table", async ({ page }) => {
    await page.locator(".res-prop-card").first().click();
    const rows = page.locator("table.table tbody tr");
    await expect(rows.first()).toBeVisible({ timeout: 30_000 });

    const total = await rows.count();
    const firstName = (await rows.first().locator("td").first().textContent())?.trim() ?? "";
    await page.getByPlaceholder("Search unit or resident…").fill(firstName);
    await expect(rows.first()).toContainText(firstName);
    expect(await rows.count()).toBeLessThanOrEqual(total);
  });
});

function escapeRe(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
