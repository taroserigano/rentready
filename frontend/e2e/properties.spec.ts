import { test, expect } from "@playwright/test";

/**
 * Properties (browse homes) e2e — filters, debounce, sort, saved toggle, and
 * opening a listing. Targets the PropertyBrowser perf refactor: the Max rent
 * debounce, the narrowed server-fetch dependency, and the removed gridKey
 * remount (cards should reconcile, not disappear/reappear as a blank grid).
 */
test.describe("Properties page", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/#/properties");
    await expect(page.getByRole("heading", { name: "Browse homes", level: 1 })).toBeVisible();
  });

  test("lists homes and the match count", async ({ page }) => {
    await expect(page.locator(".prop-card").first()).toBeVisible({ timeout: 15_000 });
    await expect(page.locator(".filter-count")).toContainText(/of \d+ homes match/);
  });

  test("area filter narrows the list without an error", async ({ page }) => {
    await expect(page.locator(".prop-card").first()).toBeVisible({ timeout: 15_000 });
    const before = await page.locator(".prop-card").count();
    const areaSelect = page.locator("select").first();
    const options = await areaSelect.locator("option").allTextContents();
    const specificArea = options.find((o) => o !== "All areas");
    test.skip(!specificArea, "no specific area option available");
    await areaSelect.selectOption({ label: specificArea! });
    await expect(page.locator(".filter-count")).toContainText(/of \d+ homes match/);
    const after = await page.locator(".prop-card").count();
    expect(after).toBeLessThanOrEqual(before);
    await expect(page.locator("text=/something went wrong/i")).toHaveCount(0);
  });

  test("max rent debounces (no flicker/crash while typing) and produces an active filter chip", async ({ page }) => {
    await expect(page.locator(".prop-card").first()).toBeVisible({ timeout: 15_000 });
    const maxRent = page.getByPlaceholder("Any").first();
    await maxRent.fill("1500");
    // Chip should show the committed value once the debounce settles.
    await expect(page.locator(".active-filters")).toContainText("$1500/mo", { timeout: 2_000 });
    await expect(page.locator(".filter-count")).toContainText(/of \d+ homes match/);
    // Every remaining card should be at or under the cap (client + server agree).
    const cardTexts = await page.locator(".prop-card").allTextContents();
    expect(cardTexts.length).toBeGreaterThan(0);
    for (const t of cardTexts) {
      const m = t.match(/\$([\d,]+)\/mo/);
      expect(m).not.toBeNull();
      const n = Number(m![1].replace(/,/g, ""));
      expect(n).toBeLessThanOrEqual(1500);
    }
  });

  test("clearing all filters restores the full list and resets the max rent input", async ({ page }) => {
    await expect(page.locator(".prop-card").first()).toBeVisible({ timeout: 15_000 });
    const maxRent = page.getByPlaceholder("Any").first();
    await maxRent.fill("900");
    await expect(page.locator(".active-filters")).toBeVisible({ timeout: 2_000 });
    await page.getByRole("button", { name: "Clear all" }).click();
    await expect(page.locator(".active-filters")).toHaveCount(0);
    await expect(maxRent).toHaveValue("");
  });

  test("advanced search toggles extra filters", async ({ page }) => {
    await expect(page.getByLabel("Min rent ($ / month)")).toHaveCount(0);
    await page.getByRole("button", { name: /Advanced search/ }).click();
    await expect(page.getByText("Min rent ($ / month)")).toBeVisible();
    await page.getByRole("button", { name: /Hide advanced/ }).click();
    await expect(page.getByText("Min rent ($ / month)")).toHaveCount(0);
  });

  test("sorting does not blank the grid or error out", async ({ page }) => {
    await expect(page.locator(".prop-card").first()).toBeVisible({ timeout: 15_000 });
    const before = await page.locator(".prop-card").count();
    await page.locator("select").filter({ hasText: "Price: low to high" }).selectOption({ label: "Price: high to low" });
    await expect(page.locator(".prop-card").first()).toBeVisible();
    const after = await page.locator(".prop-card").count();
    expect(after).toBe(before);
    await expect(page.locator("text=/something went wrong/i")).toHaveCount(0);
  });

  test("saving a home updates the saved count and Saved only filter works", async ({ page }) => {
    await expect(page.locator(".prop-card").first()).toBeVisible({ timeout: 15_000 });
    await page.locator(".prop-card").first().locator("button").first().click();
    await expect(page.locator(".filter-count")).toContainText(/saved/, { timeout: 2_000 });
    const savedOnly = page.getByRole("checkbox", { name: "Saved only" });
    await savedOnly.check();
    await expect(page.locator(".prop-card")).toHaveCount(1);
    await savedOnly.uncheck();
  });

  test("opening a card shows the property detail panel", async ({ page }) => {
    await expect(page.locator(".prop-card").first()).toBeVisible({ timeout: 15_000 });
    await page.locator(".prop-card").first().click();
    await expect(page.getByRole("dialog")).toBeVisible({ timeout: 10_000 });
  });
});
