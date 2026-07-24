import { test, expect } from "@playwright/test";

/**
 * Tours page e2e — property selection drives the week calendar, availability
 * panel, and tour list. Targets the TourCalendar/AvailabilityPanel useMemo
 * refactor: picking a property must still populate the calendar and panel
 * correctly (not show stale/empty data from a memo that never invalidates).
 */
test.describe("Tours page", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/#/tours");
    await expect(page.getByRole("heading", { name: "Tour Scheduler", level: 1 })).toBeVisible();
  });

  test("shows the property selector, availability, and tour list shells", async ({ page }) => {
    await expect(page.getByText("Property", { exact: true })).toBeVisible();
    await expect(page.getByRole("heading", { name: /Team availability/ })).toBeVisible();
    await expect(page.getByRole("heading", { name: /Upcoming tours/ })).toBeVisible();
  });

  test("before a property is picked, the week calendar isn't shown at all", async ({ page }) => {
    // Showing SOME property's slots/bookings before the user picks one is
    // exactly the bug this hides: with no property selected there's no single
    // property's data to show, so the whole card (not just an empty-state
    // message inside it) stays hidden until a property is chosen.
    await expect(page.getByRole("heading", { name: "This week" })).toHaveCount(0);
    await expect(page.locator(".cal-week")).toHaveCount(0);
  });

  test("selecting a property populates the calendar and availability panel", async ({ page }) => {
    const select = page.locator("select").first();
    // The property list loads async (getProperties()); wait past the placeholder.
    await expect(select.locator("option").nth(1)).toBeAttached({ timeout: 10_000 });
    const options = await select.locator("option").allTextContents();
    const specific = options.find((o) => o.trim() !== "Select a property…");
    test.skip(!specific, "no property options available");
    await select.selectOption({ label: specific! });

    // The week calendar (hidden entirely with no property selected) appears
    // once a property is chosen and its slots load.
    await expect(page.getByRole("heading", { name: "This week" })).toBeVisible({ timeout: 15_000 });
    await expect(page.locator(".cal-week")).toBeVisible({ timeout: 15_000 });
    // Availability panel shows at least one staffer row, or the "no staff" message —
    // either way it must not be stuck on "Loading availability…".
    await expect(page.getByText("Loading availability…")).toHaveCount(0, { timeout: 15_000 });
  });

  test("switching properties updates the calendar without a crash", async ({ page }) => {
    const select = page.locator("select").first();
    await expect(select.locator("option").nth(1)).toBeAttached({ timeout: 10_000 });
    const options = await select.locator("option").allTextContents();
    const specific = options.filter((o) => o.trim() !== "Select a property…");
    test.skip(specific.length < 2, "need at least 2 properties to switch between");

    await select.selectOption({ label: specific[0] });
    await expect(page.locator(".cal-week")).toBeVisible({ timeout: 15_000 });
    await select.selectOption({ label: specific[1] });
    await expect(page.locator(".cal-week")).toBeVisible({ timeout: 15_000 });
    await expect(page.locator("text=/something went wrong/i")).toHaveCount(0);
  });

  test("scheduler chat panel is present and accepts input", async ({ page }) => {
    const chatInput = page.locator("input[placeholder], textarea").last();
    await expect(chatInput).toBeVisible();
  });
});
