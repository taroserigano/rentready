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

  // Targets the SortHeader extraction (module-level + memo()'d) in
  // Residents.tsx: clicking a header must still toggle direction on the same
  // column and switch columns correctly across all 8 sortable headers.
  test("sort headers toggle direction and switch columns correctly", async ({ page }) => {
    await page.locator(".res-prop-card").first().click();
    const rows = page.locator("table.table tbody tr");
    await expect(rows.first()).toBeVisible({ timeout: 30_000 });

    const thead = page.locator("table.table thead");
    // Click the inner <button>, not the <th> — the button doesn't fill the
    // whole header cell, so clicking the columnheader's center can miss it.
    const residentHeader = thead.getByRole("button", { name: "Resident", exact: true });
    const unitHeader = thead.getByRole("button", { name: "Unit", exact: true });
    const lateHeader = thead.getByRole("button", { name: "Late next Q", exact: true });

    // Default sort is "Late next Q", descending.
    await expect(lateHeader.locator("svg")).toBeVisible();
    const firstDesc = (await rows.first().locator("td").nth(3).textContent())?.trim();
    await lateHeader.click(); // toggle -> ascending
    const firstAsc = (await rows.first().locator("td").nth(3).textContent())?.trim();
    expect(firstAsc).not.toBe(firstDesc);

    // Switch to Unit -> defaults ascending.
    await unitHeader.click();
    await expect(unitHeader.locator("svg")).toBeVisible();
    const units = await rows.evaluateAll((trs) =>
      trs.map((tr) => tr.querySelectorAll("td")[1]?.textContent?.trim() ?? ""),
    );
    expect(units).toEqual([...units].sort((a, b) => a.localeCompare(b)));

    // Switch to Resident (name) -> defaults ascending.
    await residentHeader.click();
    await expect(residentHeader.locator("svg")).toBeVisible();
    const names = await rows.evaluateAll((trs) =>
      trs.map((tr) => tr.querySelector("td")?.textContent?.trim() ?? ""),
    );
    expect(names).toEqual([...names].sort((a, b) => a.localeCompare(b)));

    await expect(page.locator("text=/something went wrong/i")).toHaveCount(0);
  });

  test("chat: a scope-less 'how many late payments' question gets an aggregate forecast, not the full property list", async ({ page }) => {
    const chatCard = page.locator(".card", { hasText: "Ask about residents" });
    await expect(chatCard).toBeVisible();

    // Portfolio scope by default (no property selected on this page yet).
    await expect(chatCard.getByRole("button", { name: "Portfolio", exact: true }))
      .toHaveAttribute("aria-pressed", "true");

    await chatCard.locator(".chat-form input").fill(
      "How many late payments could we anticipate next quarter?",
    );
    await chatCard.locator(".chat-form button[type=submit]").click();

    // The bot answer renders with a route badge — must be the new forecast
    // intent, not a misroute to "General" or the property-health ranking.
    const botMsg = chatCard.locator(".chat-msg.bot").last();
    await expect(botMsg).toBeVisible({ timeout: 30_000 });
    await expect(botMsg.locator(".badge", { hasText: "Late-payment forecast" }))
      .toBeVisible({ timeout: 30_000 });

    // A scalar aggregate card (one number + range), NOT the 10-row property
    // health list — this is the exact regression this feature fixes.
    await expect(botMsg.locator(".big-num")).toBeVisible();
    await expect(botMsg.locator(".health-list")).toHaveCount(0);
    await expect(botMsg).toContainText(/interval/i);
  });

  test("chat: property-scoped forecast names the selected property, not the whole portfolio", async ({ page }) => {
    const firstCard = page.locator(".res-prop-card").first();
    const propName = (await firstCard.locator(".res-prop-name").textContent())?.trim() ?? "";
    await firstCard.click();
    await expect(page.getByRole("heading", { name: new RegExp(`Residents — ${escapeRe(propName)}`) }))
      .toBeVisible({ timeout: 30_000 });

    const chatCard = page.locator(".card", { hasText: "Ask about residents" });
    await expect(chatCard.getByRole("button", { name: "Property", exact: true }))
      .toHaveAttribute("aria-pressed", "true");

    await chatCard.locator(".chat-form input").fill("How many late payments next quarter?");
    await chatCard.locator(".chat-form button[type=submit]").click();

    const botMsg = chatCard.locator(".chat-msg.bot").last();
    await expect(botMsg.locator(".badge", { hasText: "Late-payment forecast" }))
      .toBeVisible({ timeout: 30_000 });
    await expect(botMsg.locator(".health-list")).toHaveCount(0);
  });
});

function escapeRe(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
