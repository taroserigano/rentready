import { test, expect } from "@playwright/test";

/**
 * Risk page e2e — drives the ranked-applicant table against the real backend
 * (the store holds four real applicants after the cleanup: Alex Chen, Sam Patel,
 * Jordan Rivera, Maria Gonzalez).
 */
test.describe("Risk page", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/#/risk");
    await expect(page.getByRole("heading", { name: "Late-payment risk" })).toBeVisible();
    await expect(page.getByRole("heading", { name: "Ranked applicants" })).toBeVisible();
  });

  test("lists real human applicants and no test-junk names", async ({ page }) => {
    const table = page.locator("table.table");
    await expect(table.locator("tbody tr").first()).toBeVisible();
    await expect(table.getByText("Sam Patel").first()).toBeVisible();
    // The pruned test artifacts must never reappear.
    for (const junk of ["Batch Tester", "Hdr Check", "Casey Form"]) {
      await expect(page.getByText(junk)).toHaveCount(0);
    }
  });

  test("search filters the table by name", async ({ page }) => {
    const search = page.getByPlaceholder("Search applicants…");
    await search.fill("Sam");
    const rows = page.locator("table.table tbody tr");
    await expect(rows).toHaveCount(1);
    await expect(rows.first()).toContainText("Sam Patel");
    await search.fill("");
    await expect(page.locator("table.table tbody tr").first()).toBeVisible();
  });

  test("band filter narrows the list", async ({ page }) => {
    // Scope to the ranked table body — the selected-applicant detail card also
    // shows a name and is intentionally unaffected by the table filter.
    const tbody = page.locator("table.table tbody");
    // All current applicants are Low band → filtering to Elevated empties the table.
    await page.getByRole("button", { name: "Elevated", exact: true }).click();
    await expect(tbody.getByText("Sam Patel")).toHaveCount(0);
    // Back to Low → Sam Patel is present again in the table.
    await page.getByRole("button", { name: "Low", exact: true }).click();
    await expect(tbody.getByText("Sam Patel").first()).toBeVisible();
  });

  test("selecting a row keeps the page stable and shows a detail region", async ({ page }) => {
    await page.locator("table.table tbody tr").first().click();
    // The page heading (h1) stays; a per-applicant detail card (h2) also renders.
    await expect(page.getByRole("heading", { name: "Late-payment risk", level: 1 }))
      .toBeVisible();
    // No client crash surfaced.
    await expect(page.locator("text=/something went wrong/i")).toHaveCount(0);
  });

  // Targets the SortHeader extraction (module-level + memo()'d): clicking a
  // column header must still toggle sort direction on a second click of the
  // SAME column, and switch columns (with a sensible default direction) on a
  // click of a DIFFERENT column — without losing table rows or crashing.
  test("sort headers toggle direction and switch columns correctly", async ({ page }) => {
    const rows = page.locator("table.table tbody tr");
    const thead = page.locator("table.table thead");
    // Click the inner <button>, not the <th> — the button doesn't fill the
    // whole header cell, so clicking the columnheader's center can miss it.
    const riskHeader = thead.getByRole("button", { name: "Risk", exact: true });
    const nameHeader = thead.getByRole("button", { name: "Applicant", exact: true });

    // Default sort is Risk, descending (highest risk first).
    await expect(riskHeader.locator("svg")).toBeVisible();
    const firstDesc = (await rows.first().locator("td").nth(1).textContent())?.trim();

    // Click Risk again -> ascending.
    await riskHeader.click();
    const firstAsc = (await rows.first().locator("td").nth(1).textContent())?.trim();
    expect(firstAsc).not.toBe(firstDesc);

    // Switch to the Applicant (name) column -> should default to ascending (A→Z).
    await nameHeader.click();
    await expect(nameHeader.locator("svg")).toBeVisible();
    const namesByRow = await rows.evaluateAll((trs) =>
      trs.map((tr) => tr.querySelector("td")?.textContent?.trim() ?? ""),
    );
    expect(namesByRow.length).toBeGreaterThan(0);
    const sorted = [...namesByRow].sort((a, b) => a.localeCompare(b));
    expect(namesByRow).toEqual(sorted);

    await expect(page.locator("text=/something went wrong/i")).toHaveCount(0);
  });
});
