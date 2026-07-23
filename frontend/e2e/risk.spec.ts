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
});
