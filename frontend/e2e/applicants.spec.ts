import { test, expect } from "@playwright/test";

/** Applicants directory e2e — list + open-detail flow. */
test.describe("Applicants directory", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/#/applicants");
    await expect(page.getByRole("heading", { name: "Saved applicants", level: 1 })).toBeVisible();
  });

  test("lists saved applicants", async ({ page }) => {
    const rows = page.locator("table.table tbody tr");
    await expect(rows.first()).toBeVisible({ timeout: 15_000 });
    await expect(page.getByText("Sam Patel")).toBeVisible();
  });

  test("opening a row loads the applicant's profile + eligibility", async ({ page }) => {
    const rows = page.locator("table.table tbody tr");
    await expect(rows.first()).toBeVisible({ timeout: 15_000 });
    await rows.first().click();
    await expect(page.getByRole("heading", { name: /Eligibility/ })).toBeVisible({ timeout: 30_000 });
    await expect(page.locator("text=/something went wrong/i")).toHaveCount(0);
  });

  test("refresh button reloads the list without error", async ({ page }) => {
    await page.getByRole("button", { name: "Refresh" }).click();
    await expect(page.locator("table.table tbody tr").first()).toBeVisible({ timeout: 15_000 });
  });
});
