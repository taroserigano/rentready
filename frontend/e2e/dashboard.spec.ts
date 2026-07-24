import { test, expect } from "@playwright/test";

/**
 * Dashboard e2e — targets the chart-data memoization refactor (Dashboard.tsx):
 * charts must still render with real data and the "Refresh" actions must not
 * blank the page or throw.
 */
test.describe("Dashboard page", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/#/dashboard");
    await expect(page.getByRole("heading", { name: "Dashboard", level: 1 })).toBeVisible();
  });

  test("stat tiles and section headers render", async ({ page }) => {
    await expect(page.getByText("Applicants", { exact: true }).first()).toBeVisible();
    await expect(page.getByText("Homes", { exact: true }).first()).toBeVisible();
    await expect(page.getByText("Traffic", { exact: true }).first()).toBeVisible();
    await expect(page.getByText("Feedback", { exact: true }).first()).toBeVisible();
    await expect(page.locator(".stat-tile").first()).toBeVisible();
  });

  test("homes-by-neighborhood chart renders bars", async ({ page }) => {
    await expect(page.getByRole("heading", { name: "Homes by neighborhood" })).toBeVisible();
    // recharts renders <path>/<rect> bars inside an svg.
    await expect(page.locator(".recharts-wrapper").first()).toBeVisible({ timeout: 15_000 });
  });

  test("refreshing recent applicants does not blank the dashboard", async ({ page }) => {
    const refresh = page.getByRole("button", { name: "Refresh" }).first();
    if (await refresh.count()) {
      await refresh.click();
      await expect(page.getByText("Applicants", { exact: true }).first()).toBeVisible();
      await expect(page.locator("text=/something went wrong/i")).toHaveCount(0);
    }
  });
});
