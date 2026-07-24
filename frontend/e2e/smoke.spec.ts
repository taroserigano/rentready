import { test, expect } from "@playwright/test";
import { expectShell } from "./helpers";

test.describe("smoke", () => {
  test("app boots with the nav shell and default Apply view", async ({ page }) => {
    await page.goto("/");
    await expectShell(page);
    // Brand + all eight nav tabs render.
    for (const label of [
      "Apply",
      "Applicants",
      "Properties",
      "Tours",
      "Ask",
      "Dashboard",
      "Risk Assessment",
      "Residents",
    ]) {
      await expect(
        page.getByRole("button", { name: label, exact: true }).first(),
      ).toBeVisible();
    }
    // Default view is Apply.
    await expect(page.getByRole("heading", { name: "Apply", level: 1 })).toBeVisible();
  });

  test("removed nav tabs are gone (regression)", async ({ page }) => {
    await page.goto("/");
    await expectShell(page);
    for (const label of ["Evaluations", "Monitoring", "A/B Lab", "Learn"]) {
      await expect(
        page.getByRole("button", { name: label, exact: true }),
      ).toHaveCount(0);
    }
  });

  test("the 'How it works' explainer was removed (regression)", async ({ page }) => {
    await page.goto("/");
    await expectShell(page);
    await expect(page.getByText("How it works")).toHaveCount(0);
    await expect(page.getByText("Take the tour")).toHaveCount(0);
  });

  test("theme toggle flips data-theme", async ({ page }) => {
    await page.goto("/");
    await expectShell(page);
    const before = await page.evaluate(() =>
      document.documentElement.getAttribute("data-theme"),
    );
    await page.getByRole("button", { name: /Switch to (light|dark) theme/ }).click();
    const after = await page.evaluate(() =>
      document.documentElement.getAttribute("data-theme"),
    );
    expect(after).not.toBe(before);
  });
});
