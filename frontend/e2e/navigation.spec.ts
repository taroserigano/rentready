import { test, expect } from "@playwright/test";
import { clickNav, expectShell } from "./helpers";

test.describe("navigation + deep links", () => {
  test("clicking a nav tab updates the URL hash and the page", async ({ page }) => {
    await page.goto("/");
    await expectShell(page);

    await clickNav(page, "Risk Assessment");
    await expect(page).toHaveURL(/#\/risk$/);
    await expect(page.getByRole("heading", { name: "Late-payment risk", level: 1 })).toBeVisible();

    await clickNav(page, "Residents");
    await expect(page).toHaveURL(/#\/residents$/);
    await expect(page.getByRole("heading", { name: "Residents", level: 1 })).toBeVisible();
  });

  test("deep links render the right page directly", async ({ page }) => {
    await page.goto("/#/risk");
    await expect(page.getByRole("heading", { name: "Late-payment risk", level: 1 })).toBeVisible();

    await page.goto("/#/residents");
    await expect(page.getByRole("heading", { name: "Residents", level: 1 })).toBeVisible();

    await page.goto("/#/dashboard");
    await expectShell(page);
    await expect(page.getByRole("button", { name: "Dashboard", exact: true }).first())
      .toHaveClass(/active/);
  });

  test("an unknown hash falls back to Apply", async ({ page }) => {
    await page.goto("/#/does-not-exist");
    await expect(page.getByRole("heading", { name: "Apply", level: 1 })).toBeVisible();
  });

  test("back/forward navigation works", async ({ page }) => {
    await page.goto("/#/risk");
    await expect(page.getByRole("heading", { name: "Late-payment risk", level: 1 })).toBeVisible();
    await page.goto("/#/residents");
    await expect(page.getByRole("heading", { name: "Residents", level: 1 })).toBeVisible();
    await page.goBack();
    await expect(page.getByRole("heading", { name: "Late-payment risk", level: 1 })).toBeVisible();
    await page.goForward();
    await expect(page.getByRole("heading", { name: "Residents", level: 1 })).toBeVisible();
  });
});
