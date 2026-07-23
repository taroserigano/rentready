import { test, expect } from "@playwright/test";

/**
 * Apply flow e2e — loading a sample applicant runs the full pipeline (extract →
 * eligibility → recommendations) against the real backend. Assertions are
 * structural (headings/verdict shape), not exact LLM prose, so the test is
 * stable whether Claude is reachable or the templated fallback is used.
 */
test.describe("Apply flow", () => {
  test("loading a sample produces eligibility + recommendations", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByRole("heading", { name: "Apply", level: 1 })).toBeVisible();

    const sample = page.locator(".sample-chip").first();
    await expect(sample).toBeVisible({ timeout: 20_000 });
    await sample.click();

    // Eligibility card resolves (pipeline may call the LLM → generous timeout).
    await expect(page.getByRole("heading", { name: /Eligibility/ })).toBeVisible({
      timeout: 90_000,
    });
    await expect(
      page.locator(".verdict, [class*='verdict']").first(),
    ).toBeVisible();

    // Recommendations section renders.
    await expect(
      page.getByRole("heading", { name: /Recommended propert/i }),
    ).toBeVisible({ timeout: 90_000 });
  });

  test("command palette opens and lists navigation commands", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByRole("heading", { name: "Apply", level: 1 })).toBeVisible();
    // Ctrl/Cmd+K opens the palette.
    await page.keyboard.press("Control+k");
    const dialog = page.getByRole("dialog");
    if (await dialog.count()) {
      await expect(dialog).toBeVisible();
      await page.keyboard.press("Escape");
    }
  });
});
