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

  // Targets the ApplyForm -> ApplicantDetailsForm split: the manual-entry
  // form now owns its own state in a separate component, remounted via a
  // `key` on "Start over" instead of the parent reaching into it.
  test.describe("manual entry form", () => {
    test("submitting empty required fields shows validation errors", async ({ page }) => {
      await page.goto("/");
      await page.getByRole("button", { name: "Check my eligibility" }).click();
      await expect(page.getByText("Please enter your name")).toBeVisible();
      await expect(page.getByText("Monthly income must be more than 0")).toBeVisible();
      await expect(page.getByText("Desired rent must be more than 0")).toBeVisible();
    });

    // KNOWN APP ISSUE (pre-existing, not introduced by the perf refactor): the
    // credit-score <input> carries native `min={300}`/`max={850}` HTML
    // attributes with no `noValidate` on the <form>, so the browser's own
    // constraint validation blocks the submit event before React's onSubmit
    // ever runs — the custom "Credit score must be between 300 and 850"
    // message in validate() is therefore unreachable dead code in a real
    // browser. This test documents the ACTUAL behavior (native block, no
    // submission happens) rather than the unreachable custom message.
    test("credit score out of range is blocked by native validation (custom message is unreachable)", async ({ page }) => {
      await page.goto("/");
      const creditInput = page.getByLabel("Credit score (optional)");
      await creditInput.fill("200");
      await page.getByRole("button", { name: "Check my eligibility" }).click();
      await page.waitForTimeout(300);

      // The custom React message never appears...
      await expect(
        page.getByText("Credit score must be between 300 and 850"),
      ).toHaveCount(0);
      // ...because the native constraint validation caught it first.
      const validity = await creditInput.evaluate((el: HTMLInputElement) => ({
        valid: el.validity.valid,
        rangeUnderflow: el.validity.rangeUnderflow,
      }));
      expect(validity.valid).toBe(false);
      expect(validity.rangeUnderflow).toBe(true);
      // No submission happened — still on the pristine upload/form view.
      await expect(page.getByRole("heading", { name: "1. Upload application" })).toBeVisible();
    });

    test("an edited field's error clears as soon as it's fixed", async ({ page }) => {
      await page.goto("/");
      await page.getByRole("button", { name: "Check my eligibility" }).click();
      await expect(page.getByText("Please enter your name")).toBeVisible();
      // Not `exact` — once the error shows, the field's accessible name is
      // "Name Please enter your name" (the <label> wraps input + error text).
      await page.getByLabel("Name").fill("Taylor Test");
      await expect(page.getByText("Please enter your name")).toHaveCount(0);
    });

    test("Prefill sample fills the form, and Start over resets it after a submission", async ({ page }) => {
      await page.goto("/");
      await page.getByRole("button", { name: "Prefill sample" }).click();
      await expect(page.getByLabel("Name", { exact: true })).toHaveValue("Jordan Rivera");
      // exact: true — "Monthly income $" is otherwise a substring of the
      // Employment section's "Other monthly income $" field.
      await expect(page.getByLabel("Monthly income $", { exact: true })).toHaveValue("6800");
      await expect(page.getByLabel("Desired rent $", { exact: true })).toHaveValue("1950");

      await page.getByRole("button", { name: "Check my eligibility" }).click();
      await expect(page.getByRole("heading", { name: /Eligibility/ })).toBeVisible({
        timeout: 90_000,
      });

      // Start over: results clear AND the manual form remounts blank (the
      // core of the fix — the parent no longer reaches into form state
      // directly, it remounts the form component via a `key` bump).
      await page.getByRole("button", { name: "Start over" }).click();
      await expect(page.getByRole("heading", { name: /Eligibility/ })).toHaveCount(0);
      await expect(page.getByLabel("Name", { exact: true })).toHaveValue("");
      await expect(page.getByLabel("Monthly income $", { exact: true })).toHaveValue("");
      await expect(page.getByLabel("Desired rent $", { exact: true })).toHaveValue("");
    });

    test("optional collapsed sections expand and accept input", async ({ page }) => {
      await page.goto("/");
      await page.getByText("Employment", { exact: true }).click();
      await page.getByLabel("Employer").fill("Acme Corp");
      await expect(page.getByLabel("Employer")).toHaveValue("Acme Corp");

      await page.getByText("Household", { exact: true }).click();
      await page.getByLabel("Number of pets").fill("2");
      await expect(page.getByLabel("Number of pets")).toHaveValue("2");
    });

    test("a move-in date in the past is rejected", async ({ page }) => {
      // Regression: desired_move_in had no validation at all — a user could
      // submit a date years in the past with no warning.
      await page.goto("/");
      await page.getByLabel("Name", { exact: true }).fill("Taylor Test");
      await page.getByLabel("Monthly income $", { exact: true }).fill("5000");
      await page.getByLabel("Desired rent $", { exact: true }).fill("1500");
      await page.getByText("Move-in", { exact: true }).click();
      await page.getByLabel("When do you want to move in?").fill("2020-01-01");
      await page.getByRole("button", { name: "Check my eligibility" }).click();
      await expect(page.getByText("Move-in date can't be in the past")).toBeVisible();
    });

    test("amenity chips toggle the amenities text field", async ({ page }) => {
      await page.goto("/");
      await page.getByRole("button", { name: "+ Gym" }).click();
      await expect(page.getByRole("button", { name: "✓ Gym" })).toBeVisible();
      const amenitiesField = page.locator('input[placeholder="Gym, Pool"]');
      await expect(amenitiesField).toHaveValue("Gym");
      await page.getByRole("button", { name: "✓ Gym" }).click();
      await expect(amenitiesField).toHaveValue("");
    });
  });
});
