import { test, expect, type Locator } from "@playwright/test";

/**
 * Ask (Concierge) page e2e — targets the recent UX change requiring a
 * property to be selected before asking anything (no more "All properties"
 * option), and the compact chip-style starter suggestions.
 */

/** Property options load async (getProperties()); wait past the placeholder. */
async function firstRealOption(select: Locator): Promise<string> {
  await expect(select.locator("option").nth(1)).toBeAttached({ timeout: 10_000 });
  const options = await select.locator("option").allTextContents();
  const specific = options.find((o) => o.trim() !== "Select a property…");
  if (!specific) test.skip(true, "no property options available");
  return specific!;
}

test.describe("Ask page", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/#/ask");
    await expect(page.getByLabel("Scope to a property")).toBeVisible();
  });

  test("no property selected: asking is disabled and starters are disabled", async ({ page }) => {
    // "All properties" must be gone; the placeholder option is disabled.
    await expect(page.getByLabel("Scope to a property")).toHaveValue("");
    await expect(page.locator("option", { hasText: "All properties" })).toHaveCount(0);
    await expect(page.getByText("Select a property above to ask questions about it.")).toBeVisible();

    const input = page.getByPlaceholder("Select a property to ask…");
    await expect(input).toBeDisabled();
    await expect(page.locator(".chat-form").getByRole("button", { name: "Ask" })).toBeDisabled();

    const starter = page.locator(".ask-starter-tile").first();
    await expect(starter).toBeVisible();
    await expect(starter).toBeDisabled();
  });

  test("selecting a property enables asking and a starter chip sends a question", async ({ page }) => {
    const select = page.getByLabel("Scope to a property");
    const specific = await firstRealOption(select);
    await select.selectOption({ label: specific });

    await expect(page.getByPlaceholder("e.g. What's the pet policy?")).toBeEnabled();
    await expect(page.locator(".chat-form").getByRole("button", { name: "Ask" })).toBeEnabled();

    const starter = page.locator(".ask-starter-tile").first();
    await expect(starter).toBeEnabled();
    await starter.click();

    // A user bubble with the question, and a bot response, should appear.
    await expect(page.locator(".chat-msg.user").first()).toBeVisible();
    await expect(page.locator(".chat-msg.bot").first()).toBeVisible({ timeout: 30_000 });
    await expect(page.locator("text=/something went wrong/i")).toHaveCount(0);
  });

  test("typing a custom question and pressing Ask works once scoped", async ({ page }) => {
    const select = page.getByLabel("Scope to a property");
    const specific = await firstRealOption(select);
    await select.selectOption({ label: specific });

    const input = page.getByPlaceholder("e.g. What's the pet policy?");
    await input.fill("What's the monthly rent?");
    await page.locator(".chat-form").getByRole("button", { name: "Ask" }).click();
    await expect(page.locator(".chat-msg.user").first()).toContainText("monthly rent");
    await expect(page.locator(".chat-msg.bot").first()).toBeVisible({ timeout: 30_000 });
  });

  test("the placeholder option cannot be re-selected (no unscoped option exists)", async ({ page }) => {
    const select = page.getByLabel("Scope to a property");
    const specific = await firstRealOption(select);
    await select.selectOption({ label: specific });
    await expect(page.getByPlaceholder("e.g. What's the pet policy?")).toBeEnabled();
    // The placeholder is a disabled option, so there is no way back to "no property" via the select.
    await expect(select.locator("option", { hasText: "Select a property…" })).toBeDisabled();
  });
});
