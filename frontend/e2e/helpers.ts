import { expect, type Page } from "@playwright/test";

/** Open a view by its URL hash and wait for the app shell (nav) to be ready. */
export async function gotoHash(page: Page, hash: string): Promise<void> {
  await page.goto(`/${hash}`);
  await expect(page.getByRole("navigation")).toBeVisible();
}

/** The top nav is always present; use it as the app-loaded signal. */
export async function expectShell(page: Page): Promise<void> {
  await expect(page.getByText("RentReady", { exact: true }).first()).toBeVisible();
}

/** Click a top-nav tab by its visible label. */
export async function clickNav(page: Page, label: string): Promise<void> {
  await page.getByRole("button", { name: label, exact: true }).first().click();
}
