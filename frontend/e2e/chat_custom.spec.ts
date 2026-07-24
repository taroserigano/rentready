import { test, expect, type Locator, type Page, type TestInfo } from "@playwright/test";
import { gotoHash } from "./helpers";

/**
 * Custom free-form chat smoke across EVERY chat surface of RentReady.
 *
 * For each surface we PHYSICALLY TYPE a question of our own wording (never a
 * pre-baked suggestion chip), submit it, and assert a genuine, non-empty
 * assistant answer comes back — the typing indicator clears and the text is
 * not the app's error / "something went wrong" state. The answer text is
 * echoed to the console and attached to the test as evidence the chat really
 * responded against the live backend (real Claude on :8100).
 */

// The e2e backend answers with real Claude, so allow generous end-to-end time.
const ANSWER_TIMEOUT = 60_000;

test.describe.configure({ mode: "serial" });

/** Record the answer as console output + a test annotation (evidence). */
function evidence(testInfo: TestInfo, surface: string, question: string, answer: string): void {
  const head = answer.replace(/\s+/g, " ").trim().slice(0, 200);
  // eslint-disable-next-line no-console
  console.log(`\n[${surface}]\n  Q: ${question}\n  A: ${head}\n`);
  testInfo.annotations.push({ type: `${surface} · question`, description: question });
  testInfo.annotations.push({ type: `${surface} · answer`, description: head });
}

/** A genuine answer: non-empty and not one of the app's failure surfaces. */
function assertGenuine(answer: string): void {
  expect(answer.trim().length, "answer should be non-empty").toBeGreaterThan(0);
  expect(answer).not.toMatch(/something went wrong/i);
  expect(answer).not.toMatch(/^\s*Sorry —/);
  expect(answer).not.toMatch(/^\s*Error:/i);
  expect(answer).not.toMatch(/Could not reach the server/i);
}

/**
 * Wait until a NEW assistant message inside `scope` has finished: a
 * `.chat-msg.bot` that no longer contains the `.chat-typing` dots and carries
 * real text. Returns the answer text.
 */
async function waitForAnswer(scope: Locator, timeout = ANSWER_TIMEOUT): Promise<string> {
  const answered = scope.locator(".chat-msg.bot:not(:has(.chat-typing))");
  await expect(answered.last()).toBeVisible({ timeout });
  await expect
    .poll(async () => (await answered.last().innerText()).trim().length, {
      timeout,
      message: "assistant message should acquire real text (not stuck typing)",
    })
    .toBeGreaterThan(0);
  return (await answered.last().innerText()).trim();
}

/** Type into a card's chat input and submit via its send button. */
async function typeAndSend(card: Locator, question: string): Promise<void> {
  const input = card.locator("form.chat-form input");
  await expect(input).toBeEnabled({ timeout: 15_000 });
  await input.fill(question);
  await card.locator("form.chat-form button[type=submit]").click();
}

/** The `.card` that contains a given heading. */
function cardWithHeading(page: Page, name: string | RegExp): Locator {
  return page.locator(".card").filter({ has: page.getByRole("heading", { name }) });
}

/** Pick the first real (non-placeholder) option of a <select>. */
async function pickFirstOption(select: Locator): Promise<void> {
  await expect(select.locator("option").nth(1)).toBeAttached({ timeout: 15_000 });
  await select.selectOption({ index: 1 });
}

// 1. Risk chat — #/risk auto-selects the top applicant.
test("Risk chat responds to a custom question", async ({ page }, testInfo) => {
  test.setTimeout(180_000);
  await gotoHash(page, "#/risk");
  const card = cardWithHeading(page, "Ask about risk");
  await expect(card).toBeVisible({ timeout: 30_000 });

  const q = "If their credit score were 720, how would that change things?";
  await typeAndSend(card, q);
  const answer = await waitForAnswer(card);
  evidence(testInfo, "Risk chat", q, answer);
  assertGenuine(answer);
});

// 2. Residents chat — #/residents portfolio scope shows the chat before any
//    property is picked.
test("Residents chat responds to a custom question", async ({ page }, testInfo) => {
  test.setTimeout(180_000);
  await gotoHash(page, "#/residents");
  const card = cardWithHeading(page, "Ask about residents");
  await expect(card).toBeVisible({ timeout: 60_000 });

  const q = "Which property should I worry about most and why?";
  await typeAndSend(card, q);
  const answer = await waitForAnswer(card);
  evidence(testInfo, "Residents chat", q, answer);
  assertGenuine(answer);
});

// 3. Concierge (Ask page) — #/ask, requires a property scope first.
test("Concierge (Ask) responds to a custom question", async ({ page }, testInfo) => {
  test.setTimeout(180_000);
  await gotoHash(page, "#/ask");
  const select = page.getByLabel("Scope to a property");
  await expect(select).toBeVisible();
  await pickFirstOption(select);

  const card = page.locator(".ask-chat-card");
  await expect(card.getByPlaceholder("e.g. What's the pet policy?")).toBeEnabled();

  const q = "Which pet-friendly places rent under $2000?";
  await typeAndSend(card, q);
  const answer = await waitForAnswer(card);
  evidence(testInfo, "Concierge (Ask)", q, answer);
  assertGenuine(answer);
});

// 4. Applicant Q&A — "/" then load a sample; chat is "5. Ask about this application".
test("Applicant Q&A responds to a custom question", async ({ page }, testInfo) => {
  test.setTimeout(240_000);
  await page.goto("/");
  const chip = page.locator(".sample-chip").first();
  await expect(chip).toBeVisible({ timeout: 30_000 });
  await chip.click();

  // The Q&A chat only renders once the extract→screen→recommend pipeline lands.
  const card = cardWithHeading(page, "5. Ask about this application");
  await expect(card).toBeVisible({ timeout: 180_000 });

  const q = "How stable is their employment situation?";
  await typeAndSend(card, q);
  const answer = await waitForAnswer(card);
  evidence(testInfo, "Applicant Q&A", q, answer);
  assertGenuine(answer);
});

// 5. Property-Graph — same "/" page after a sample loads, "6. Ask the property graph".
//    Neo4j is OFF in e2e; the surface degrades safely. NOTE: when offline the
//    component renders NO text input at all (only a static offline alert), so
//    free-form typing is physically impossible here in this environment — the
//    test asserts the safe-degradation message (a response, no crash) instead,
//    and types only if an input is actually present (Neo4j online).
test("Property-Graph degrades safely / responds to a custom question", async ({ page }, testInfo) => {
  test.setTimeout(240_000);
  await page.goto("/");
  const chip = page.locator(".sample-chip").first();
  await expect(chip).toBeVisible({ timeout: 30_000 });
  await chip.click();

  const card = cardWithHeading(page, /Ask the property graph/);
  await expect(card).toBeVisible({ timeout: 180_000 });

  const input = card.locator("form.chat-form input");
  const q = "Which downtown buildings have a gym?";

  if ((await input.count()) === 0) {
    // Offline: no input rendered — assert the safe degradation message shows.
    const alert = card.locator(".alert.bad");
    await expect(alert).toBeVisible();
    const msg = (await alert.innerText()).trim();
    evidence(testInfo, "Property-Graph (offline)", q, msg);
    expect(msg).toMatch(/offline|not available|connect Neo4j/i);
    // No client crash surfaced.
    await expect(page.locator("text=/something went wrong/i")).toHaveCount(0);
  } else {
    // Online: type a real question and assert a genuine answer.
    await input.fill(q);
    await card.locator("form.chat-form button[type=submit]").click();
    const answer = await waitForAnswer(card);
    evidence(testInfo, "Property-Graph", q, answer);
    assertGenuine(answer);
  }
});

// 6. Tours scheduler — #/tours, pick a property, then chat via SchedulerChat.
test("Tours scheduler responds to a custom question", async ({ page }, testInfo) => {
  test.setTimeout(180_000);
  await gotoHash(page, "#/tours");

  // PropertySelector is a plain <select>; pick the first real property.
  const select = page.getByRole("combobox").first();
  await expect(select).toBeVisible();
  await pickFirstOption(select);

  const card = cardWithHeading(page, "Schedule a tour");
  await expect(card).toBeVisible();

  const q = "Can I tour this Saturday afternoon?";
  await typeAndSend(card, q);
  const answer = await waitForAnswer(card);
  evidence(testInfo, "Tours scheduler", q, answer);
  assertGenuine(answer);
});
