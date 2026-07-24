import { test, expect, type Locator, type Page, type TestInfo } from "@playwright/test";
import { gotoHash } from "./helpers";

/**
 * Custom free-form chat stress across EVERY chat surface of RentReady, 20
 * DISTINCT questions each (our own wording — never a suggestion chip).
 *
 * Each test establishes context ONCE, then loops over 20 realistic questions on
 * the SAME page (no reload per question). For every question we record the
 * current count of finished bot messages, submit, then wait until a NEW bot
 * message appears with non-empty text and the typing indicator gone — asserting
 * a genuine, live-Claude answer (backend on :8100). Failures are recorded
 * per-question (not thrown mid-loop) so we get full 20/20 evidence, then the
 * test fails at the end if any question did not produce a genuine answer.
 */

// Live-Claude streaming answers run ~2-6s each; allow generous slack per answer.
const ANSWER_TIMEOUT = 90_000;

test.describe.configure({ mode: "serial" });

interface QResult {
  index: number;
  question: string;
  ok: boolean;
  answer: string;
  error?: string;
}

/** A genuine answer: non-empty and not one of the app's failure surfaces. */
function genuineProblem(answer: string): string | null {
  const a = answer.trim();
  if (a.length === 0) return "empty answer";
  if (/something went wrong/i.test(a)) return "app error boundary ('something went wrong')";
  if (/^\s*Sorry —/.test(a)) return "app 'Sorry —' failure state";
  if (/^\s*Error:/i.test(a)) return "app 'Error:' state";
  if (/Could not reach the server/i.test(a)) return "'Could not reach the server'";
  return null;
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

/**
 * Ask ONE question in `scope` and wait for a brand-new finished bot answer.
 * "Finished" = a `.chat-msg.bot` that no longer holds the `.chat-typing` dots.
 * Uses the count of finished bot messages before submit as the baseline, so we
 * only accept a message that appeared AFTER this question was sent.
 */
async function askOne(scope: Locator, question: string, index: number): Promise<QResult> {
  const finished = scope.locator(".chat-msg.bot:not(:has(.chat-typing))");
  try {
    const before = await finished.count();

    const input = scope.locator("form.chat-form input");
    await expect(input).toBeEnabled({ timeout: 15_000 });
    await input.fill(question);
    await scope.locator("form.chat-form button[type=submit]").click();

    // A NEW finished bot message must appear (count strictly increases).
    await expect
      .poll(async () => finished.count(), {
        timeout: ANSWER_TIMEOUT,
        message: `Q${index} produced no new bot answer (hang?)`,
      })
      .toBeGreaterThan(before);

    // ...and that newest message must carry real text (not stuck mid-stream).
    await expect
      .poll(async () => (await finished.last().innerText()).trim().length, {
        timeout: ANSWER_TIMEOUT,
        message: `Q${index} bot answer never acquired text`,
      })
      .toBeGreaterThan(0);

    const answer = (await finished.last().innerText()).trim();
    const problem = genuineProblem(answer);
    return { index, question, ok: problem === null, answer, error: problem ?? undefined };
  } catch (err) {
    return {
      index,
      question,
      ok: false,
      answer: "",
      error: `hang/timeout: ${(err as Error).message.split("\n")[0]}`,
    };
  }
}

/** Run the 20-question loop against `scope`, log a summary, attach evidence. */
async function runSurface(
  scope: Locator,
  surface: string,
  questions: string[],
  testInfo: TestInfo,
  // Stateful chats (e.g. the tours booking flow) reach a terminal state after a
  // few turns and disable the input. For those, pass a reset fn that returns a
  // FRESH chat scope — each question is then an independent first-turn inquiry.
  resetBeforeEach?: () => Promise<Locator>,
): Promise<void> {
  const results: QResult[] = [];
  for (let i = 0; i < questions.length; i++) {
    const s = resetBeforeEach ? await resetBeforeEach() : scope;
    results.push(await askOne(s, questions[i], i + 1));
  }

  const passed = results.filter((r) => r.ok);
  const failed = results.filter((r) => !r.ok);

  // eslint-disable-next-line no-console
  console.log(`\n===== ${surface}: ${passed.length}/${questions.length} answered =====`);
  for (const r of results) {
    const head = r.answer.replace(/\s+/g, " ").trim().slice(0, 120);
    const tag = r.ok ? "PASS" : "FAIL";
    // eslint-disable-next-line no-console
    console.log(`  [${tag}] Q${r.index}: ${r.question}`);
    if (r.ok) {
      // eslint-disable-next-line no-console
      console.log(`         A: ${head}`);
    } else {
      // eslint-disable-next-line no-console
      console.log(`         !! ${r.error}`);
    }
  }

  testInfo.annotations.push({
    type: `${surface} · score`,
    description: `${passed.length}/${questions.length} answered`,
  });
  for (const r of passed.slice(0, 4)) {
    testInfo.annotations.push({
      type: `${surface} · sample`,
      description: `${r.question} → ${r.answer.replace(/\s+/g, " ").slice(0, 120)}`,
    });
  }
  for (const r of failed) {
    testInfo.annotations.push({
      type: `${surface} · FAILURE`,
      description: `Q${r.index}: ${r.question} — ${r.error}`,
    });
  }

  expect(
    failed,
    `${surface}: ${failed.length} question(s) did not return a genuine answer — ${failed
      .map((r) => `Q${r.index} (${r.error})`)
      .join("; ")}`,
  ).toHaveLength(0);
}

// ---------------------------------------------------------------------------
// 1. Risk chat — #/risk auto-selects the top applicant.
// ---------------------------------------------------------------------------
const RISK_QS = [
  "What are the top factors driving this applicant's risk score?",
  "Why did the model flag this application the way it did?",
  "If their monthly income rose by $1,500, how would the risk change?",
  "Suppose their credit score dropped to 580 — what happens to the assessment?",
  "How much would paying down their debt lower the risk?",
  "What's the single most effective change they could make to improve their odds?",
  "How does this applicant compare to the rest of the portfolio?",
  "Is this applicant riskier or safer than the typical one you see?",
  "Which factors are explicitly excluded from the model by policy?",
  "How confident is the model in this particular prediction?",
  "What does this risk band actually mean in practice?",
  "Where does this applicant fall on the overall risk distribution?",
  "If their debt-to-income ratio were cut in half, would the band change?",
  "Which single piece of information moved the score the most?",
  "Are there fair-housing considerations the model deliberately avoids?",
  "What would it take to move them from high risk to medium?",
  "How would adding a co-signer affect this risk assessment?",
  "Does the length of their employment materially affect the score?",
  "If the rent were $200 cheaper, does affordability improve the outcome?",
  "Summarize this applicant's risk in one sentence for a leasing manager.",
];

test("Risk chat — 20 custom questions", async ({ page }, testInfo) => {
  test.setTimeout(420_000);
  await gotoHash(page, "#/risk");
  const card = cardWithHeading(page, "Ask about risk");
  await expect(card).toBeVisible({ timeout: 30_000 });
  await runSurface(card, "Risk chat", RISK_QS, testInfo);
});

// ---------------------------------------------------------------------------
// 2. Residents chat — #/residents portfolio scope shows the chat before any
//    property is picked.
// ---------------------------------------------------------------------------
const RESIDENTS_QS = [
  "Which residents are most likely to miss a payment next month?",
  "What's the delinquency outlook for the coming quarter?",
  "Over the next year, which properties trend worst?",
  "How often do late payments recur among current residents?",
  "How severe are the expected arrears in dollar terms?",
  "What's the total projected dollar value at risk this month?",
  "Which residents have the best chance of curing a late balance?",
  "How does retention look for the upcoming lease renewals?",
  "Which property is the healthiest right now?",
  "Rank the properties by overall resident risk.",
  "Compare the two weakest properties for me.",
  "Are there residents who are chronically late versus one-off?",
  "What renewal risk should I plan for this quarter?",
  "Which residents should the team reach out to first?",
  "What governance rules constrain these resident predictions?",
  "How many residents are currently in arrears?",
  "What's driving the risk at the worst-performing property?",
  "Is delinquency rising or falling across the portfolio?",
  "Which residents are safe bets to renew?",
  "Give me a one-paragraph portfolio health summary.",
];

test("Residents chat — 20 custom questions", async ({ page }, testInfo) => {
  test.setTimeout(420_000);
  await gotoHash(page, "#/residents");
  const card = cardWithHeading(page, "Ask about residents");
  await expect(card).toBeVisible({ timeout: 60_000 });
  await runSurface(card, "Residents chat", RESIDENTS_QS, testInfo);
});

// ---------------------------------------------------------------------------
// 3. Concierge (Ask page) — #/ask, requires a property scope first.
// ---------------------------------------------------------------------------
const CONCIERGE_QS = [
  "Which buildings have an on-site gym?",
  "What's the cheapest available unit right now?",
  "Do any of these properties allow large dogs?",
  "What is the late fee policy?",
  "Am I allowed to sublet my apartment?",
  "What's the pet deposit at these properties?",
  "How much notice do I need to give before moving out?",
  "What happens if I need to break my lease early?",
  "Which places have in-unit laundry?",
  "What's the average rent for a two-bedroom?",
  "Are utilities included in the rent anywhere?",
  "Which property has the largest floor plans?",
  "Is parking available, and does it cost extra?",
  "What's the security deposit typically?",
  "Do any of the units come furnished?",
  "Which pet-friendly places rent under $1,800?",
  "Compare the two most affordable properties.",
  "What amenities does the most expensive building offer?",
  "Are there any move-in specials or concessions right now?",
  "Which building is closest to downtown?",
];

test("Concierge (Ask) — 20 custom questions", async ({ page }, testInfo) => {
  test.setTimeout(420_000);
  await gotoHash(page, "#/ask");
  const select = page.getByLabel("Scope to a property");
  await expect(select).toBeVisible();
  await pickFirstOption(select);

  const card = page.locator(".ask-chat-card");
  await expect(card.getByPlaceholder("e.g. What's the pet policy?")).toBeEnabled();
  await runSurface(card, "Concierge (Ask)", CONCIERGE_QS, testInfo);
});

// ---------------------------------------------------------------------------
// 4. Applicant Q&A — "/" then load a sample; chat is "5. Ask about this
//    application" (renders only after extract→screen→recommend lands).
// ---------------------------------------------------------------------------
const APPLICANT_QS = [
  "What is the applicant's stated monthly income?",
  "How long have they been at their current job?",
  "What does their credit history look like?",
  "Did they provide references, and what do they say?",
  "Do they have any pets?",
  "When are they hoping to move in?",
  "What rent amount are they seeking?",
  "What was their previous address?",
  "How many people will be living in the unit?",
  "Is their employment full-time or part-time?",
  "What's their overall debt situation?",
  "Are there any red flags in this application?",
  "Does their income comfortably cover the rent they want?",
  "Have they rented before, and how did it go?",
  "What's the applicant's name and contact information?",
  "Is there a co-applicant or a guarantor?",
  "What's their stated reason for moving?",
  "Are there gaps in their rental or employment history?",
  "How complete is this application overall?",
  "Summarize this applicant in two sentences.",
];

test("Applicant Q&A — 20 custom questions", async ({ page }, testInfo) => {
  test.setTimeout(420_000);
  await page.goto("/");
  const chip = page.locator(".sample-chip").first();
  await expect(chip).toBeVisible({ timeout: 30_000 });
  await chip.click();

  const card = cardWithHeading(page, "5. Ask about this application");
  await expect(card).toBeVisible({ timeout: 180_000 });
  await runSurface(card, "Applicant Q&A", APPLICANT_QS, testInfo);
});

// ---------------------------------------------------------------------------
// 5. Tours scheduler — #/tours, pick a property, then chat via SchedulerChat.
// ---------------------------------------------------------------------------
const TOURS_QS = [
  "Can I tour this Saturday afternoon?",
  "Is a Monday morning slot available?",
  "What about next Tuesday around 10am?",
  "Do you have anything open this weekend?",
  "I'd like to visit Friday after 5pm — is that possible?",
  "Can we do a tour on Wednesday at noon?",
  "What times are open next week?",
  "Is Sunday an option at all?",
  "Could I come by tomorrow afternoon?",
  "Are there any evening slots on weekdays?",
  "I need to reschedule — what else is open?",
  "What's the earliest available tour?",
  "Can I book something about two weeks out?",
  "Do you offer weekend morning tours?",
  "Is Thursday at 3pm free?",
  "What other times do you have besides mornings?",
  "Can I tour next Monday around lunchtime?",
  "Are there any slots left this Friday?",
  "I prefer late afternoons — what fits?",
  "When is the soonest I could see the place?",
];

test("Tours scheduler — 20 custom questions", async ({ page }, testInfo) => {
  test.setTimeout(600_000);
  // Tours is a stateful booking conversation — a single thread progresses to a
  // terminal state (booked / no-availability) and disables the input. So treat
  // each of the 20 as an independent first-turn scheduling inquiry: reset to a
  // fresh conversation (re-navigate + re-pick the property) before every one.
  const freshTours = async (): Promise<Locator> => {
    // A hash-only URL change does NOT reload a loaded SPA, so SchedulerChat's
    // React state would persist across inquiries and reach its terminal
    // (input-disabled) booking state. Force a full remount with reload().
    await page.goto("/#/tours");
    await page.reload();
    await expect(page.getByRole("navigation")).toBeVisible();
    const select = page.getByRole("combobox").first();
    await expect(select).toBeVisible();
    await pickFirstOption(select);
    const card = cardWithHeading(page, "Schedule a tour");
    await expect(card).toBeVisible();
    return card;
  };
  await runSurface(page.locator("body"), "Tours scheduler", TOURS_QS, testInfo, freshTours);
});

// ---------------------------------------------------------------------------
// 6. Property-Graph — "/" after a sample loads, "6. Ask the property graph".
//    Neo4j may be OFFLINE (no input; static offline notice) or ONLINE (input
//    present). Detect which: type 20 questions if online, else assert the
//    safe-degradation notice and log the offline state (do NOT fail the suite
//    solely because Neo4j is down).
// ---------------------------------------------------------------------------
const GRAPH_QS = [
  "Which buildings have a gym?",
  "What are the cheapest two-bedroom units downtown?",
  "Show me pet-friendly properties under $2,000.",
  "Which properties have a pool?",
  "List units with in-unit laundry.",
  "Which buildings are within walking distance of downtown?",
  "Which properties allow cats?",
  "Find three-bedroom units under $3,000.",
  "Which buildings have covered parking?",
  "What's the most expensive property in the graph?",
  "Show properties with more than two bathrooms.",
  "Which units are available right now?",
  "Find pet-friendly buildings that also have a gym.",
  "What properties have a rooftop deck?",
  "Which two-bedrooms rent for less than $2,500?",
  "List buildings with elevator access.",
  "Which properties are closest to public transit?",
  "Find furnished units downtown.",
  "What's the largest available floor plan?",
  "Which buildings offer both parking and a gym?",
];

test("Property-Graph — 20 custom questions or safe offline degradation", async ({
  page,
}, testInfo) => {
  test.setTimeout(600_000);
  await page.goto("/");
  const chip = page.locator(".sample-chip").first();
  await expect(chip).toBeVisible({ timeout: 30_000 });
  await chip.click();

  const card = cardWithHeading(page, /Ask the property graph/);
  await expect(card).toBeVisible({ timeout: 180_000 });

  const input = card.locator("form.chat-form input");

  if ((await input.count()) === 0) {
    // OFFLINE: no input rendered — assert the safe-degradation notice, no crash.
    const alert = card.locator(".alert.bad");
    await expect(alert).toBeVisible();
    const msg = (await alert.innerText()).trim();
    // eslint-disable-next-line no-console
    console.log(`\n===== Property-Graph: GRAPH OFFLINE — 0/20 typed (Neo4j down) =====`);
    // eslint-disable-next-line no-console
    console.log(`  offline notice: ${msg.replace(/\s+/g, " ").slice(0, 160)}`);
    testInfo.annotations.push({
      type: "Property-Graph · score",
      description: "GRAPH OFFLINE — 0/20 typed (Neo4j down)",
    });
    expect(msg).toMatch(/offline|not available|connect Neo4j/i);
    await expect(page.locator("text=/something went wrong/i")).toHaveCount(0);
  } else {
    // ONLINE. GraphAsk is a single Q→A box (NOT an appending chat log): each
    // query clears then refills ONE `.chat-msg.bot`, and the "Query" button
    // disables while busy. So drive it as fill → click → wait for the button to
    // re-enable → read the (replaced) answer — the generic new-message loop
    // never sees a "new" message here.
    // eslint-disable-next-line no-console
    console.log("\n===== Property-Graph: ONLINE — typing 20 questions =====");
    const submit = card.locator("form.chat-form button[type='submit']");
    const answerBox = card.locator(".chat-msg.bot").last();
    const results: QResult[] = [];
    for (let i = 0; i < GRAPH_QS.length; i++) {
      const q = GRAPH_QS[i];
      try {
        await expect(submit).toBeEnabled({ timeout: 60_000 });
        await input.fill(q);
        await submit.click();
        // busy → disabled (may flip too fast to observe) → re-enabled when done.
        await expect(submit).toBeDisabled({ timeout: 8_000 }).catch(() => {});
        await expect(submit).toBeEnabled({ timeout: 150_000 });
        await expect(answerBox).toBeVisible({ timeout: 10_000 });
        const answer = (await answerBox.innerText()).trim();
        const bad =
          answer.length < 3 ||
          /error:|couldn'?t run|not available|set anthropic/i.test(answer);
        results.push({ index: i + 1, question: q, ok: !bad, answer, error: bad ? "graph answer not genuine" : undefined });
      } catch (err) {
        results.push({ index: i + 1, question: q, ok: false, answer: "", error: `hang/timeout: ${(err as Error).message.split("\n")[0]}` });
      }
    }
    const passed = results.filter((r) => r.ok);
    const failed = results.filter((r) => !r.ok);
    // eslint-disable-next-line no-console
    console.log(`\n===== Property-Graph: ${passed.length}/${GRAPH_QS.length} answered =====`);
    for (const r of results) {
      // eslint-disable-next-line no-console
      console.log(`  [${r.ok ? "PASS" : "FAIL"}] Q${r.index}: ${r.question}`);
      // eslint-disable-next-line no-console
      if (r.ok) console.log(`         A: ${r.answer.replace(/\s+/g, " ").slice(0, 120)}`);
      else console.log(`         !! ${r.error}`);
    }
    testInfo.annotations.push({ type: "Property-Graph · score", description: `${passed.length}/${GRAPH_QS.length} answered` });
    for (const r of passed.slice(0, 4)) {
      testInfo.annotations.push({ type: "Property-Graph · sample", description: `${r.question} → ${r.answer.replace(/\s+/g, " ").slice(0, 120)}` });
    }
    expect(
      failed,
      `Property-Graph: ${failed.length} question(s) not genuine — ${failed.map((r) => `Q${r.index} (${r.error})`).join("; ")}`,
    ).toHaveLength(0);
  }
});
