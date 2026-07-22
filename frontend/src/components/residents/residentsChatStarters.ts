import type { ResidentChatScope } from "../../types";

/**
 * Suggested opening questions for the residents chat, seeded when the thread is
 * empty and again after a context switch. Pure — no I/O. Resident scope is
 * personalized by `name`; property/portfolio scope lean on the health ranking.
 */
export function residentStarters(
  scope: ResidentChatScope,
  name?: string,
  propertyName?: string,
): string[] {
  if (scope === "resident") {
    const who = name && name.trim() ? name.trim() : "this resident";
    return [
      `How likely is ${who} to pay late next year?`,
      "How often will they be late?",
      "How severe could it get?",
      "Will they renew?",
      "Why are they flagged?",
    ];
  }

  if (scope === "property") {
    const where = propertyName && propertyName.trim() ? propertyName.trim() : "this property";
    return [
      `How healthy is ${where}?`,
      "Which residents need outreach first?",
      "What's driving arrears here?",
      "Which apartments are healthiest?",
    ];
  }

  return [
    "Which apartments are healthiest?",
    "Which property needs attention?",
    "How is late-payment risk spread across the portfolio?",
    "What does the model measure, and what does it exclude?",
  ];
}
