import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { EligibilityCard } from "../components/EligibilityCard";
import { Recommendations } from "../components/Recommendations";
import { ProfileCard } from "../components/ProfileCard";
import { Badge } from "../components/Badge";
import { Markdown } from "../components/Markdown";
import { Stepper, SkeletonCard } from "../components/Loading";
import { RiskChatMessage } from "../components/risk/RiskChatMessage";
import type { RiskChatMsg } from "../components/risk/RiskChatMessage";
import { ResidentFamilies } from "../components/residents/ResidentFamilies";
import {
  ArrearsPredictionCard,
  ChurnPredictionCard,
  LatePredictionCard,
  SeriousPredictionCard,
} from "../components/residents/PredictionCards";
import type {
  ApplicantProfile,
  EligibilityResult,
  RecommendResponse,
  PropertyRecommendation,
  ReasonCode,
  ResidentArrearsPrediction,
  ResidentChurnPrediction,
  ResidentClassPrediction,
  ResidentHead,
  ResidentSeriousPrediction,
  RiskChatAnswer,
} from "../types";

// --- shared fixtures --------------------------------------------------------

const REASONS: ReasonCode[] = [
  { feature: "dti", label: "High debt-to-income", direction: "increases", contribution: 0.4 },
  { feature: "streak", label: "Long on-time streak", direction: "decreases", contribution: -0.2 },
];

function rec(overrides: Partial<PropertyRecommendation> = {}): PropertyRecommendation {
  return {
    property_id: "PROP-1",
    name: "Maple Court",
    area: "Downtown",
    property_type: "Apartment",
    monthly_rent: 1500,
    bedrooms: 2,
    bathrooms: 1,
    bathroom_type: "full",
    square_feet: 800,
    has_balcony: false,
    in_unit_laundry: false,
    pets_allowed: true,
    parking_type: "",
    walk_score: null,
    transit_score: null,
    amenities: [],
    match_reason: "close to work",
    fit_highlights: [],
    score: 0.75,
    signal_breakdown: { area: 0.9 },
    ...overrides,
  };
}

// --- EligibilityCard edge states -------------------------------------------

describe("EligibilityCard edge states", () => {
  it("renders the needs_review verdict label", () => {
    const result: EligibilityResult = {
      verdict: "needs_review",
      reasons: ["Borderline ratio."],
      income_to_rent_ratio: 2.9,
      explanation: "",
    };
    render(<EligibilityCard result={result} />);
    expect(screen.getByText("Needs review")).toBeInTheDocument();
    // Empty explanation is not rendered as a paragraph.
    expect(screen.queryByText("", { selector: "p" })).not.toBeInTheDocument();
  });

  it("lists every reason and shows the explanation when present", () => {
    const result: EligibilityResult = {
      verdict: "qualified",
      reasons: ["Reason A", "Reason B"],
      income_to_rent_ratio: 3.5,
      explanation: "Looks strong.",
    };
    render(<EligibilityCard result={result} />);
    expect(screen.getByText("Reason A")).toBeInTheDocument();
    expect(screen.getByText("Reason B")).toBeInTheDocument();
    expect(screen.getByText("Looks strong.")).toBeInTheDocument();
  });
});

// --- Recommendations edge states -------------------------------------------

describe("Recommendations edge states", () => {
  it("shows the rent-burden chip when income is provided", () => {
    const data: RecommendResponse = {
      source: "scorer",
      graph_backend: "memory",
      relaxed: false,
      recommendations: [rec({ monthly_rent: 1500 })],
    };
    render(<Recommendations data={data} monthlyIncome={5000} />);
    // 1500 / 5000 = 30% => "good" chip
    expect(screen.getByText(/30% of income/)).toBeInTheDocument();
  });

  it("fires onViewListing with the property id", () => {
    const onView = vi.fn();
    const data: RecommendResponse = {
      source: "scorer",
      graph_backend: "memory",
      relaxed: false,
      recommendations: [rec({ property_id: "PROP-42" })],
    };
    render(<Recommendations data={data} onViewListing={onView} />);
    fireEvent.click(screen.getByText(/View listing/));
    expect(onView).toHaveBeenCalledWith("PROP-42");
  });

  it("notes a relaxed budget in the sub-line", () => {
    const data: RecommendResponse = {
      source: "scorer",
      graph_backend: "neo4j",
      relaxed: true,
      recommendations: [rec()],
    };
    render(<Recommendations data={data} />);
    expect(screen.getByText(/budget relaxed to show options/)).toBeInTheDocument();
  });

  it("renders fit highlights when present", () => {
    const data: RecommendResponse = {
      source: "scorer",
      graph_backend: "memory",
      relaxed: false,
      recommendations: [rec({ fit_highlights: ["Pet friendly", "Great walkability"] })],
    };
    render(<Recommendations data={data} />);
    expect(screen.getByText("Pet friendly")).toBeInTheDocument();
    expect(screen.getByText("Great walkability")).toBeInTheDocument();
  });
});

// --- ProfileCard edge states -----------------------------------------------

describe("ProfileCard edge states", () => {
  const base: ApplicantProfile = {
    name: "Sam Doe",
    monthly_income: 4000,
    desired_rent: 1200,
    credit_score: null,
    employment_status: "",
    bedrooms_wanted: null,
    bathrooms_wanted: null,
    bath_type_wanted: "",
    min_square_feet: null,
    has_pets: false,
    needs_balcony: false,
    needs_parking: false,
    needs_in_unit_laundry: false,
    furnished_wanted: false,
    preferred_area: "",
    wanted_amenities: [],
    lease_term_wanted: null,
  };

  it("falls back to em dashes for missing optional fields", () => {
    render(<ProfileCard profile={base} chunks={0} />);
    // credit score null => em dash present at least once
    expect(screen.getAllByText("—").length).toBeGreaterThan(0);
    expect(screen.getByText(/Indexed 0 chunk/)).toBeInTheDocument();
  });

  it("renders history badges for flagged fields", () => {
    render(
      <ProfileCard
        profile={{
          ...base,
          evictions_count: 1,
          late_payments_12mo: 2,
          landlord_reference: true,
          smoker: true,
        }}
        chunks={5}
      />,
    );
    expect(screen.getByText("1 eviction(s)")).toBeInTheDocument();
    expect(screen.getByText("2 late payment(s) in 12 mo")).toBeInTheDocument();
    expect(screen.getByText("Landlord reference")).toBeInTheDocument();
    expect(screen.getByText("Smoker")).toBeInTheDocument();
  });

  it("shows 'Unknown' in the header when the name is blank", () => {
    render(<ProfileCard profile={{ ...base, name: "" }} chunks={1} />);
    expect(screen.getByText("Unknown")).toBeInTheDocument();
  });
});

// --- Badge ------------------------------------------------------------------

describe("Badge", () => {
  it("shows a connected state with a good dot", () => {
    const { container } = render(<Badge on label="Neo4j" />);
    expect(screen.getByText("Neo4j")).toBeInTheDocument();
    expect(screen.getByText("connected")).toBeInTheDocument();
    expect(container.querySelector(".dot.good")).toBeTruthy();
    expect(screen.getByTitle("Neo4j is connected")).toBeTruthy();
  });

  it("shows a not-connected state with a bad dot", () => {
    const { container } = render(<Badge on={false} label="Phoenix" />);
    expect(screen.getByText("not connected")).toBeInTheDocument();
    expect(container.querySelector(".dot.bad")).toBeTruthy();
    expect(screen.getByTitle("Phoenix is not connected")).toBeTruthy();
  });

  it("passes a tone through as data-tone", () => {
    const { container } = render(<Badge on label="X" tone="teal" />);
    expect(container.querySelector('[data-tone="teal"]')).toBeTruthy();
  });
});

// --- Markdown ---------------------------------------------------------------

describe("Markdown", () => {
  it("renders nothing for empty text", () => {
    const { container } = render(<Markdown text="" />);
    expect(container.innerHTML).toBe("");
  });

  it("renders headings at demoted levels (# -> h3)", () => {
    render(<Markdown text={"# Title\n## Sub\n### Deep"} />);
    expect(screen.getByRole("heading", { level: 3 })).toHaveTextContent("Title");
    expect(screen.getByRole("heading", { level: 4 })).toHaveTextContent("Sub");
    expect(screen.getByRole("heading", { level: 5 })).toHaveTextContent("Deep");
  });

  it("renders bold, italic and inline code", () => {
    const { container } = render(<Markdown text={"**bold** _em_ `code`"} />);
    expect(container.querySelector("strong")).toHaveTextContent("bold");
    expect(container.querySelector("em")).toHaveTextContent("em");
    expect(container.querySelector("code")).toHaveTextContent("code");
  });

  it("renders unordered and ordered lists", () => {
    const { container } = render(<Markdown text={"- a\n- b"} />);
    expect(container.querySelectorAll("ul li")).toHaveLength(2);
    const ol = render(<Markdown text={"1. one\n2. two"} />);
    expect(ol.container.querySelectorAll("ol li")).toHaveLength(2);
  });

  it("renders a blockquote", () => {
    const { container } = render(<Markdown text={"> quoted"} />);
    expect(container.querySelector("blockquote")).toHaveTextContent("quoted");
  });

  it("joins plain paragraph lines into one paragraph", () => {
    const { container } = render(<Markdown text={"line one\nline two"} />);
    const ps = container.querySelectorAll("p.md-p");
    expect(ps).toHaveLength(1);
    expect(ps[0]).toHaveTextContent("line one line two");
  });
});

// --- Loading: Stepper + SkeletonCard ---------------------------------------

describe("Stepper", () => {
  const ready = { profile: false, eligibility: false, recs: false };

  it("renders nothing when idle or done", () => {
    const { container } = render(<Stepper phase="idle" ready={ready} />);
    expect(container.innerHTML).toBe("");
    const done = render(<Stepper phase="done" ready={ready} />);
    expect(done.container.innerHTML).toBe("");
  });

  it("marks the extract step active while extracting", () => {
    render(<Stepper phase="extracting" ready={ready} />);
    const active = screen.getByText("Extract profile").closest(".step");
    expect(active?.className).toContain("active");
    expect(screen.getByText("Extract profile")).toBeInTheDocument();
    expect(screen.getByText("Check eligibility")).toBeInTheDocument();
    expect(screen.getByText("Match properties")).toBeInTheDocument();
  });

  it("marks completed steps done", () => {
    render(
      <Stepper phase="screening" ready={{ profile: true, eligibility: false, recs: false }} />,
    );
    const step = screen.getByText("Extract profile").closest(".step");
    expect(step?.className).toContain("done");
  });
});

describe("SkeletonCard", () => {
  it("renders the title and the requested number of shimmer lines", () => {
    const { container } = render(<SkeletonCard title="Loading risk" lines={4} />);
    expect(screen.getByText("Loading risk")).toBeInTheDocument();
    expect(container.querySelectorAll(".skel-line")).toHaveLength(4);
    expect(container.querySelector(".skel-block")).toBeFalsy();
  });

  it("renders a block placeholder when block is set", () => {
    const { container } = render(<SkeletonCard title="X" block={120} />);
    expect(container.querySelector(".skel-block")).toBeTruthy();
    // default line count is 3
    expect(container.querySelectorAll(".skel-line")).toHaveLength(3);
  });
});

// --- RiskChatMessage --------------------------------------------------------

function answer(overrides: Partial<RiskChatAnswer> = {}): RiskChatAnswer {
  return {
    scope: "applicant",
    applicant_id: "A1",
    intent: "explain",
    follow_ups: [],
    artifact: { kind: "none" },
    answer: "Here is the explanation.",
    source: "anthropic",
    ...overrides,
  };
}

describe("RiskChatMessage", () => {
  const noop = () => {};

  it("shows only the typing indicator while pending with no text", () => {
    const msg: RiskChatMsg = { id: 1, who: "bot", text: "", pending: true };
    const { container } = render(<RiskChatMessage msg={msg} onFollowUp={noop} />);
    expect(container.querySelector(".chat-typing")).toBeTruthy();
    expect(screen.getByLabelText("Assistant is thinking")).toBeInTheDocument();
  });

  it("renders the intent badge label", () => {
    const msg: RiskChatMsg = {
      id: 2,
      who: "bot",
      text: "Explanation text",
      res: answer({ intent: "explain" }),
    };
    render(<RiskChatMessage msg={msg} onFollowUp={noop} />);
    expect(screen.getByText("Explanation")).toBeInTheDocument();
  });

  it("shows the exploratory badge for what-if / counterfactual intents", () => {
    const msg: RiskChatMsg = {
      id: 3,
      who: "bot",
      text: "What-if result",
      res: answer({ intent: "whatif" }),
    };
    render(<RiskChatMessage msg={msg} onFollowUp={noop} />);
    expect(screen.getByText("What-if")).toBeInTheDocument();
    expect(screen.getByText("Exploratory — not the saved score")).toBeInTheDocument();
  });

  it("shows an offline badge when the answer came from rules", () => {
    const msg: RiskChatMsg = {
      id: 4,
      who: "bot",
      text: "Offline answer",
      res: answer({ source: "rules", intent: "general" }),
    };
    render(<RiskChatMessage msg={msg} onFollowUp={noop} />);
    expect(screen.getByText("Offline")).toBeInTheDocument();
    expect(screen.getByText("General")).toBeInTheDocument();
  });

  it("renders follow-up chips and fires onFollowUp on click", () => {
    const onFollowUp = vi.fn();
    const msg: RiskChatMsg = {
      id: 5,
      who: "bot",
      text: "Answer",
      res: answer({ follow_ups: ["What lowers this?", "Compare to peers"] }),
    };
    render(<RiskChatMessage msg={msg} onFollowUp={onFollowUp} />);
    const chip = screen.getByText("What lowers this?");
    fireEvent.click(chip);
    expect(onFollowUp).toHaveBeenCalledWith("What lowers this?");
    expect(screen.getByText("Compare to peers")).toBeInTheDocument();
  });

  it("maps exclusions intent to a governance label", () => {
    const msg: RiskChatMsg = {
      id: 6,
      who: "bot",
      text: "Governance",
      res: answer({ intent: "exclusions" }),
    };
    render(<RiskChatMessage msg={msg} onFollowUp={noop} />);
    expect(screen.getByText("Model governance")).toBeInTheDocument();
  });
});

// --- ResidentFamilies -------------------------------------------------------

describe("ResidentFamilies", () => {
  it("renders nothing when no families have heads", () => {
    const { container } = render(<ResidentFamilies heads={{}} families={{}} />);
    expect(container.innerHTML).toBe("");
  });

  it("renders family headings and formats head values", () => {
    const heads: Record<string, ResidentHead> = {
      late_12m: {
        kind: "binary",
        family: "late",
        probability: 0.25,
        band: "medium",
        range: [0.18, 0.33],
        reason_codes: REASONS,
        confidence: "high",
        source: "model",
      },
      late_count_12m: {
        kind: "count",
        family: "frequency",
        expected: 3,
        interval: [1, 5],
        reason_codes: [],
        confidence: "low",
        source: "heuristic",
      },
      arrears_3m: {
        kind: "regression",
        family: "arrears",
        expected: 1200,
        interval: [800, 1600],
        reason_codes: [],
        confidence: "high",
        source: "model",
      },
    };
    const families = {
      late: ["late_12m"],
      frequency: ["late_count_12m"],
      arrears: ["arrears_3m"],
    };
    render(<ResidentFamilies heads={heads} families={families} />);
    expect(screen.getByText("Late-payment outlook")).toBeInTheDocument();
    expect(screen.getByText("How often")).toBeInTheDocument();
    expect(screen.getByText("Arrears ($)")).toBeInTheDocument();
    // count head expected value + money-formatted arrears
    expect(screen.getByText("3")).toBeInTheDocument();
    expect(screen.getByText("$1,200")).toBeInTheDocument();
    expect(screen.getByText(/interval \$800 – \$1,600/)).toBeInTheDocument();
  });

  it("renders a multiclass head with per-label bars", () => {
    const heads: Record<string, ResidentHead> = {
      delinquency_bucket_12m: {
        kind: "multiclass",
        family: "severity",
        class_probs: { none: 0.6, "30d": 0.3, "60d": 0.1 },
        predicted_bucket: "none",
        reason_codes: [],
        confidence: "high",
        source: "model",
      },
    };
    render(
      <ResidentFamilies heads={heads} families={{ severity: ["delinquency_bucket_12m"] }} />,
    );
    expect(screen.getByText("How severe")).toBeInTheDocument();
    expect(screen.getByText("60%")).toBeInTheDocument();
    expect(screen.getByText("30%")).toBeInTheDocument();
  });
});

// --- PredictionCards --------------------------------------------------------

describe("PredictionCards", () => {
  const classPred: ResidentClassPrediction = {
    probability: 0.25,
    band: "medium",
    range: [0.18, 0.33],
    reason_codes: REASONS,
    confidence: "high",
    source: "model",
    model_type: "xgboost",
  };

  it("LatePredictionCard shows the band label and calibration range", () => {
    render(<LatePredictionCard pred={classPred} />);
    expect(screen.getByText("Late payment — next quarter")).toBeInTheDocument();
    expect(screen.getByText(/Late risk: ~25% \(18–33%\)/)).toBeInTheDocument();
    expect(screen.getByText("Moderate")).toBeInTheDocument();
    expect(screen.getByText("High confidence")).toBeInTheDocument();
    expect(screen.getByText("xgboost")).toBeInTheDocument();
  });

  it("SeriousPredictionCard always shows a routes-to-review badge", () => {
    const serious: ResidentSeriousPrediction = { ...classPred, routes_to_review: true };
    render(<SeriousPredictionCard pred={serious} />);
    expect(screen.getByText(/Routes to review/)).toBeInTheDocument();
  });

  it("ArrearsPredictionCard formats the expected balance and interval", () => {
    const arrears: ResidentArrearsPrediction = {
      expected_balance: 950,
      interval: [400, 1500],
      reason_codes: [],
      confidence: "low",
      source: "heuristic",
    };
    render(<ArrearsPredictionCard pred={arrears} />);
    expect(screen.getByText("$950")).toBeInTheDocument();
    expect(screen.getByText(/interval \$400 – \$1,500/)).toBeInTheDocument();
    expect(screen.getByText("Low confidence")).toBeInTheDocument();
    expect(screen.getByText("heuristic")).toBeInTheDocument();
  });

  it("ChurnPredictionCard renders the not-applicable notice out of window", () => {
    const churn: ResidentChurnPrediction = {
      probability: null,
      band: "not_applicable",
      months_to_lease_end: 11,
      reason_codes: [],
      confidence: "low",
      source: "heuristic",
    };
    render(<ChurnPredictionCard pred={churn} />);
    expect(screen.getByText("Not applicable")).toBeInTheDocument();
    expect(screen.getByText(/lease is not ending within 6 months/)).toBeInTheDocument();
    expect(screen.getByText(/about 11 months to lease end/)).toBeInTheDocument();
  });

  it("ChurnPredictionCard renders the gauge view when in the renewal window", () => {
    const churn: ResidentChurnPrediction = {
      probability: 0.4,
      band: "high",
      months_to_lease_end: 3,
      reason_codes: REASONS,
      confidence: "high",
      source: "model",
    };
    render(<ChurnPredictionCard pred={churn} />);
    expect(screen.getByText("Elevated")).toBeInTheDocument();
    expect(screen.getByText(/Non-renewal: 40%/)).toBeInTheDocument();
    expect(screen.getByText(/3 mo to lease end/)).toBeInTheDocument();
  });
});
