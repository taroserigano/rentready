import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { EligibilityCard } from "../components/EligibilityCard";
import { Recommendations } from "../components/Recommendations";
import { ProfileCard } from "../components/ProfileCard";
import type {
  ApplicantProfile,
  EligibilityResult,
  RecommendResponse,
} from "../types";

describe("EligibilityCard", () => {
  it("shows a friendly label for the verdict", () => {
    const result: EligibilityResult = {
      verdict: "qualified",
      reasons: ["Income is 3.4x rent (needs 3.0x)."],
      income_to_rent_ratio: 3.4,
      explanation: "You look good!",
    };
    render(<EligibilityCard result={result} />);
    expect(screen.getByText("Qualified")).toBeInTheDocument();
    expect(screen.getByText(/3.4×/)).toBeInTheDocument();
  });

  it("renders the not-qualified state", () => {
    const result: EligibilityResult = {
      verdict: "not_qualified",
      reasons: ["Income is only 2.1x rent (needs 3.0x)."],
      income_to_rent_ratio: 2.1,
      explanation: "",
    };
    render(<EligibilityCard result={result} />);
    expect(screen.getByText("Not qualified")).toBeInTheDocument();
  });
});

describe("Recommendations", () => {
  it("lists matched properties, score and highlights", () => {
    const data: RecommendResponse = {
      source: "scorer",
      graph_backend: "neo4j",
      relaxed: false,
      recommendations: [
        {
          property_id: "PROP-004",
          name: "The Heights",
          area: "South Congress",
          property_type: "Apartment",
          monthly_rent: 1300,
          bedrooms: 1,
          bathrooms: 1,
          bathroom_type: "full",
          square_feet: 600,
          has_balcony: true,
          in_unit_laundry: false,
          pets_allowed: true,
          parking_type: "covered",
          walk_score: 88,
          transit_score: 72,
          amenities: ["Bike Storage"],
          match_reason: "in your preferred area",
          fit_highlights: ["In your preferred area"],
          score: 0.9,
          signal_breakdown: { area: 1.0 },
        },
      ],
    };
    render(<Recommendations data={data} />);
    expect(screen.getByText("The Heights")).toBeInTheDocument();
    expect(screen.getByText(/graph: neo4j/)).toBeInTheDocument();
    expect(screen.getByText(/90% match/)).toBeInTheDocument();
  });

  it("handles an empty recommendation list", () => {
    const data: RecommendResponse = {
      source: "none",
      graph_backend: "memory",
      relaxed: false,
      recommendations: [],
    };
    render(<Recommendations data={data} />);
    expect(screen.getByText(/No properties matched/)).toBeInTheDocument();
  });
});

describe("ProfileCard", () => {
  it("renders extracted fields", () => {
    const profile: ApplicantProfile = {
      name: "Jordan Rivera",
      monthly_income: 6200,
      desired_rent: 1800,
      credit_score: 712,
      employment_status: "employed",
      bedrooms_wanted: 2,
      bathrooms_wanted: 2,
      bath_type_wanted: "full",
      min_square_feet: 900,
      has_pets: true,
      needs_balcony: true,
      needs_parking: true,
      needs_in_unit_laundry: true,
      furnished_wanted: false,
      preferred_area: "South Congress",
      wanted_amenities: ["Pet Park"],
      lease_term_wanted: 12,
    };
    render(<ProfileCard profile={profile} chunks={3} />);
    expect(screen.getByText("Jordan Rivera")).toBeInTheDocument();
    expect(screen.getByText(/Indexed 3 chunk/)).toBeInTheDocument();
  });
});
