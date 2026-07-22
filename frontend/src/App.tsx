import { useEffect, useMemo, useRef, useState } from "react";
import {
  Upload,
  Sun,
  Moon,
  FileText,
  ShieldCheck,
  Building2,
  MessagesSquare,
  ArrowRight,
} from "lucide-react";
import {
  ask,
  getEligibility,
  getHealth,
  getRecommendations,
  getSamples,
  loadSample,
  uploadPdf,
  type Sample,
} from "./api";
import type {
  AskResponse,
  EligibilityResult,
  RecommendResponse,
  UploadResponse,
} from "./types";
import { ProfileCard } from "./components/ProfileCard";
import { Stepper, SkeletonCard, type Phase } from "./components/Loading";
import { CommandPalette, type Command } from "./components/CommandPalette";
import { Toaster } from "./components/Toaster";
import { ApplicationPdf } from "./components/ApplicationPdf";
import { EligibilityCard } from "./components/EligibilityCard";
import { FinancialHealth } from "./components/FinancialHealth";
import { StrengthCard } from "./components/StrengthCard";
import { WhatIfSimulator } from "./components/WhatIfSimulator";
import { Recommendations } from "./components/Recommendations";
import { Chat } from "./components/Chat";
import { GraphAsk } from "./components/GraphAsk";
import { SampleApplicants } from "./components/SampleApplicants";
import { Evaluations } from "./components/Evaluations";
import { Monitoring } from "./components/Monitoring";
import { ABLab } from "./components/ABLab";
import { Learn } from "./components/Learn";
import { ApplyForm } from "./components/ApplyForm";
import { ApplicantsDirectory } from "./components/ApplicantsDirectory";
import { PropertyBrowser } from "./components/PropertyBrowser";
import { Dashboard } from "./components/Dashboard";
import { PropertyPage } from "./components/PropertyPage";
import { Tours } from "./components/Tours";
import { Concierge } from "./components/Concierge";
import { Risk } from "./components/risk/Risk";
import { RiskCard } from "./components/risk/RiskCard";
import { Residents } from "./components/residents/Residents";

type View =
  | "workspace"
  | "apply"
  | "applicants"
  | "properties"
  | "property"
  | "tours"
  | "ask"
  | "dashboard"
  | "risk"
  | "residents"
  | "evaluations"
  | "monitoring"
  | "ab"
  | "learn";

const VIEWS: View[] = [
  "workspace", "apply", "applicants", "properties", "property", "tours", "ask",
  "dashboard", "risk", "residents", "evaluations", "monitoring", "ab", "learn",
];
const VIEW_SET = new Set<string>(VIEWS);

/**
 * The URL hash is the source of truth for the current page + any deep-link ids:
 *   "#/residents/PROP-041/RES-0018" -> {view:"residents", ids:["PROP-041","RES-0018"]}
 *   "#/property/PROP-007"           -> {view:"property",  ids:["PROP-007"]}
 *   "#/risk/APP-123"                -> {view:"risk",      ids:["APP-123"]}
 *   ""                              -> {view:"workspace", ids:[]}
 */
function parseHash(): { view: View; ids: string[] } {
  const parts = window.location.hash.replace(/^#\/?/, "").split("/").filter(Boolean);
  const slug = parts[0] || "workspace";
  if (!VIEW_SET.has(slug)) return { view: "workspace", ids: [] };
  return { view: slug as View, ids: parts.slice(1) };
}

/** Build the hash for a view (+ optional deep-link ids). workspace -> "#/". */
function routeToHash(view: View, ids: (string | null | undefined)[] = []): string {
  const clean = ids.filter(Boolean) as string[];
  if (view === "workspace" && clean.length === 0) return "#/";
  return "#/" + [view, ...clean].join("/");
}

export default function App() {
  const [health, setHealth] = useState<Record<string, unknown> | null>(null);
  const [upload, setUpload] = useState<UploadResponse | null>(null);
  const [eligibility, setEligibility] = useState<EligibilityResult | null>(null);
  const [recs, setRecs] = useState<RecommendResponse | null>(null);
  const [phase, setPhase] = useState<Phase>("idle");
  const [error, setError] = useState("");
  const runIdRef = useRef(0);
  const loading = phase === "extracting" || phase === "screening";
  // Initial view + any deep-link ids come from the URL hash (see parseHash).
  const r0 = parseHash();
  const idAt = (v: View, i: number) => (r0.view === v ? (r0.ids[i] ?? null) : null);
  const [propertyId, setPropertyId] = useState<string | null>(() => idAt("property", 0));
  const [tourPropertyId, setTourPropertyId] = useState<string | null>(() => idAt("tours", 0));
  const [askPropertyId, setAskPropertyId] = useState<string | null>(() => idAt("ask", 0));
  const [riskApplicantId, setRiskApplicantId] = useState<string | null>(() => idAt("risk", 0));
  const [residentPropertyId, setResidentPropertyId] = useState<string | null>(() => idAt("residents", 0));
  const [residentId, setResidentId] = useState<string | null>(() => idAt("residents", 1));
  const [view, setView] = useState<View>(r0.view);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth(null));
  }, []);

  // URL hash is the source of truth: nav clicks set the hash, and this syncs
  // view + deep-link ids on every change (so back/forward and direct links work).
  useEffect(() => {
    const onHash = () => {
      const r = parseHash();
      setPropertyId(r.view === "property" ? (r.ids[0] ?? null) : null);
      setTourPropertyId(r.view === "tours" ? (r.ids[0] ?? null) : null);
      setAskPropertyId(r.view === "ask" ? (r.ids[0] ?? null) : null);
      setRiskApplicantId(r.view === "risk" ? (r.ids[0] ?? null) : null);
      setResidentPropertyId(r.view === "residents" ? (r.ids[0] ?? null) : null);
      setResidentId(r.view === "residents" ? (r.ids[1] ?? null) : null);
      setView(r.view);
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  // Navigation is expressed as a URL-hash change; the hashchange listener above
  // then syncs view + deep-link ids. Every page therefore has its own URL.
  /** Open the dedicated page for a property. */
  function goToProperty(id: string) {
    window.location.hash = routeToHash("property", [id]);
  }
  /** Open the Tour Scheduler locked to a specific property. */
  function goToTours(id: string) {
    window.location.hash = routeToHash("tours", [id]);
  }
  /** Open the Concierge (Ask) page scoped to a specific property. */
  function goToAsk(id: string) {
    window.location.hash = routeToHash("ask", [id]);
  }
  /** Open the Risk page with a specific applicant pre-selected. */
  function goToRisk(id: string) {
    window.location.hash = routeToHash("risk", [id]);
  }
  /** Open the Residents page, optionally scoped to a property and/or resident. */
  function goToResidents(opts?: { propertyId?: string; residentId?: string }) {
    window.location.hash = routeToHash("residents", [opts?.propertyId, opts?.residentId]);
  }
  /** Nav switcher — sets the URL for the target view. */
  function navigate(v: View) {
    window.location.hash = routeToHash(v);
  }

  async function runFlow(getUpload: () => Promise<UploadResponse>) {
    const myRun = ++runIdRef.current;
    const alive = () => myRun === runIdRef.current;
    setError("");
    setPhase("extracting");
    setUpload(null);
    setEligibility(null);
    setRecs(null);
    try {
      const res = await getUpload();
      if (!alive()) return;
      setUpload(res);
      setPhase("screening");
      // Eligibility and recommendations run concurrently but reveal
      // independently, so each card replaces its skeleton as it resolves.
      const eligP = getEligibility(res.applicant_id).then((e) => {
        if (alive()) setEligibility(e);
      });
      const recP = getRecommendations(res.applicant_id).then((r) => {
        if (alive()) setRecs(r);
      });
      await Promise.all([eligP, recP]);
      if (alive()) setPhase("done");
    } catch (e) {
      if (alive()) {
        setError(String(e));
        setPhase("error");
      }
    }
  }

  const handleFile = (file: File) => runFlow(() => uploadPdf(file));
  const handleSample = (slug: string) => runFlow(() => loadSample(slug));

  async function handleAsk(question: string): Promise<AskResponse> {
    return ask(upload!.applicant_id, question);
  }

  // Samples power the command palette's "Load sample" commands.
  const [samples, setSamples] = useState<Sample[]>([]);
  useEffect(() => {
    getSamples().then(setSamples).catch(() => {});
  }, []);

  const commands = useMemo<Command[]>(() => {
    const views: Array<[View, string]> = [
      ["workspace", "Workspace"],
      ["apply", "Apply"],
      ["applicants", "Applicants"],
      ["properties", "Properties"],
      ["tours", "Tours"],
      ["ask", "Ask"],
      ["dashboard", "Dashboard"],
      ["risk", "Risk"],
      ["residents", "Residents"],
      ["evaluations", "Evaluations"],
      ["monitoring", "Monitoring"],
      ["ab", "A/B Lab"],
      ["learn", "Learn"],
    ];
    const cmds: Command[] = views.map(([v, label]) => ({
      id: `go-${v}`,
      label: `Go to ${label}`,
      group: "Navigate",
      run: () => navigate(v),
    }));
    for (const s of samples) {
      cmds.push({
        id: `sample-${s.slug}`,
        label: `Load sample: ${s.name}`,
        group: "Sample",
        run: () => {
          navigate("workspace");
          handleSample(s.slug);
        },
      });
    }
    cmds.push({
      id: "residents-portfolio",
      label: "Residents: open portfolio overview",
      group: "Navigate",
      run: () => goToResidents(),
    });
    cmds.push({
      id: "theme",
      label: "Toggle light / dark theme",
      group: "Theme",
      run: () => {
        const cur = document.documentElement.getAttribute("data-theme") || "dark";
        const next = cur === "dark" ? "light" : "dark";
        document.documentElement.setAttribute("data-theme", next);
        try {
          localStorage.setItem("rr-theme", next);
        } catch {
          /* ignore */
        }
      },
    });
    return cmds;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [samples]);

  if (view === "apply") {
    return (
      <>
        <Nav view={view} setView={navigate} commands={commands} />
        <ApplyForm />
      </>
    );
  }

  if (view === "applicants") {
    return (
      <>
        <Nav view={view} setView={navigate} commands={commands} />
        <ApplicantsDirectory onViewRisk={goToRisk} />
      </>
    );
  }

  if (view === "properties") {
    return (
      <>
        <Nav view={view} setView={navigate} commands={commands} />
        <PropertyBrowser onOpenListing={goToProperty} />
      </>
    );
  }

  if (view === "property" && propertyId) {
    return (
      <>
        <Nav view={view} setView={navigate} commands={commands} />
        <PropertyPage
          propertyId={propertyId}
          onBack={() => navigate("properties")}
          onApply={() => navigate("apply")}
          onBookTour={goToTours}
          onAsk={goToAsk}
        />
      </>
    );
  }

  if (view === "ask") {
    return (
      <>
        <Nav view={view} setView={navigate} commands={commands} />
        <Concierge
          initialPropertyId={askPropertyId ?? undefined}
          onViewProperty={goToProperty}
        />
      </>
    );
  }

  if (view === "tours") {
    return (
      <>
        <Nav view={view} setView={navigate} commands={commands} />
        <Tours initialPropertyId={tourPropertyId ?? undefined} />
      </>
    );
  }

  if (view === "dashboard") {
    return (
      <>
        <Nav view={view} setView={navigate} commands={commands} />
        <Dashboard />
      </>
    );
  }

  if (view === "risk") {
    return (
      <>
        <Nav view={view} setView={navigate} commands={commands} />
        <Risk initialApplicantId={riskApplicantId ?? undefined} />
      </>
    );
  }

  if (view === "residents") {
    return (
      <>
        <Nav view={view} setView={navigate} commands={commands} />
        <Residents
          initialPropertyId={residentPropertyId ?? undefined}
          initialResidentId={residentId ?? undefined}
        />
      </>
    );
  }

  if (view === "evaluations") {
    return (
      <>
        <Nav view={view} setView={navigate} commands={commands} />
        <Evaluations />
      </>
    );
  }

  if (view === "monitoring") {
    return (
      <>
        <Nav view={view} setView={navigate} commands={commands} />
        <Monitoring />
      </>
    );
  }

  if (view === "ab") {
    return (
      <>
        <Nav view={view} setView={navigate} commands={commands} />
        <ABLab />
      </>
    );
  }

  if (view === "learn") {
    return (
      <>
        <Nav view={view} setView={navigate} commands={commands} />
        <Learn setView={navigate} />
      </>
    );
  }

  const applicantsLoaded = health?.applicants_loaded;
  const reviewedCount =
    typeof applicantsLoaded === "number" ? applicantsLoaded : null;

  return (
    <>
      <Nav view={view} setView={navigate} commands={commands} />
      <div className="app">
      <header>
        <h1>Workspace</h1>
        <p>
          Upload a rental application PDF to check eligibility and get matched
          to properties.
        </p>
        <div className="badges">
          <Badge on={!!health?.anthropic_key_set} label="Claude" tone="violet" />
          <Badge on={!!health?.langsmith} label="LangSmith" tone="teal" />
          <Badge on={!!health?.phoenix} label="Phoenix" tone="magenta" />
          <Badge on={!!health?.neo4j_available} label="Neo4j" tone="blue" />
        </div>
      </header>

      <div className={upload ? undefined : "ws-grid"}>
      <div className="card">
        <h2>1. Upload application</h2>
        <div
          className="dropzone"
          onClick={() => fileRef.current?.click()}
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault();
            const f = e.dataTransfer.files?.[0];
            if (f) handleFile(f);
          }}
        >
          <input
            ref={fileRef}
            type="file"
            accept="application/pdf"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) handleFile(f);
            }}
          />
          <Upload size={22} aria-hidden />
          <span className="dz-title">
            {loading ? "Processing…" : "Drop a rental application PDF"}
          </span>
          {!loading && <span className="dz-sub">or click to browse</span>}
        </div>
        <SampleApplicants onPick={handleSample} disabled={loading} />
        {error && <div className="error">{error}</div>}
      </div>

      {!upload && (
        <aside className="card ws-aside">
          <div className="eyebrow">How it works</div>
          <ol className="how-steps">
            <li className="how-step">
              <span className="how-icon" data-tone="violet"><FileText size={16} aria-hidden /></span>
              <div className="how-body">
                <div className="how-title">Extract the profile</div>
                <div className="how-desc">
                  Claude reads the application PDF into clean, structured fields.
                </div>
              </div>
            </li>
            <li className="how-step">
              <span className="how-icon" data-tone="teal"><ShieldCheck size={16} aria-hidden /></span>
              <div className="how-body">
                <div className="how-title">Check eligibility</div>
                <div className="how-desc">
                  Transparent rules produce a verdict and the reasons behind it.
                </div>
              </div>
            </li>
            <li className="how-step">
              <span className="how-icon" data-tone="magenta"><Building2 size={16} aria-hidden /></span>
              <div className="how-body">
                <div className="how-title">Match properties</div>
                <div className="how-desc">
                  Rank homes that fit the budget, must-haves, and location.
                </div>
              </div>
            </li>
            <li className="how-step">
              <span className="how-icon" data-tone="warm"><MessagesSquare size={16} aria-hidden /></span>
              <div className="how-body">
                <div className="how-title">Ask anything</div>
                <div className="how-desc">
                  Chat over the applicant and query the property graph.
                </div>
              </div>
            </li>
          </ol>
          <div className="ws-aside-foot">
            {reviewedCount !== null ? (
              <span className="badge tone-warm">
                <span className="dot" aria-hidden />
                {reviewedCount} applicants reviewed
              </span>
            ) : (
              <span>New here? Start with a sample.</span>
            )}
            <button
              className="btn-small btn-ghost"
              onClick={() => navigate("learn")}
            >
              Take the tour <ArrowRight size={13} aria-hidden />
            </button>
          </div>
        </aside>
      )}
      </div>

      <Stepper
        phase={phase}
        ready={{ profile: !!upload, eligibility: !!eligibility, recs: !!recs }}
      />

      {upload ? (
        <ProfileCard profile={upload.profile} chunks={upload.chunks_indexed} />
      ) : (
        phase === "extracting" && <SkeletonCard title="2. Extracted profile" lines={5} />
      )}
      {upload?.has_pdf && (
        <ApplicationPdf applicantId={upload.applicant_id} sectionNumber="2b." />
      )}
      {eligibility ? (
        <EligibilityCard result={eligibility} applicantId={upload?.applicant_id} />
      ) : (
        phase === "screening" && <SkeletonCard title="3. Eligibility" lines={2} block={80} />
      )}
      {upload && <FinancialHealth profile={upload.profile} />}
      {upload && <StrengthCard applicantId={upload.applicant_id} sectionNumber="3c." />}
      {upload && (
        <RiskCard
          applicantId={upload.applicant_id}
          sectionNumber="3d."
          onOpenFull={() => goToRisk(upload.applicant_id)}
        />
      )}
      {upload && (
        <WhatIfSimulator profile={upload.profile} applicantId={upload.applicant_id} />
      )}
      {recs ? (
        <Recommendations
          data={recs}
          applicantId={upload?.applicant_id}
          monthlyIncome={upload?.profile.monthly_income}
          onViewListing={goToProperty}
        />
      ) : (
        phase === "screening" && (
          <SkeletonCard title="4. Recommended properties" lines={2} block={200} />
        )
      )}
      {upload && <Chat onAsk={handleAsk} applicantId={upload?.applicant_id} />}
      {upload && <GraphAsk neo4jAvailable={!!health?.neo4j_available} />}
      </div>
    </>
  );
}

function Nav({
  view,
  setView,
  commands,
}: {
  view: View;
  setView: (v: View) => void;
  commands?: Command[];
}) {
  // A property page is still "Properties" as far as the nav is concerned.
  const isActive = (id: View) =>
    view === id || (view === "property" && id === "properties");
  const tab = (id: View, label: string) => (
    <button
      className={`nav-link${isActive(id) ? " active" : ""}`}
      aria-current={isActive(id) ? "page" : undefined}
      onClick={() => setView(id)}
    >
      {label}
    </button>
  );
  return (
    <nav className="nav">
      <span className="brand">RentReady</span>
      {tab("workspace", "Workspace")}
      {tab("apply", "Apply")}
      {tab("applicants", "Applicants")}
      {tab("properties", "Properties")}
      {tab("tours", "Tours")}
      {tab("ask", "Ask")}
      {tab("dashboard", "Dashboard")}
      {tab("risk", "Risk")}
      {tab("residents", "Residents")}
      <ThemeToggle />
      {commands && <CommandPalette commands={commands} />}
      <Toaster />
    </nav>
  );
}

type Theme = "light" | "dark";

/** Nav-right theme switch. Persists to localStorage; the initial value is set
 *  before first paint by the inline script in index.html. */
function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>(
    () =>
      (document.documentElement.getAttribute("data-theme") as Theme) || "dark",
  );

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    try {
      localStorage.setItem("rr-theme", theme);
    } catch {
      /* private mode / storage disabled — ignore */
    }
  }, [theme]);

  const next: Theme = theme === "dark" ? "light" : "dark";
  return (
    <button
      className="nav-link theme-toggle"
      onClick={() => setTheme(next)}
      title={`Switch to ${next} theme`}
      aria-label={`Switch to ${next} theme`}
    >
      {theme === "dark" ? <Sun size={15} aria-hidden /> : <Moon size={15} aria-hidden />}
    </button>
  );
}

/** Quiet service-health chip: neutral outline, colored status dot. */
function Badge({
  on,
  label,
  tone,
}: {
  on: boolean;
  label: string;
  tone?: "violet" | "teal" | "magenta" | "blue";
}) {
  return (
    <span
      className="badge tone-info"
      data-tone={tone}
      title={`${label} is ${on ? "connected" : "not connected"}`}
    >
      <span className={`dot ${on ? "good" : "bad"}`} aria-hidden />
      {label}
      <span className="sr-only">{on ? "connected" : "not connected"}</span>
    </span>
  );
}
