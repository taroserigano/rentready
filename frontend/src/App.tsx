import { Suspense, lazy, useEffect, useMemo, useState, type ReactNode } from "react";
import { Sun, Moon, KeyRound } from "lucide-react";
import { getHealth, getSamples, type Sample } from "./api";
import { CommandPalette, type Command } from "./components/CommandPalette";
import { Toaster } from "./components/Toaster";

// Route-level code splitting. Every view was statically imported before, so the
// first paint of Apply pulled recharts (97kB gz) + framer-motion (42kB gz) and
// the four retired views (15kB gz) it never renders.
const lz = <K extends string>(k: K, f: () => Promise<Record<K, React.ComponentType<any>>>) =>
  lazy(() => f().then((m) => ({ default: m[k] })));

const Evaluations = lz("Evaluations", () => import("./components/Evaluations"));
const Monitoring = lz("Monitoring", () => import("./components/Monitoring"));
const ABLab = lz("ABLab", () => import("./components/ABLab"));
const Learn = lz("Learn", () => import("./components/Learn"));
const ApplyForm = lz("ApplyForm", () => import("./components/ApplyForm"));
const ApplicantsDirectory = lz("ApplicantsDirectory", () => import("./components/ApplicantsDirectory"));
const PropertyBrowser = lz("PropertyBrowser", () => import("./components/PropertyBrowser"));
const Dashboard = lz("Dashboard", () => import("./components/Dashboard"));
const PropertyPage = lz("PropertyPage", () => import("./components/PropertyPage"));
const Tours = lz("Tours", () => import("./components/Tours"));
const Concierge = lz("Concierge", () => import("./components/Concierge"));
const Risk = lz("Risk", () => import("./components/risk/Risk"));
const Residents = lz("Residents", () => import("./components/residents/Residents"));
const AiReport = lz("AiReport", () => import("./components/AiReport"));

type View =
  | "apply"
  | "applicants"
  | "properties"
  | "property"
  | "tours"
  | "ask"
  | "dashboard"
  | "risk"
  | "residents"
  | "report"
  | "evaluations"
  | "monitoring"
  | "ab"
  | "learn";

const VIEWS: View[] = [
  "apply", "applicants", "properties", "property", "tours", "ask",
  "dashboard", "risk", "residents", "report", "evaluations", "monitoring", "ab", "learn",
];
const VIEW_SET = new Set<string>(VIEWS);

/**
 * The URL hash is the source of truth for the current page + any deep-link ids:
 *   "#/residents/PROP-041/RES-0018" -> {view:"residents", ids:["PROP-041","RES-0018"]}
 *   "#/property/PROP-007"           -> {view:"property",  ids:["PROP-007"]}
 *   "#/risk/APP-123"                -> {view:"risk",      ids:["APP-123"]}
 *   "#/apply/jordan-rivera"         -> {view:"apply",      ids:["jordan-rivera"]} (sample slug)
 *   ""                              -> {view:"apply", ids:[]}
 */
function parseHash(): { view: View; ids: string[] } {
  const parts = window.location.hash.replace(/^#\/?/, "").split("/").filter(Boolean);
  const slug = parts[0] || "apply";
  if (!VIEW_SET.has(slug)) return { view: "apply", ids: [] };
  return { view: slug as View, ids: parts.slice(1) };
}

/** Build the hash for a view (+ optional deep-link ids). apply (no ids) -> "#/". */
function routeToHash(view: View, ids: (string | null | undefined)[] = []): string {
  const clean = ids.filter(Boolean) as string[];
  if (view === "apply" && clean.length === 0) return "#/";
  return "#/" + [view, ...clean].join("/");
}

export default function App() {
  const [health, setHealth] = useState<Record<string, unknown> | null>(null);
  // Initial view + any deep-link ids come from the URL hash (see parseHash).
  const r0 = parseHash();
  const idAt = (v: View, i: number) => (r0.view === v ? (r0.ids[i] ?? null) : null);
  const [propertyId, setPropertyId] = useState<string | null>(() => idAt("property", 0));
  const [tourPropertyId, setTourPropertyId] = useState<string | null>(() => idAt("tours", 0));
  const [askPropertyId, setAskPropertyId] = useState<string | null>(() => idAt("ask", 0));
  const [riskApplicantId, setRiskApplicantId] = useState<string | null>(() => idAt("risk", 0));
  const [residentPropertyId, setResidentPropertyId] = useState<string | null>(() => idAt("residents", 0));
  const [residentId, setResidentId] = useState<string | null>(() => idAt("residents", 1));
  // Sample-slug deep link for Apply, e.g. "#/apply/jordan-rivera" (command palette).
  const [applySampleSlug, setApplySampleSlug] = useState<string | null>(() => idAt("apply", 0));
  const [view, setView] = useState<View>(r0.view);

  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth(null));
  }, []);

  // Every nav-triggered page change starts scrolled to the top, rather than
  // keeping whatever scroll position the previous page was left at.
  useEffect(() => {
    window.scrollTo(0, 0);
  }, [view]);

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
      setApplySampleSlug(r.view === "apply" ? (r.ids[0] ?? null) : null);
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

  // Samples power the command palette's "Load sample" commands.
  const [samples, setSamples] = useState<Sample[]>([]);
  useEffect(() => {
    getSamples().then(setSamples).catch(() => {});
  }, []);

  const commands = useMemo<Command[]>(() => {
    const views: Array<[View, string]> = [
      ["apply", "Apply"],
      ["applicants", "Applicants"],
      ["properties", "Properties"],
      ["tours", "Tours"],
      ["ask", "Lease Q&A"],
      ["dashboard", "Dashboard"],
      ["risk", "Risk Assessment"],
      ["residents", "Residents"],
      ["report", "AI Report"],
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
          window.location.hash = routeToHash("apply", [s.slug]);
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

  // ONE shell for every view: a skip link, the nav, and a <main> landmark.
  // Previously each of the 14 branches returned its own <Nav/> + bare view with
  // no landmark and no skip link, so a screen-reader user had to tab the 9-item
  // nav on every page with no way past it.
  let page: ReactNode;
  if (view === "applicants") {
    page = <ApplicantsDirectory onViewRisk={goToRisk} />;
  } else if (view === "properties") {
    page = <PropertyBrowser onOpenListing={goToProperty} health={health} />;
  } else if (view === "property" && propertyId) {
    page = (
      <PropertyPage
        propertyId={propertyId}
        onBack={() => navigate("properties")}
        onApply={() => navigate("apply")}
        onBookTour={goToTours}
        onAsk={goToAsk}
      />
    );
  } else if (view === "ask") {
    page = (
      <Concierge
        initialPropertyId={askPropertyId ?? undefined}
        onViewProperty={goToProperty}
        health={health}
      />
    );
  } else if (view === "tours") {
    page = <Tours initialPropertyId={tourPropertyId ?? undefined} health={health} />;
  } else if (view === "dashboard") {
    page = <Dashboard health={health} />;
  } else if (view === "risk") {
    page = <Risk initialApplicantId={riskApplicantId ?? undefined} health={health} />;
  } else if (view === "residents") {
    page = (
      <Residents
        initialPropertyId={residentPropertyId ?? undefined}
        initialResidentId={residentId ?? undefined}
        health={health}
      />
    );
  } else if (view === "report") {
    page = <AiReport />;
  } else if (view === "evaluations") {
    page = <Evaluations />;
  } else if (view === "monitoring") {
    page = <Monitoring />;
  } else if (view === "ab") {
    page = <ABLab />;
  } else if (view === "learn") {
    page = <Learn setView={navigate} />;
  } else {
    // Default / fallback (unmatched hash, or `property` with no id yet): Apply —
    // upload a PDF or fill in details manually; every downstream result card
    // (eligibility, risk, recommendations, chat) lives on this one page.
    page = (
      <ApplyForm
        health={health}
        initialSampleSlug={applySampleSlug ?? undefined}
        onOpenRisk={goToRisk}
        onViewListing={goToProperty}
      />
    );
  }

  return (
    <>
      <a className="skip-link" href="#main-content">Skip to main content</a>
      <Nav view={view} setView={navigate} commands={commands} />
      <main id="main-content">
        {/* Views are code-split (React.lazy), so a fallback is required. */}
        <Suspense fallback={<ViewFallback />}>{page}</Suspense>
      </main>
    </>
  );
}

/** Placeholder while a lazily-loaded view's chunk is fetched. */
function ViewFallback() {
  return (
    <div className="app">
      <div className="card">
        <p className="muted" style={{ margin: 0 }} role="status">
          Loading…
        </p>
      </div>
    </div>
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
      <button
        type="button"
        className="brand"
        onClick={() => setView("apply")}
        aria-label="RentReady home"
        title="Go to home"
      >
        <span className="brand-mark" aria-hidden>
          <KeyRound size={15} strokeWidth={2.4} />
        </span>
        RentReady
      </button>
      {tab("apply", "Apply")}
      {tab("applicants", "Applicants")}
      {tab("properties", "Properties")}
      {tab("tours", "Tours")}
      {tab("ask", "Lease Q&A")}
      {tab("dashboard", "Dashboard")}
      {tab("risk", "Risk Assessment")}
      {tab("residents", "Residents")}
      {tab("report", "AI Report")}
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

