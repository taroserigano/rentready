import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  AlignLeft,
  Car,
  DollarSign,
  Download,
  ExternalLink,
  FileText,
  PawPrint,
  ShieldCheck,
  Sofa,
  CalendarDays,
  X,
} from "lucide-react";
import type { LeaseDoc } from "../types";
import { getLease, leasePdfUrl } from "../api";

/** Module-level cache so reopening a lease is instant. */
const leaseCache = new Map<string, LeaseDoc>();

/** Stable anchor id from a section title (e.g. "Pets & Animals" -> "sec-pets-animals"). */
export function sectionSlug(section: string): string {
  return (
    "sec-" +
    section
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "")
  );
}

function money(n: number): string {
  return `$${Math.round(n).toLocaleString()}`;
}

type Tab = "summary" | "document";

interface Props {
  propertyId: string;
  /** Section title to scroll to + briefly highlight on open (forces the Summary tab). */
  initialSection?: string;
  /** Which tab to open on. Ignored when `initialSection` is set. Default "summary". */
  initialTab?: Tab;
  onClose: () => void;
}

export function LeaseViewer({
  propertyId,
  initialSection,
  initialTab,
  onClose,
}: Props) {
  const [doc, setDoc] = useState<LeaseDoc | null>(
    () => leaseCache.get(propertyId) ?? null,
  );
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>(initialSection ? "summary" : (initialTab ?? "summary"));
  const panelRef = useRef<HTMLDivElement>(null);

  // Load (once) with a per-property cache.
  useEffect(() => {
    let alive = true;
    const cached = leaseCache.get(propertyId);
    if (cached) {
      setDoc(cached);
      return;
    }
    setDoc(null);
    setError(null);
    getLease(propertyId)
      .then((d) => {
        leaseCache.set(propertyId, d);
        if (alive) setDoc(d);
      })
      .catch((e) => {
        if (alive)
          setError(
            e instanceof Error ? e.message : "Could not load the lease.",
          );
      });
    return () => {
      alive = false;
    };
  }, [propertyId]);

  // Move focus into the dialog; Esc closes.
  useEffect(() => {
    const prevFocus = document.activeElement as HTMLElement | null;
    panelRef.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      prevFocus?.focus?.();
    };
  }, [onClose]);

  // Once the doc is present, scroll the requested section into view + flash it.
  useEffect(() => {
    if (!doc || !initialSection || tab !== "summary") return;
    const id = sectionSlug(initialSection);
    // Wait a frame so the sections are in the DOM.
    const t = window.setTimeout(() => {
      const el = panelRef.current?.querySelector<HTMLElement>(`#${CSS.escape(id)}`);
      if (!el) return;
      el.scrollIntoView({ block: "start" });
      el.classList.add("highlight");
      window.setTimeout(() => el.classList.remove("highlight"), 1600);
    }, 60);
    return () => window.clearTimeout(t);
  }, [doc, initialSection, tab]);

  const kt = doc?.key_terms;
  const pdfHref = leasePdfUrl(propertyId);

  return (
    <AnimatePresence>
      <motion.div
        className="modal-backdrop"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.15 }}
        onClick={onClose}
      >
        <motion.div
          className="modal-panel lease-viewer-panel"
          ref={panelRef}
          tabIndex={-1}
          role="dialog"
          aria-modal="true"
          aria-label={doc ? `${doc.property_name} lease` : "Lease"}
          initial={{ opacity: 0, scale: 0.96 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.96 }}
          transition={{ duration: 0.18, ease: "easeOut" }}
          onClick={(e) => e.stopPropagation()}
        >
          <div className="modal-body lease-doc">
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "flex-start",
                gap: 12,
              }}
            >
              <div>
                <div className="eyebrow" style={{ marginBottom: 4 }}>
                  Lease
                </div>
                <div style={{ fontWeight: 700, fontSize: 17 }}>
                  {doc ? doc.property_name : "Loading lease…"}
                </div>
              </div>
              <button
                className="btn-small btn-ghost"
                onClick={onClose}
                aria-label="Close"
                style={{ display: "inline-flex", alignItems: "center", gap: 6, flexShrink: 0 }}
              >
                <X size={14} /> Close
              </button>
            </div>

            {kt && (
              <div className="lease-key-terms">
                <span className="badge tone-info icon-line">
                  <DollarSign size={12} /> Rent {money(kt.rent)}/mo
                </span>
                <span className="badge tone-info icon-line">
                  <ShieldCheck size={12} /> Deposit {money(kt.deposit)}
                </span>
                <span className="badge tone-info icon-line">
                  <CalendarDays size={12} /> Term {kt.term_months} months
                </span>
                <span
                  className={`badge icon-line ${kt.pets ? "tone-good" : "tone-info"}`}
                >
                  <PawPrint size={12} /> Pets {kt.pets ? "Yes" : "No"}
                </span>
                <span className="badge tone-info icon-line">
                  <Car size={12} /> {kt.parking || "No parking"}
                </span>
                <span
                  className={`badge icon-line ${kt.furnished ? "tone-good" : "tone-info"}`}
                >
                  <Sofa size={12} /> {kt.furnished ? "Furnished" : "Unfurnished"}
                </span>
              </div>
            )}

            <div className="lease-tabs" role="tablist">
              <button
                type="button"
                role="tab"
                aria-selected={tab === "summary"}
                className={`lease-tab${tab === "summary" ? " active" : ""}`}
                onClick={() => setTab("summary")}
              >
                <AlignLeft size={14} /> Summary
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={tab === "document"}
                className={`lease-tab${tab === "document" ? " active" : ""}`}
                onClick={() => setTab("document")}
              >
                <FileText size={14} /> Document (PDF)
              </button>
            </div>

            {tab === "summary" ? (
              <>
                {error && (
                  <div className="alert bad" style={{ marginTop: 14 }}>
                    {error}
                  </div>
                )}

                {!doc && !error && (
                  <div
                    className="chat-typing"
                    aria-label="Loading lease"
                    style={{ marginTop: 16 }}
                  >
                    <span />
                    <span />
                    <span />
                  </div>
                )}

                {doc &&
                  doc.sections.map((s) => (
                    <section
                      key={s.section}
                      id={sectionSlug(s.section)}
                      className="lease-section"
                    >
                      <h3>{s.section}</h3>
                      <p>{s.text}</p>
                    </section>
                  ))}
              </>
            ) : (
              <div className="lease-pdf-pane">
                <div className="lease-pdf-toolbar">
                  <a
                    className="btn-small btn-ghost icon-line"
                    href={pdfHref}
                    download
                  >
                    <Download size={14} /> Download
                  </a>
                  <a
                    className="btn-small btn-ghost icon-line"
                    href={pdfHref}
                    target="_blank"
                    rel="noreferrer"
                  >
                    <ExternalLink size={14} /> Open in new tab
                  </a>
                </div>
                <object
                  data={pdfHref}
                  type="application/pdf"
                  className="lease-pdf-frame"
                  aria-label="Lease agreement PDF"
                >
                  <p className="muted" style={{ padding: 12 }}>
                    Can't display the PDF inline.{" "}
                    <a href={pdfHref} target="_blank" rel="noreferrer">
                      Open it in a new tab
                    </a>
                    .
                  </p>
                </object>
              </div>
            )}
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}
