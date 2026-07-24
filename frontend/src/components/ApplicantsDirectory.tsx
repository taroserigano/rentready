import { useCallback, useEffect, useRef, useState } from "react";
import { Cpu } from "lucide-react";
import type {
  ApplicantProfile,
  ApplicantSummary,
  EligibilityResult,
  RecommendResponse,
} from "../types";
import {
  deleteApplicant,
  getApplicant,
  getEligibility,
  getRecommendations,
  listApplicants,
} from "../api";
import { ProfileCard } from "./ProfileCard";
import { EligibilityCard } from "./EligibilityCard";
import { Recommendations } from "./Recommendations";
import { DecisionBar, STATUS_LABEL, STATUS_TONE } from "./DecisionBar";
import { StrengthCard } from "./StrengthCard";
import { RiskCard } from "./risk/RiskCard";
import { Avatar } from "./Avatar";
import { TechBadge } from "./TechBadge";

function errText(e: unknown): string {
  // fetch throws TypeError when the server is unreachable.
  if (e instanceof TypeError) {
    return "Could not reach the server. Is the backend running?";
  }
  if (e instanceof Error && e.message) return e.message;
  return "Could not reach the server. Is the backend running?";
}

function formatCreated(iso: string): string {
  // created_at is UTC; add "Z" when the timezone marker is missing
  // so the browser shows local time.
  const hasZone = /[zZ]$|[+-]\d\d:?\d\d$/.test(iso);
  const d = new Date(hasZone ? iso : `${iso.replace(" ", "T")}Z`);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

interface Detail {
  profile: ApplicantProfile;
  chunks: number;
  eligibility: EligibilityResult | null;
  recommendations: RecommendResponse | null;
}

export function ApplicantsDirectory({
  onViewRisk,
}: {
  onViewRisk?: (id: string) => void;
}) {
  const [applicants, setApplicants] = useState<ApplicantSummary[] | null>(null);
  const [listError, setListError] = useState("");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  const [detail, setDetail] = useState<Detail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState("");

  // Ignore results from stale detail fetches when the user clicks around.
  const detailRequest = useRef(0);

  const loadList = useCallback(async () => {
    setListError("");
    setApplicants(null);
    try {
      setApplicants(await listApplicants());
    } catch (e) {
      setListError(errText(e));
      setApplicants([]);
    }
  }, []);

  useEffect(() => {
    loadList();
  }, [loadList]);

  const clearDetail = () => {
    detailRequest.current += 1;
    setSelectedId(null);
    setDetail(null);
    setDetailError("");
    setDetailLoading(false);
  };

  async function openApplicant(id: string) {
    const req = ++detailRequest.current;
    setSelectedId(id);
    setDetail(null);
    setDetailError("");
    setDetailLoading(true);
    try {
      const { profile } = await getApplicant(id);
      const chunks =
        applicants?.find((a) => a.id === id)?.chunks_indexed ?? 0;
      const [eligibility, recommendations] = await Promise.all([
        getEligibility(id),
        getRecommendations(id),
      ]);
      if (req !== detailRequest.current) return;
      setDetail({ profile, chunks, eligibility, recommendations });
    } catch (e) {
      if (req !== detailRequest.current) return;
      setDetailError(errText(e));
    } finally {
      if (req === detailRequest.current) setDetailLoading(false);
    }
  }

  async function handleDelete(a: ApplicantSummary) {
    if (!window.confirm(`Delete ${a.name}'s application? This cannot be undone.`)) {
      return;
    }
    setDeletingId(a.id);
    try {
      await deleteApplicant(a.id);
      if (selectedId === a.id) clearDetail();
      await loadList();
    } catch (e) {
      setListError(errText(e));
    } finally {
      setDeletingId(null);
    }
  }

  const loading = applicants === null && !listError;

  return (
    <div className="app">
      <header>
        <h1>Saved applicants</h1>
        <p>Re-open an earlier application or delete ones you no longer need.</p>
        <div className="badges">
          <TechBadge
            icon={Cpu}
            label="XGBoost"
            title="Each applicant's late-payment risk badge is scored by a trained XGBoost model with TreeSHAP reason codes."
          />
        </div>
      </header>

      <div className="card">
        <div className="toolbar">
          <span className="toolbar-label">
            Saved applicants{applicants && applicants.length > 0 ? ` · ${applicants.length}` : ""}
          </span>
          <button
            className="btn-small btn-ghost"
            onClick={loadList}
            disabled={loading}
          >
            Refresh
          </button>
        </div>

        {loading && <p className="muted">Loading applicants…</p>}

        {listError && (
          <>
            <div className="error">{listError}</div>
            <button
              className="btn-small btn-ghost"
              style={{ marginTop: 10 }}
              onClick={loadList}
            >
              Retry
            </button>
          </>
        )}

        {applicants && !listError && applicants.length === 0 && (
          <p className="muted">
            No applicants yet. Use the Apply tab to upload a PDF or fill in details.
          </p>
        )}

        {applicants && applicants.length > 0 && (
          <table className="table rows-clickable" style={{ marginTop: 12 }}>
            <thead>
              <tr>
                <th>Name</th>
                <th>Status</th>
                <th>Created</th>
                <th>Documents</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {applicants.map((a) => {
                const deleting = deletingId === a.id;
                return (
                  <tr
                    key={a.id}
                    onClick={() => openApplicant(a.id)}
                    style={
                      selectedId === a.id
                        ? { background: "var(--panel2)" }
                        : undefined
                    }
                  >
                    <td>
                      <span className="cell-avatar">
                        <Avatar name={a.name} size={28} />
                        {a.name}
                      </span>
                    </td>
                    <td>
                      <span className={`badge tone-${STATUS_TONE[a.status ?? "new"] ?? "info"}`}>
                        {STATUS_LABEL[a.status ?? "new"] ?? a.status}
                      </span>
                    </td>
                    <td className="secondary">{formatCreated(a.created_at)}</td>
                    <td>
                      <span className="badge tone-info">
                        <span
                          className={`dot ${a.chunks_indexed > 0 ? "blue" : "teal"}`}
                        />
                        {a.chunks_indexed > 0 ? "PDF indexed" : "Form entry"}
                      </span>
                    </td>
                    <td>
                      <div className="row-actions" style={{ display: "flex", gap: 8 }}>
                        <button
                          className="btn-small btn-ghost"
                          disabled={deleting}
                          onClick={(e) => {
                            e.stopPropagation();
                            openApplicant(a.id);
                          }}
                        >
                          Open
                        </button>
                        <button
                          className="btn-small btn-ghost danger"
                          disabled={deleting}
                          onClick={(e) => {
                            e.stopPropagation();
                            handleDelete(a);
                          }}
                        >
                          {deleting ? "Deleting…" : "Delete"}
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>

      {selectedId && detailLoading && (
        <div className="card">
          <p className="muted" style={{ margin: 0 }}>
            Loading application…
          </p>
        </div>
      )}

      {selectedId && detailError && (
        <div className="card">
          <div className="error">{detailError}</div>
          <button
            className="btn-small btn-ghost"
            style={{ marginTop: 10 }}
            onClick={() => openApplicant(selectedId)}
          >
            Retry
          </button>
        </div>
      )}

      {detail && selectedId && (
        <>
          <ProfileCard profile={detail.profile} chunks={detail.chunks} />
          {detail.eligibility && (
            <EligibilityCard result={detail.eligibility} applicantId={selectedId} />
          )}
          <DecisionBar
            applicantId={selectedId}
            reviewer="Taro"
            onDecision={loadList}
          />
          <StrengthCard applicantId={selectedId} />
          <RiskCard
            applicantId={selectedId}
            onOpenFull={onViewRisk ? () => onViewRisk(selectedId) : undefined}
          />
          {detail.recommendations && (
            <Recommendations
              data={detail.recommendations}
              applicantId={selectedId}
              monthlyIncome={detail.profile.monthly_income}
            />
          )}
        </>
      )}
    </div>
  );
}
