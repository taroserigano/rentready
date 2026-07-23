import { useState } from "react";
import { Download, FileText } from "lucide-react";
import { pdfUrl } from "../api";

/**
 * Shows the application PDF (uploaded, sample, or generated on apply) in an
 * inline viewer with a download link. Collapsible to keep the workspace tidy.
 */
export function ApplicationPdf({
  applicantId,
  sectionNumber,
}: {
  applicantId: string;
  sectionNumber?: string;
}) {
  const [open, setOpen] = useState(true);
  const url = pdfUrl(applicantId);
  // Chrome's built-in PDF viewer renders its own toolbar + thumbnail sidebar
  // inside the <object>; these fragment params (a long-standing convention it
  // honors) suppress that chrome so only the page itself shows — our own
  // header above already provides download/hide.
  const viewerUrl = `${url}#toolbar=0&navpanes=0&scrollbar=0`;

  return (
    <div className="card">
      <div className="rec-head">
        <h2 style={{ margin: 0 }}>
          {sectionNumber ? `${sectionNumber} ` : ""}Application PDF
        </h2>
        <span style={{ marginLeft: "auto", display: "inline-flex", gap: 8 }}>
          <a
            className="btn-small btn-ghost icon-line"
            href={url}
            target="_blank"
            rel="noreferrer"
            download
          >
            <Download size={14} /> Download
          </a>
          <button
            className="btn-small btn-ghost icon-line"
            aria-expanded={open}
            onClick={() => setOpen((v) => !v)}
          >
            <FileText size={14} /> {open ? "Hide" : "Show"}
          </button>
        </span>
      </div>
      {open && (
        <object
          data={viewerUrl}
          type="application/pdf"
          aria-label="Application PDF"
          style={{
            width: "100%",
            height: 520,
            border: "1px solid var(--line)",
            borderRadius: "var(--r-md)",
            marginTop: 10,
            background: "var(--field)",
          }}
        >
          <p className="muted" style={{ padding: 12 }}>
            Can't display the PDF inline.{" "}
            <a href={url} target="_blank" rel="noreferrer">
              Open it in a new tab
            </a>
            .
          </p>
        </object>
      )}
    </div>
  );
}
