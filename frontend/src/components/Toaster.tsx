import { CheckCircle2, Info, XCircle } from "lucide-react";
import { useToasts } from "../toast";

/** Fixed-position stack of transient toasts. Mount once (in Nav). */
export function Toaster() {
  const toasts = useToasts();
  if (toasts.length === 0) return null;
  return (
    <div className="toast-wrap" role="status" aria-live="polite">
      {toasts.map((t) => (
        <div key={t.id} className={`toast ${t.tone}`}>
          {t.tone === "good" ? (
            <CheckCircle2 size={15} />
          ) : t.tone === "bad" ? (
            <XCircle size={15} />
          ) : (
            <Info size={15} />
          )}
          {t.message}
        </div>
      ))}
    </div>
  );
}
