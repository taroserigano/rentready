import { useEffect, useState } from "react";

export interface Toast {
  id: number;
  message: string;
  tone: "good" | "bad" | "info";
}

type Listener = (toasts: Toast[]) => void;

let toasts: Toast[] = [];
let nextId = 1;
const listeners = new Set<Listener>();

function emit() {
  for (const l of listeners) l(toasts);
}

/** Show a transient toast. Call from anywhere. */
export function toast(message: string, tone: Toast["tone"] = "info") {
  const id = nextId++;
  toasts = [...toasts, { id, message, tone }];
  emit();
  setTimeout(() => {
    toasts = toasts.filter((t) => t.id !== id);
    emit();
  }, 3200);
}

/** Subscribe a component to the toast list. */
export function useToasts(): Toast[] {
  const [state, setState] = useState<Toast[]>(toasts);
  useEffect(() => {
    listeners.add(setState);
    return () => {
      listeners.delete(setState);
    };
  }, []);
  return state;
}
