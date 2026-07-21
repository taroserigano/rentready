import { useCallback, useEffect, useState } from "react";

/**
 * Shortlist of saved property ids, persisted to localStorage.
 * Mirrors the try/catch guard used by ThemeToggle in App.tsx so private
 * mode / disabled storage degrades to in-memory state instead of throwing.
 */
const KEY = "rr-saved";

function load(): Set<string> {
  try {
    const raw = localStorage.getItem(KEY);
    if (raw) return new Set(JSON.parse(raw) as string[]);
  } catch {
    /* private mode / storage disabled — start empty */
  }
  return new Set();
}

export function useSavedProperties() {
  const [saved, setSaved] = useState<Set<string>>(load);

  useEffect(() => {
    try {
      localStorage.setItem(KEY, JSON.stringify([...saved]));
    } catch {
      /* ignore — keep in-memory */
    }
  }, [saved]);

  const toggle = useCallback((id: string) => {
    setSaved((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const isSaved = useCallback((id: string) => saved.has(id), [saved]);

  return { saved, toggle, isSaved, count: saved.size };
}
