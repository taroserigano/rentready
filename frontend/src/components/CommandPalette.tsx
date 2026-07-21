import { useEffect, useMemo, useRef, useState } from "react";

export interface Command {
  id: string;
  label: string;
  group: string;
  run: () => void;
}

/**
 * ⌘K / Ctrl-K command palette. Self-contained: owns its open state + a global
 * hotkey, renders the passed commands, and supports arrow/enter/esc.
 */
export function CommandPalette({ commands }: { commands: Command[] }) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [sel, setSel] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setOpen((v) => !v);
      } else if (e.key === "Escape") {
        setOpen(false);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  useEffect(() => {
    if (open) {
      setQuery("");
      setSel(0);
      setTimeout(() => inputRef.current?.focus(), 0);
    }
  }, [open]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return commands;
    return commands.filter(
      (c) =>
        c.label.toLowerCase().includes(q) || c.group.toLowerCase().includes(q),
    );
  }, [commands, query]);

  useEffect(() => {
    if (sel >= filtered.length) setSel(0);
  }, [filtered, sel]);

  if (!open) return null;

  const run = (c?: Command) => {
    if (!c) return;
    setOpen(false);
    c.run();
  };

  return (
    <div
      className="cmdk-backdrop"
      onClick={() => setOpen(false)}
      role="dialog"
      aria-modal="true"
      aria-label="Command palette"
    >
      <div className="cmdk-panel" onClick={(e) => e.stopPropagation()}>
        <input
          ref={inputRef}
          className="cmdk-input"
          placeholder="Type a command… (samples, pages, theme)"
          value={query}
          role="combobox"
          aria-expanded="true"
          aria-controls="cmdk-list"
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "ArrowDown") {
              e.preventDefault();
              setSel((s) => Math.min(s + 1, filtered.length - 1));
            } else if (e.key === "ArrowUp") {
              e.preventDefault();
              setSel((s) => Math.max(s - 1, 0));
            } else if (e.key === "Enter") {
              e.preventDefault();
              run(filtered[sel]);
            }
          }}
        />
        <div className="cmdk-list" id="cmdk-list" role="listbox">
          {filtered.length === 0 ? (
            <div className="cmdk-empty">No commands match “{query}”.</div>
          ) : (
            filtered.map((c, i) => (
              <div
                key={c.id}
                className="cmdk-item"
                role="option"
                aria-selected={i === sel}
                onMouseEnter={() => setSel(i)}
                onClick={() => run(c)}
              >
                {c.label}
                <span className="cmdk-group">{c.group}</span>
              </div>
            ))
          )}
        </div>
        <div className="cmdk-hint">
          <kbd>↑</kbd> <kbd>↓</kbd> navigate · <kbd>↵</kbd> run · <kbd>esc</kbd> close
        </div>
      </div>
    </div>
  );
}
