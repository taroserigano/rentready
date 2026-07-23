import type { ReactNode } from "react";

/**
 * Minimal, dependency-free markdown-lite renderer for LLM chat answers.
 * Covers the subset our prompts actually produce: headings (#/##/###), bold,
 * italic, inline code, blockquotes (>), unordered/ordered lists, and
 * paragraphs. Anything unrecognized just falls through as plain text --
 * never throws on unexpected input.
 */

type Block = { type: "h1" | "h2" | "h3" | "quote" | "ul" | "ol" | "p"; lines: string[] };

const HEADING_RE = /^(#{1,3})\s+(.*)$/;
const QUOTE_RE = /^>\s?(.*)$/;
const UL_RE = /^[-*]\s+(.*)$/;
const OL_RE = /^\d+\.\s+(.*)$/;

function parseBlocks(text: string): Block[] {
  const lines = text.replace(/\r\n/g, "\n").split("\n");
  const blocks: Block[] = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (line.trim() === "") {
      i++;
      continue;
    }
    const heading = HEADING_RE.exec(line);
    if (heading) {
      blocks.push({ type: `h${heading[1].length}` as "h1" | "h2" | "h3", lines: [heading[2]] });
      i++;
      continue;
    }
    if (QUOTE_RE.test(line)) {
      const buf: string[] = [];
      while (i < lines.length && QUOTE_RE.test(lines[i])) {
        buf.push(lines[i].replace(QUOTE_RE, "$1"));
        i++;
      }
      blocks.push({ type: "quote", lines: buf });
      continue;
    }
    if (UL_RE.test(line)) {
      const buf: string[] = [];
      while (i < lines.length && UL_RE.test(lines[i])) {
        buf.push(lines[i].replace(UL_RE, "$1"));
        i++;
      }
      blocks.push({ type: "ul", lines: buf });
      continue;
    }
    if (OL_RE.test(line)) {
      const buf: string[] = [];
      while (i < lines.length && OL_RE.test(lines[i])) {
        buf.push(lines[i].replace(OL_RE, "$1"));
        i++;
      }
      blocks.push({ type: "ol", lines: buf });
      continue;
    }
    const buf: string[] = [];
    while (
      i < lines.length &&
      lines[i].trim() !== "" &&
      !HEADING_RE.test(lines[i]) &&
      !QUOTE_RE.test(lines[i]) &&
      !UL_RE.test(lines[i]) &&
      !OL_RE.test(lines[i])
    ) {
      buf.push(lines[i]);
      i++;
    }
    blocks.push({ type: "p", lines: buf });
  }
  return blocks;
}

const INLINE_RE = /(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*|_[^_]+_)/g;

function renderInline(text: string, keyPrefix: string): ReactNode[] {
  const nodes: ReactNode[] = [];
  let lastIndex = 0;
  let i = 0;
  let m: RegExpExecArray | null;
  INLINE_RE.lastIndex = 0;
  while ((m = INLINE_RE.exec(text))) {
    if (m.index > lastIndex) nodes.push(text.slice(lastIndex, m.index));
    const token = m[0];
    const key = `${keyPrefix}-${i++}`;
    if (token.startsWith("**")) nodes.push(<strong key={key}>{token.slice(2, -2)}</strong>);
    else if (token.startsWith("`")) nodes.push(<code key={key}>{token.slice(1, -1)}</code>);
    else nodes.push(<em key={key}>{token.slice(1, -1)}</em>);
    lastIndex = m.index + token.length;
  }
  if (lastIndex < text.length) nodes.push(text.slice(lastIndex));
  return nodes;
}

export function Markdown({ text }: { text: string }) {
  if (!text) return null;
  const blocks = parseBlocks(text);
  return (
    <>
      {blocks.map((b, bi) => {
        const key = `b${bi}`;
        switch (b.type) {
          case "h1":
            return <h3 key={key} className="md-h">{renderInline(b.lines[0], key)}</h3>;
          case "h2":
            return <h4 key={key} className="md-h">{renderInline(b.lines[0], key)}</h4>;
          case "h3":
            return <h5 key={key} className="md-h">{renderInline(b.lines[0], key)}</h5>;
          case "quote":
            return (
              <blockquote key={key} className="md-quote">
                {b.lines.map((l, li) => (
                  <p key={li}>{renderInline(l, `${key}-${li}`)}</p>
                ))}
              </blockquote>
            );
          case "ul":
            return (
              <ul key={key} className="md-list">
                {b.lines.map((l, li) => (
                  <li key={li}>{renderInline(l, `${key}-${li}`)}</li>
                ))}
              </ul>
            );
          case "ol":
            return (
              <ol key={key} className="md-list">
                {b.lines.map((l, li) => (
                  <li key={li}>{renderInline(l, `${key}-${li}`)}</li>
                ))}
              </ol>
            );
          default:
            return <p key={key} className="md-p">{renderInline(b.lines.join(" "), key)}</p>;
        }
      })}
    </>
  );
}
