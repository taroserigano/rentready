import { useEffect, useRef, useState } from "react";
// The approved showcase page, verbatim. Kept as a standalone HTML document and
// rendered inside a sandboxed iframe so its self-contained styles/scripts never
// collide with the app's global CSS (and vice-versa).
import reportBody from "./aiReport.html?raw";

/** Current app theme from the <html data-theme> the ThemeToggle maintains. */
function currentTheme(): string {
  return document.documentElement.getAttribute("data-theme") || "dark";
}

/**
 * "AI Report" page — hosts the self-contained system-report HTML in an iframe.
 * Auto-sizes to its content (no inner scrollbar) and mirrors the app's
 * light/dark theme live via postMessage (no reload flash on toggle).
 */
export function AiReport() {
  const frameRef = useRef<HTMLIFrameElement>(null);
  const [height, setHeight] = useState(2400);

  // Wrap the report body as a full document. The injected script reports its
  // height to the parent and applies theme changes pushed from the app.
  const srcDoc = `<!doctype html><html data-theme="${currentTheme()}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"></head><body style="margin:0">${reportBody}<script>
    (function(){
      function postHeight(){
        parent.postMessage({ __aiReport: "height", value: document.documentElement.scrollHeight }, "*");
      }
      window.addEventListener("load", postHeight);
      if (window.ResizeObserver) new ResizeObserver(postHeight).observe(document.body);
      setTimeout(postHeight, 250);
      window.addEventListener("message", function(e){
        if (e.data && e.data.__aiReport === "theme" && e.data.value) {
          document.documentElement.setAttribute("data-theme", e.data.value);
          setTimeout(postHeight, 60);
        }
      });
    })();
  <\/script></body></html>`;

  // Grow the iframe to fit its content.
  useEffect(() => {
    function onMsg(e: MessageEvent) {
      const d = e.data as { __aiReport?: string; value?: number };
      if (d && d.__aiReport === "height" && typeof d.value === "number") {
        setHeight(Math.max(600, d.value));
      }
    }
    window.addEventListener("message", onMsg);
    return () => window.removeEventListener("message", onMsg);
  }, []);

  // Mirror the app's theme toggle into the iframe, live.
  useEffect(() => {
    const push = () => {
      frameRef.current?.contentWindow?.postMessage(
        { __aiReport: "theme", value: currentTheme() },
        "*",
      );
    };
    const obs = new MutationObserver(push);
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => obs.disconnect();
  }, []);

  return (
    <iframe
      ref={frameRef}
      title="RentReady AI — System Report"
      srcDoc={srcDoc}
      onLoad={() =>
        frameRef.current?.contentWindow?.postMessage(
          { __aiReport: "theme", value: currentTheme() },
          "*",
        )
      }
      style={{ width: "100%", border: "none", display: "block", height }}
    />
  );
}
