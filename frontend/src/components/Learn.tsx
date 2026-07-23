import { useEffect, useState, type ReactNode } from "react";

type TabId = "apply" | "evaluations" | "monitoring" | "ab";

interface Quiz {
  q: string;
  options: string[];
  answer: number;
  why: string;
}

interface TryIt {
  label: string;
  hint: string;
  tab?: TabId;
  href?: string;
}

interface Module {
  id: string;
  title: string;
  minutes: number;
  blurb: string;
  body: ReactNode;
  tryIt?: TryIt;
  quiz: Quiz;
}

/** Plain-English, hands-on lessons for tracing & evaluation, tied to this app. */
export function Learn({ setView }: { setView: (v: TabId) => void }) {
  const [done, setDone] = useState<Record<string, boolean>>(() => {
    try {
      return JSON.parse(localStorage.getItem("learn_done") || "{}");
    } catch {
      return {};
    }
  });
  const [open, setOpen] = useState<string | null>("tracing-101");

  useEffect(() => {
    localStorage.setItem("learn_done", JSON.stringify(done));
  }, [done]);

  const modules = buildModules(setView);
  const completed = modules.filter((m) => done[m.id]).length;

  return (
    <div className="app">
      <header>
        <h1>Learn: Tracing, Observability &amp; Evaluation</h1>
        <p>
          Short, hands-on lessons on LangSmith, spans &amp; tracing, Phoenix, and
          RAGAS. Read the idea, try it live, then answer one quick check. Your
          progress is saved in this browser.
        </p>
        <div className="learn-progress">
          <div className="learn-progress-bar">
            <div
              className="learn-progress-fill"
              style={{ width: `${(completed / modules.length) * 100}%` }}
            />
          </div>
          <span className="muted">
            {completed} / {modules.length} done
          </span>
        </div>
      </header>

      {modules.map((m, i) => (
        <ModuleCard
          key={m.id}
          index={i + 1}
          module={m}
          isOpen={open === m.id}
          isDone={!!done[m.id]}
          onToggle={() => setOpen(open === m.id ? null : m.id)}
          onComplete={() => setDone((d) => ({ ...d, [m.id]: true }))}
        />
      ))}
    </div>
  );
}

function ModuleCard({
  index,
  module: m,
  isOpen,
  isDone,
  onToggle,
  onComplete,
}: {
  index: number;
  module: Module;
  isOpen: boolean;
  isDone: boolean;
  onToggle: () => void;
  onComplete: () => void;
}) {
  return (
    <div className="card learn-module">
      <div className="learn-head" onClick={onToggle}>
        <div>
          <div className="learn-eyebrow">
            Module {index} · {m.minutes} min {isDone && "· ✓ done"}
          </div>
          <h2 style={{ margin: "2px 0 4px" }}>{m.title}</h2>
          <p className="muted" style={{ margin: 0 }}>
            {m.blurb}
          </p>
        </div>
        <span className="learn-chevron">{isOpen ? "▲" : "▼"}</span>
      </div>

      {isOpen && (
        <div className="learn-body">
          {m.body}

          {m.tryIt && (
            <div className="learn-tryit">
              <div>
                <b>Try it</b>
                <p className="muted" style={{ margin: "4px 0 0" }}>
                  {m.tryIt.hint}
                </p>
              </div>
              {m.tryIt.href ? (
                <a
                  className="learn-tryit-btn"
                  href={m.tryIt.href}
                  target="_blank"
                  rel="noreferrer"
                >
                  {m.tryIt.label}
                </a>
              ) : (
                <button onClick={() => m.tryIt?.tab && setViewSafe(m.tryIt.tab)}>
                  {m.tryIt.label}
                </button>
              )}
            </div>
          )}

          <QuizBlock quiz={m.quiz} onPass={onComplete} />
        </div>
      )}
    </div>
  );
}

// setView is closed over per-module; kept simple via a module-scoped ref.
let _setView: (v: TabId) => void = () => {};
function setViewSafe(tab: TabId) {
  _setView(tab);
}

function QuizBlock({ quiz, onPass }: { quiz: Quiz; onPass: () => void }) {
  const [picked, setPicked] = useState<number | null>(null);
  const correct = picked === quiz.answer;

  return (
    <div className="learn-quiz">
      <div className="learn-quiz-q">Quick check: {quiz.q}</div>
      <div className="learn-quiz-opts">
        {quiz.options.map((opt, i) => {
          const state =
            picked === null
              ? ""
              : i === quiz.answer
                ? "right"
                : i === picked
                  ? "wrong"
                  : "";
          return (
            <button
              key={i}
              className={`learn-opt ${state}`}
              onClick={() => {
                setPicked(i);
                if (i === quiz.answer) onPass();
              }}
            >
              {opt}
            </button>
          );
        })}
      </div>
      {picked !== null && (
        <p className={`learn-feedback ${correct ? "ok" : "no"}`}>
          {correct ? "Correct. " : "Not quite. "}
          {quiz.why}
        </p>
      )}
    </div>
  );
}

function Term({ children }: { children: ReactNode }) {
  return <span className="learn-term">{children}</span>;
}

function buildModules(setView: (v: TabId) => void): Module[] {
  _setView = setView;
  return [
    {
      id: "tracing-101",
      title: "Tracing & spans 101 — the flight recorder for AI",
      minutes: 5,
      blurb:
        "What a trace and a span are, why every AI app needs them, and what's recorded at each step.",
      body: (
        <>
          <p>
            When this app handles one request — say "recommend properties for
            Alex" — a lot happens under the hood: read the PDF, search a vector
            store, call the LLM, query the graph, call the LLM again. <b>Tracing</b>{" "}
            records all of it, like a flight recorder (black box) for that one
            request, so you can see exactly what happened and where time/money
            went.
          </p>
          <ul className="learn-list">
            <li>
              A <Term>trace</Term> = the whole story of one request, start to
              finish.
            </li>
            <li>
              A <Term>span</Term> = one step inside that story (e.g. "retrieve
              chunks", "call Claude"). Spans nest: a parent span can contain
              child spans, forming a tree you can expand.
            </li>
            <li>
              Each span carries <Term>attributes</Term>: its inputs, its output,
              how long it took (<i>latency</i>), how many tokens it used (which
              drives <i>cost</i>), and any error.
            </li>
          </ul>
          <p>
            Why bother? Without traces, a wrong or slow answer is a black box.
            With them you can pinpoint <i>which</i> step retrieved the wrong
            context, <i>which</i> LLM call was slow, and <i>what</i> exact prompt
            was sent. It's the difference between "the AI is acting weird" and
            "the retrieval step returned empty context on span #3."
          </p>
          <p>
            This app uses <b>two</b> tracing tools, on purpose — each is best at
            the framework it instruments:
          </p>
          <ul className="learn-list">
            <li>
              <Term>LangSmith</Term> traces the <b>LangChain</b> side
              (recommendations, eligibility, graph questions).
            </li>
            <li>
              <Term>Phoenix</Term> traces the <b>LlamaIndex</b> side (the PDF
              question-answering / RAG).
            </li>
          </ul>
          <div className="learn-callout">
            Trace = the whole request. Span = one step in it. Attributes = the
            inputs, outputs, latency, tokens, and errors on each step.
          </div>
        </>
      ),
      tryIt: {
        label: "Open Apply →",
        tab: "apply",
        hint: "Pick a sample applicant to fire a real request. That single click generates a trace (with many spans) in LangSmith and Phoenix behind the scenes.",
      },
      quiz: {
        q: "What's the difference between a trace and a span?",
        options: [
          "A trace is one step; a span is the whole request",
          "A trace is the whole request; a span is one step inside it",
          "They're two words for the same thing",
        ],
        answer: 1,
        why: "A trace is the end-to-end story of one request; spans are the individual nested steps within it (retrieve, LLM call, etc.).",
      },
    },
    {
      id: "langsmith",
      title: "LangSmith — tracing the LangChain side (+ experiments)",
      minutes: 6,
      blurb:
        "How LangSmith captures traces here, how it's switched on, and what a tracked 'experiment' is.",
      body: (
        <>
          <p>
            <Term>LangSmith</Term> is the trace + evaluation dashboard for the
            LangChain parts of this app. Every recommendation, eligibility
            decision, and graph question becomes a trace you can open and inspect
            span-by-span: the prompt sent to Claude, the response, token counts,
            latency, and any error.
          </p>
          <p>
            <b>How it's turned on (it's just env vars).</b> LangChain auto-sends
            traces when these are set — no extra code in the business logic:
          </p>
          <ul className="learn-list">
            <li>
              <code>LANGCHAIN_TRACING_V2=true</code> — flip tracing on.
            </li>
            <li>
              <code>LANGCHAIN_API_KEY</code> — your LangSmith key (in this repo it
              comes from <code>.env</code>).
            </li>
            <li>
              <code>LANGCHAIN_PROJECT</code> — which project the traces land in.
            </li>
          </ul>
          <p>
            The <b>Claude</b> badge at the top of Apply and the{" "}
            <b>LangSmith</b> badge tell you whether real keys are present. With no
            key, the app falls back to a mock LLM and simply skips tracing — it
            never breaks.
          </p>
          <p>
            <b>Experiments — eval that's tracked over time.</b> Beyond live
            traces, LangSmith can run a whole dataset as an <Term>experiment</Term>:
            a versioned, shareable scorecard. This app pushes its golden{" "}
            <i>eligibility</i> dataset, runs the deterministic rule engine as the
            "system under test", and scores each row with a custom evaluator
            (<code>verdict_exact_match</code>). The result is a run you can compare
            against future runs to spot drift — the core "is the new version still
            correct?" workflow.
          </p>
          <div className="learn-callout">
            LangSmith = traces (one request, inspectable) <i>plus</i> experiments
            (a whole dataset scored and tracked over time).
          </div>
        </>
      ),
      tryIt: {
        label: "Open Evaluations →",
        tab: "evaluations",
        hint: "In the LLM tier, click 'Push LangSmith experiment'. It uploads the eligibility dataset and scores it — then open LangSmith to compare runs. (Skips cleanly if no key is set.)",
      },
      quiz: {
        q: "How is LangSmith tracing switched on in this app?",
        options: [
          "By rewriting every function to log manually",
          "By setting environment variables like LANGCHAIN_TRACING_V2 that LangChain reads automatically",
          "By installing a browser extension",
        ],
        answer: 1,
        why: "LangChain emits traces automatically when the LANGCHAIN_* env vars (tracing flag, API key, project) are present — no changes to the business logic.",
      },
    },
    {
      id: "phoenix",
      title: "Phoenix — tracing the RAG (LlamaIndex) side, locally",
      minutes: 5,
      blurb:
        "A local trace UI for the PDF question-answering pipeline, powered by OpenTelemetry.",
      body: (
        <>
          <p>
            <Term>Arize Phoenix</Term> is an open-source trace viewer that runs{" "}
            <b>locally</b> (no cloud account, no key) at{" "}
            <code>http://localhost:6006</code>. In this app it watches the{" "}
            <b>LlamaIndex</b> RAG pipeline: turning a PDF into chunks, embedding
            and storing them, retrieving the relevant ones for a question, and
            generating the grounded answer.
          </p>
          <p>
            <b>How it works.</b> Phoenix uses <Term>OpenTelemetry</Term> — an
            industry-standard way for software to emit traces. The app registers a
            Phoenix collector and "instruments" LlamaIndex, so every RAG call
            automatically sends its spans there:
          </p>
          <ul className="learn-list">
            <li>
              a <b>retrieve</b> span — which chunks were pulled, and their
              similarity scores;
            </li>
            <li>
              an <b>LLM</b> span — the exact prompt (question + retrieved
              context) and the answer;
            </li>
            <li>
              timings and token usage on each.
            </li>
          </ul>
          <p>
            This is gold for debugging RAG: if an answer is wrong, you open the
            trace and check the <b>retrieve</b> span first — usually the model
            answered fine but was handed the wrong context.
          </p>
          <div className="learn-callout">
            Two lenses, by design: <b>Phoenix</b> for the LlamaIndex/RAG side
            (local), <b>LangSmith</b> for the LangChain side (cloud). Same app,
            best tool per framework.
          </div>
        </>
      ),
      tryIt: {
        label: "Open Phoenix (localhost:6006) →",
        href: "http://localhost:6006",
        hint: "Ask a question in the Apply page's chat first, then open Phoenix and expand the trace. Look at the 'retrieve' span to see which PDF chunks were used.",
      },
      quiz: {
        q: "An answer from the PDF chat is wrong. In Phoenix, what should you check first?",
        options: [
          "The LLM span's wording",
          "The retrieve span — to see whether the right context was pulled",
          "The browser console",
        ],
        answer: 1,
        why: "Most RAG errors are retrieval errors: the model is fine but got the wrong context. The retrieve span shows exactly which chunks were used.",
      },
    },
    {
      id: "ragas",
      title: "RAGAS — scoring how good the RAG answers are",
      minutes: 6,
      blurb:
        "Faithfulness, answer relevancy, and answer correctness — what they mean and how they're judged.",
      body: (
        <>
          <p>
            Tracing shows you <i>what happened</i>; <Term>RAGAS</Term> tells you{" "}
            <i>how good it was</i>. RAGAS is a toolkit that scores
            retrieval-augmented answers using an LLM as the grader. This app runs
            it over a gold set of PDF questions and reports three 0–100% scores:
          </p>
          <ul className="learn-list">
            <li>
              <Term>Faithfulness</Term> — did the answer stick to the retrieved
              context, or did it make things up? (Low = hallucination.)
            </li>
            <li>
              <Term>Answer relevancy</Term> — is the answer actually on-topic for
              the question asked?
            </li>
            <li>
              <Term>Answer correctness</Term> — does it match the known correct
              answer? (This one needs a reference / gold answer.)
            </li>
          </ul>
          <p>
            <b>It needs an LLM to judge</b>, so RAGAS only runs when an
            Anthropic key is present, and it skips cleanly otherwise (so CI never
            breaks). One quirk worth knowing: RAGAS is run in a separate
            background process here, because its async library clashes with the
            web server's event loop — a nice real-world example of integration
            friction.
          </p>
          <p>
            Faithfulness vs correctness is a subtle, important pair: an answer can
            be <i>faithful</i> (true to the retrieved context) but still{" "}
            <i>incorrect</i> if the wrong context was retrieved — which is exactly
            why you pair RAGAS scores with Phoenix traces.
          </p>
          <div className="learn-callout">
            Faithful = doesn't invent beyond the context. Relevant = answers the
            question. Correct = matches the gold answer.
          </div>
        </>
      ),
      tryIt: {
        label: "Open Evaluations →",
        tab: "evaluations",
        hint: "In the LLM tier, click 'Run RAGAS'. It indexes the sample PDFs, answers the gold questions, and scores faithfulness / relevancy / correctness. (Needs an Anthropic key.)",
      },
      quiz: {
        q: "An answer is faithful to its retrieved context but still graded incorrect. What likely went wrong?",
        options: [
          "RAGAS is broken",
          "The wrong context was retrieved — faithful to bad context is still a wrong answer",
          "The LLM ran out of tokens",
        ],
        answer: 1,
        why: "Faithfulness only checks the answer against whatever context was retrieved. If retrieval pulled the wrong context, the answer can be faithful yet incorrect — check the retrieve span in Phoenix.",
      },
    },
    {
      id: "llm-judge",
      title: "LLM-as-a-judge — grading prose, with guardrails",
      minutes: 5,
      blurb:
        "Using an LLM to score free-text explanations, why guardrails matter, and the real bug it caught.",
      body: (
        <>
          <p>
            Some quality questions can't be checked by a rule — like "is this
            written recommendation actually backed by the property's facts?" For
            those, this app uses an <Term>LLM-as-a-judge</Term>: a second LLM call
            that reads an explanation and scores its <Term>groundedness</Term>{" "}
            (1–5) against the real facts it was given.
          </p>
          <p>
            An LLM grading an LLM is only trustworthy with <b>guardrails</b>:
          </p>
          <ul className="learn-list">
            <li>
              <b>Temperature 0</b> — repeatable, near-deterministic grades.
            </li>
            <li>
              <b>JSON-only output</b> that's parsed, validated, and clamped (a bad
              score becomes empty, never a crash).
            </li>
            <li>
              <b>Reference-anchored</b> — the judge only sees the facts we pass,
              so it grades against ground truth, not its own world knowledge.
            </li>
            <li>
              <b>Scoped to prose</b> — it never changes a verdict or a ranking
              (those stay deterministic); it only scores the wording.
            </li>
            <li>
              <b>Backed by a deterministic tripwire</b> — a cheap regex also flags
              any amenity the prose claims but the property lacks.
            </li>
          </ul>
          <p>
            <b>The real bug it caught:</b> recommendation explanations were
            inventing amenities — a gym, "downtown views," in-unit laundry — the
            properties didn't have. Groundedness scored a dismal <b>0.30</b>. The
            fix was to feed the explainer and the judge the <i>same</i> real
            facts; groundedness jumped to <b>0.93</b> with zero invented claims.
            That's the whole loop: <b>measure → find → fix → measure again.</b>
          </p>
        </>
      ),
      tryIt: {
        label: "Open Evaluations →",
        tab: "evaluations",
        hint: "In the LLM tier, click 'Run LLM judge' to score groundedness, then read the per-case 'Judge notes' to see exactly what it flagged.",
      },
      quiz: {
        q: "Why is the LLM judge only allowed to score the free-text explanation, not the verdict or ranking?",
        options: [
          "To save tokens",
          "So the actual decisions stay deterministic and reproducible — only the prose is judged",
          "Because LLMs can't read numbers",
        ],
        answer: 1,
        why: "Verdicts and rankings stay rule-based for reproducibility. The judge is scoped to the prose, so a noisy LLM grade can never alter a real decision.",
      },
    },
  ];
}
