import { useState } from "react";
import type { FormEvent } from "react";
import {
  Activity,
  ArrowUp,
  BarChart3,
  Bot,
  ChevronRight,
  CircleHelp,
  Clock3,
  FileText,
  Inbox,
  Layers3,
  Menu,
  MessageSquare,
  Search,
  Settings,
  ShieldCheck,
  Sparkles,
  UserRound,
  X,
} from "lucide-react";
import { analyzeMessage } from "./lib/api";
import type { AnnaResult } from "./lib/api";
import "./index.css";

const quickActions = [
  "Driver issue",
  "Payment problem",
  "Refund request",
  "Account access",
  "Lost item",
];

const history = [
  "Driver cancelled my ride",
  "Unexpected charge",
  "Can't access my account",
  "Lost item after a trip",
];

function AnnaOrb({ analyzing = false }: { analyzing?: boolean }) {
  return (
    <div className={`orb-shell ${analyzing ? "orb-analyzing" : ""}`}>
      <div className="orb-aura aura-one" />
      <div className="orb-aura aura-two" />
      <div className="anna-orb">
        <div className="orb-core" />
        <div className="orb-highlight" />
      </div>
    </div>
  );
}

function Sidebar({
  onNewRequest,
  active,
}: {
  onNewRequest: () => void;
  active: string;
}) {
  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-mark">
          <Sparkles size={16} />
        </div>
        <span>ANNA</span>
        <span className="brand-sub">AI Support</span>
      </div>

      <div className="search-box">
        <Search size={15} />
        <span>Search</span>
        <kbd>⌘ K</kbd>
      </div>

      <button className="new-request" onClick={onNewRequest}>
        <MessageSquare size={16} />
        <span>New request</span>
      </button>

      <nav className="nav-section">
        <div className="nav-label">WORKSPACE</div>
        <button className={active === "Inbox" ? "nav-item active" : "nav-item"}>
          <Inbox size={16} />
          Inbox
          <span className="nav-count">24</span>
        </button>
        <button className="nav-item">
          <Clock3 size={16} />
          Needs review
          <span className="nav-count">7</span>
        </button>
        <button className="nav-item">
          <ShieldCheck size={16} />
          Auto-handled
        </button>
        <button className="nav-item">
          <Activity size={16} />
          Escalated
        </button>
      </nav>

      <nav className="nav-section">
        <div className="nav-label">INTELLIGENCE</div>
        <button className="nav-item">
          <BarChart3 size={16} />
          Analytics
        </button>
        <button className="nav-item">
          <Layers3 size={16} />
          Evaluation
        </button>
        <button className="nav-item">
          <FileText size={16} />
          Failures
        </button>
        <button className="nav-item">
          <Bot size={16} />
          Knowledge
        </button>
      </nav>

      <div className="sidebar-bottom">
        <button className="nav-item">
          <Settings size={16} />
          Settings
        </button>
        <div className="profile">
          <div className="avatar">
            <UserRound size={15} />
          </div>
          <div>
            <strong>Support workspace</strong>
            <span>Uber · ANNA</span>
          </div>
        </div>
      </div>
    </aside>
  );
}

function Landing({
  onAnalyze,
  loading,
}: {
  onAnalyze: (message: string) => void;
  loading: boolean;
}) {
  const [message, setMessage] = useState("");

  const submit = (event: FormEvent) => {
    event.preventDefault();

    if (message.trim() && !loading) {
      onAnalyze(message.trim());
    }
  };

  return (
    <main className="landing">
      <div className="ambient ambient-one" />
      <div className="ambient ambient-two" />

      <div className="landing-content">
        <div className="eyebrow">
          <span className="status-dot" />
          ANNA is ready
        </div>

        <AnnaOrb analyzing={loading} />

        <h1>{loading ? "ANNA is analyzing" : "How can I help?"}</h1>

        <p className="landing-description">
          Understand customer issues, find relevant support history, and draft
          a grounded response.
        </p>

        {loading ? (
          <div className="analysis-progress">
            <div className="progress-step active">
              <span>01</span>
              Understanding request
            </div>
            <div className="progress-step active">
              <span>02</span>
              Finding historical cases
            </div>
            <div className="progress-step active">
              <span>03</span>
              Checking evidence
            </div>
            <div className="progress-step">
              <span>04</span>
              Generating response
            </div>
          </div>
        ) : (
          <>
            <form className="message-box" onSubmit={submit}>
              <textarea
                value={message}
                onChange={(event) => setMessage(event.target.value)}
                placeholder="Ask ANNA about a customer issue..."
                rows={2}
              />

              <button
                className="send-button"
                type="submit"
                disabled={!message.trim()}
                aria-label="Analyze request"
              >
                <ArrowUp size={18} />
              </button>
            </form>

            <div className="quick-actions">
              {quickActions.map((action) => (
                <button
                  key={action}
                  onClick={() => setMessage(action)}
                  className="quick-chip"
                >
                  {action}
                </button>
              ))}
            </div>
          </>
        )}
      </div>

      {!loading && (
        <div className="history-strip">
          <span>Recent requests</span>

          {history.map((item) => (
            <button key={item} onClick={() => setMessage(item)}>
              {item}
              <ChevronRight size={13} />
            </button>
          ))}
        </div>
      )}
    </main>
  );
}

function EvidenceCard({ result }: { result: AnnaResult }) {
  const evidencePercent = Math.round(result.evidence.score * 100);

  return (
    <section className="intelligence-section">
      <div className="section-heading">
        <div>
          <span className="section-kicker">Evidence</span>
          <h3>Historical precedent</h3>
        </div>

        <span className="case-count">
          {result.retrieved_cases.length} cases
        </span>
      </div>

      <div className="evidence-score">
        <div className="score-top">
          <span>Evidence strength</span>
          <strong>{evidencePercent}%</strong>
        </div>

        <div className="score-track">
          <div style={{ width: `${evidencePercent}%` }} />
        </div>

        <div className="score-meta">
          <span>
            {Math.round(result.evidence.similarity_strength * 100)}% similarity
          </span>
          <span>
            {Math.round(result.evidence.resolution_agreement * 100)}% resolution
            agreement
          </span>
        </div>
      </div>

      <div className="case-list">
        {result.retrieved_cases.slice(0, 3).map((item) => (
          <div className="evidence-card" key={item.case_id}>
            <div className="evidence-card-top">
              <span className="similarity">
                {Math.round(item.similarity * 100)}% match
              </span>
              <span>{item.resolution_type.replaceAll("_", " ")}</span>
            </div>

            <p>{item.customer_text}</p>

            <div className="historical-response">
              <span>Historical response</span>
              <p>{item.historical_response}</p>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

function AnalysisView({
  result,
  onNewRequest,
}: {
  result: AnnaResult;
  onNewRequest: () => void;
}) {
  const [showEvidence, setShowEvidence] = useState(true);

  return (
    <main className="analysis-view">
      <header className="analysis-header">
        <div>
          <button className="back-button" onClick={onNewRequest}>
            <ChevronRight size={15} className="rotate-180" />
            New request
          </button>

          <h1>Customer support analysis</h1>
        </div>

        <div className="header-badge">
          <span className="status-dot" />
          Uber Support
        </div>
      </header>

      <div className="analysis-grid">
        <section className="conversation-panel">
          <div className="panel-heading">
            <div>
              <span className="section-kicker">Customer request</span>
              <h2>Support conversation</h2>
            </div>

            <button className="icon-button">
              <Menu size={17} />
            </button>
          </div>

          <div className="conversation">
            <div className="message customer-message">
              <div className="message-avatar">
                <UserRound size={15} />
              </div>

              <div>
                <div className="message-author">Customer</div>

                <div className="message-bubble customer-bubble">
                  {result.customer_message}
                </div>
              </div>
            </div>

            <div className="message anna-message">
              <div className="message-avatar anna-avatar">
                <Sparkles size={15} />
              </div>

              <div className="draft-area">
                <div className="message-author">
                  ANNA <span>AI draft</span>
                </div>

                <div className="message-bubble anna-bubble">
                  {result.generation.reply}
                </div>

                <div className="draft-actions">
                  <button>Use response</button>
                  <button>Edit</button>
                  <button>Regenerate</button>
                </div>
              </div>
            </div>
          </div>

          <div className="composer">
            <textarea
              placeholder="Edit ANNA's response before sending..."
              defaultValue={result.generation.reply}
            />

            <div className="composer-bottom">
              <span>
                Draft generated from historical support evidence
              </span>

              <button>Send response</button>
            </div>
          </div>
        </section>

        <aside className="intelligence-panel">
          <div className="panel-heading">
            <div>
              <span className="section-kicker">ANNA intelligence</span>
              <h2>Analysis</h2>
            </div>

            <Sparkles size={18} />
          </div>

          <section className="intent-card">
            <div className="intent-orb">
              <Sparkles size={17} />
            </div>

            <div>
              <span>Detected intent</span>
              <strong>{result.intent.replaceAll("_", " ")}</strong>
            </div>

            <div className="confidence">
              {Math.round(result.intent_confidence * 100)}%
            </div>
          </section>

          <section className="decision-card">
            <div className="decision-header">
              <span className="section-kicker">Automation decision</span>

              <span
                className={
                  result.decision.auto_handle
                    ? "decision-pill auto"
                    : "decision-pill review"
                }
              >
                {result.decision.auto_handle
                  ? "AUTO-HANDLE"
                  : "HUMAN REVIEW"}
              </span>
            </div>

            <p>{result.decision.reason}</p>

            <div className="decision-factors">
              <div>
                <span>Intent confidence</span>
                <strong>
                  {Math.round(result.intent_confidence * 100)}%
                </strong>
              </div>

              <div>
                <span>Top similarity</span>
                <strong>
                  {Math.round(
                    Number(result.decision.factors.top_similarity) * 100
                  )}
                  %
                </strong>
              </div>

              <div>
                <span>Evidence strength</span>
                <strong>{Math.round(result.evidence.score * 100)}%</strong>
              </div>
            </div>
          </section>

          <button
            className="evidence-toggle"
            onClick={() => setShowEvidence(!showEvidence)}
          >
            <span>
              <ShieldCheck size={15} />
              Historical evidence
            </span>

            <ChevronRight
              size={15}
              className={showEvidence ? "rotate-90" : ""}
            />
          </button>

          {showEvidence && <EvidenceCard result={result} />}

          <section className="capability-note">
            <CircleHelp size={15} />

            <p>
              ANNA uses historical Twitter support evidence. It cannot access
              private accounts, current trips, payments, refunds, or internal
              Uber systems.
            </p>
          </section>
        </aside>
      </div>
    </main>
  );
}

function App() {
  const [result, setResult] = useState<AnnaResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleAnalyze = async (message: string) => {
    setLoading(true);
    setError("");

    try {
      const data = await analyzeMessage(message);
      setResult(data);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Unable to reach ANNA."
      );
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setResult(null);
    setError("");
  };

  return (
    <div className="app-shell">
      <Sidebar onNewRequest={reset} active={result ? "Inbox" : ""} />

      {result ? (
        <AnalysisView result={result} onNewRequest={reset} />
      ) : (
        <Landing onAnalyze={handleAnalyze} loading={loading} />
      )}

      {error && (
        <div className="error-toast">
          <X size={15} />
          <span>{error}</span>
          <button onClick={() => setError("")}>Dismiss</button>
        </div>
      )}
    </div>
  );
}

export default App;
