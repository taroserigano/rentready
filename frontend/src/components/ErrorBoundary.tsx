import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
  children: ReactNode;
}

interface State {
  error: Error | null;
}

/**
 * Top-level render-error catch-all. Without this, any component throwing
 * during render (a bad response shape, a null-ref bug, anything) blanks the
 * ENTIRE app to a white screen with no recovery — error boundaries are the
 * only mechanism React gives you for this (hooks can't catch render errors).
 */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("Unhandled render error:", error, info.componentStack);
  }

  render() {
    if (this.state.error) {
      return (
        <div className="app">
          <div className="card" style={{ marginTop: 48, textAlign: "center" }}>
            <h2>Something went wrong</h2>
            <p className="muted" style={{ marginTop: 8 }}>
              The page hit an unexpected error. Reloading usually clears it.
            </p>
            <div className="error" role="alert" style={{ marginTop: 12 }}>
              {this.state.error.message}
            </div>
            <button
              className="btn-small btn-ghost"
              style={{ marginTop: 16 }}
              onClick={() => window.location.reload()}
            >
              Reload
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
