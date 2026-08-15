import { Activity, BarChart3, Gauge, RotateCcw } from "lucide-react";
import { useCallback, useState } from "react";
import { api } from "./api/client";
import { MetricCard } from "./components/MetricCard";
import { StatusBadge } from "./components/StatusBadge";
import { usePolling } from "./hooks/usePolling";
import type { OpsModels, OpsOverview, OpsSection } from "./types";
import { formatValue } from "./utils/format";

type View = "overview" | "models" | "monitoring" | "operations";

const tabs: Array<{ id: View; label: string; icon: typeof Gauge }> = [
  { id: "overview", label: "Overview", icon: Gauge },
  { id: "models", label: "Models & Lifecycle", icon: BarChart3 },
  { id: "monitoring", label: "Monitoring", icon: Activity },
  { id: "operations", label: "Operations & Resilience", icon: RotateCcw }
];

export function App() {
  const [view, setView] = useState<View>("overview");
  const loadOverview = useCallback(() => api.overview(), []);
  const loadModels = useCallback(() => api.models(), []);
  const loadMonitoring = useCallback(() => api.monitoring(), []);
  const loadRetraining = useCallback(() => api.retraining(), []);
  const loadResilience = useCallback(() => api.resilience(), []);
  const overview = usePolling(loadOverview, 10000);
  const models = usePolling(loadModels, 15000);
  const monitoring = usePolling(loadMonitoring, 10000);
  const retraining = usePolling(loadRetraining, 15000);
  const resilience = usePolling(loadResilience, 15000);

  return (
    <main className="shell">
      <aside className="sidebar">
        <div className="brand">SentinelML</div>
        <nav>
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                className={view === tab.id ? "active" : ""}
                onClick={() => setView(tab.id)}
                title={tab.label}
              >
                <Icon size={18} />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </nav>
        <div className="links">
          <a href="http://localhost:5000" target="_blank" rel="noreferrer">
            Open MLflow
          </a>
          <a href="http://localhost:3000" target="_blank" rel="noreferrer">
            Open Grafana
          </a>
        </div>
      </aside>
      <section className="content">
        {view === "overview" && (
          <Overview data={overview.data} error={overview.error} />
        )}
        {view === "models" && (
          <ModelsLifecycle data={models.data} error={models.error} />
        )}
        {view === "monitoring" && (
          <Monitoring data={monitoring.data} error={monitoring.error} />
        )}
        {view === "operations" && (
          <Operations
            retraining={retraining.data}
            resilience={resilience.data}
            retrainingError={retraining.error}
            resilienceError={resilience.error}
          />
        )}
      </section>
    </main>
  );
}

function Overview({
  data,
  error
}: {
  data: OpsOverview | null;
  error: string | null;
}) {
  const champion = data?.champion ?? {};
  const metrics = data?.metrics ?? {};
  return (
    <>
      <PageHeader title="Overview" status={error ?? data?.health?.inference?.status} />
      <div className="grid two">
        <section>
          <h2>Current Champion</h2>
          <div className="metrics">
            <MetricCard label="Family" value={champion.family} />
            <MetricCard label="Version" value={champion.version} />
            <MetricCard label="Execution" value={champion.execution_mode} />
            <MetricCard label="Demo" value={champion.demo_model} />
          </div>
        </section>
        <section>
          <h2>Service Health</h2>
          <div className="health-list">
            {Object.entries(data?.health ?? {}).map(([name, item]) => (
              <div key={name}>
                <span>{name.replaceAll("_", " ")}</span>
                <StatusBadge value={item.status ?? item.error} />
              </div>
            ))}
          </div>
        </section>
      </div>
      <section>
        <h2>Core Metrics</h2>
        <div className="metrics six">
          <MetricCard label="Drift Share" value={metrics.drift_share} />
          <MetricCard label="Attack Recall" value={metrics.attack_recall} />
          <MetricCard label="F1" value={metrics.f1} />
          <MetricCard label="False Positive Rate" value={metrics.false_positive_rate} />
          <MetricCard label="Labelled Rows" value={metrics.labelled_rows} />
          <MetricCard
            label="Prediction Distribution"
            value={compactJson(metrics.prediction_distribution)}
          />
        </div>
      </section>
      <JsonPanel title="Latest Lifecycle/Retraining Event" value={data?.latest_event} />
    </>
  );
}

function ModelsLifecycle({
  data,
  error
}: {
  data: OpsModels | null;
  error: string | null;
}) {
  return (
    <>
      <PageHeader title="Models & Lifecycle" status={error ?? data?.status} />
      <section>
        <table>
          <thead>
            <tr>
              <th>Version</th>
              <th>Family</th>
              <th>State</th>
              <th>Run ID</th>
              <th>Execution</th>
              <th>Reason</th>
              <th>Created</th>
            </tr>
          </thead>
          <tbody>
            {(data?.models ?? []).map((model) => (
              <tr key={model.version} className={model.aliases?.includes("champion") ? "champion" : ""}>
                <td>{model.version}</td>
                <td>{formatValue(model.model_family)}</td>
                <td><StatusBadge value={model.lifecycle_state} /></td>
                <td className="mono">{formatValue(model.run_id)}</td>
                <td>{formatValue(model.execution_mode)}</td>
                <td>{formatValue(model.rejection_reason)}</td>
                <td>{formatValue(model.created_timestamp)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </>
  );
}

function Monitoring({
  data,
  error
}: {
  data: OpsSection | null;
  error: string | null;
}) {
  const latest = data?.latest ?? {};
  const heartbeat = data?.heartbeat ?? {};
  const metrics = pickRecord(latest.performance);
  return (
    <>
      <PageHeader title="Monitoring" status={error ?? data?.status} />
      <div className="metrics six">
        <MetricCard label="Last Check" value={heartbeat.last_check_timestamp} />
        <MetricCard label="Window Size" value={latest.window_size} />
        <MetricCard label="Drift Share" value={latest.drift_share} />
        <MetricCard
          label="Drifting Features"
          value={latest.drifting_feature_count}
        />
        <MetricCard label="Attack Recall" value={metrics.attack_recall} />
        <MetricCard label="F1" value={metrics.f1} />
      </div>
      <FriendlyPanel
        title="Top Drifting Features"
        message={friendlyMessage(data, "No monitoring report is available yet.")}
        value={latest.drifting_features}
      />
      <DetailsPanel title="Monitoring Raw Details" value={data} />
    </>
  );
}

function Operations({
  retraining,
  resilience,
  retrainingError,
  resilienceError
}: {
  retraining: OpsSection | null;
  resilience: OpsSection | null;
  retrainingError: string | null;
  resilienceError: string | null;
}) {
  const status = operationsStatus(retraining, resilience, retrainingError, resilienceError);
  return (
    <>
      <PageHeader
        title="Operations & Resilience"
        status={status}
      />
      <div className="grid two">
        <SectionSummary title="Retraining" data={retraining} />
        <SectionSummary title="Resilience" data={resilience} />
      </div>
      <FriendlyPanel
        title="Latest Retraining"
        message={friendlyMessage(retraining, "Retraining has not run yet.")}
        value={retraining?.latest ?? retraining?.latest_processed_monitoring}
      />
      <FriendlyPanel
        title="Latest Resilience"
        message={friendlyMessage(resilience, "No active probation.")}
        value={resilience?.latest}
      />
      <DetailsPanel title="Developer Details" value={{ retraining, resilience }} />
    </>
  );
}

function SectionSummary({ title, data }: { title: string; data: OpsSection | null }) {
  return (
    <section>
      <h2>{title}</h2>
      <div className="metrics">
        <div className="metric">
          <span>Status</span>
          <strong>
            <StatusBadge value={data?.status} />
          </strong>
        </div>
        <MetricCard
          label="Latest Decision"
          value={friendlyValue(data?.latest_decision)}
        />
        <MetricCard label="Latest Result" value={friendlyValue(data?.latest)} />
        <MetricCard label="Heartbeat" value={friendlyValue(data?.heartbeat)} />
      </div>
    </section>
  );
}

function PageHeader({ title, status }: { title: string; status?: string | null }) {
  return (
    <header className="page-header">
      <h1>{title}</h1>
      <StatusBadge value={status ?? "warming up"} />
    </header>
  );
}

function JsonPanel({ title, value }: { title: string; value: unknown }) {
  return (
    <section>
      <h2>{title}</h2>
      <pre>{compactJson(value)}</pre>
    </section>
  );
}

function FriendlyPanel({
  title,
  message,
  value
}: {
  title: string;
  message: string;
  value: unknown;
}) {
  const hasValue =
    value !== null &&
    value !== undefined &&
    !(Array.isArray(value) && value.length === 0);
  return (
    <section>
      <h2>{title}</h2>
      {hasValue ? <pre>{compactJson(value)}</pre> : <p className="empty">{message}</p>}
    </section>
  );
}

function DetailsPanel({ title, value }: { title: string; value: unknown }) {
  return (
    <section>
      <details>
        <summary>{title}</summary>
        <pre>{compactJson(value)}</pre>
      </details>
    </section>
  );
}

function compactJson(value: unknown): string {
  if (value === null || value === undefined) {
    return "N/A";
  }
  if (typeof value === "string" || typeof value === "number") {
    return String(value);
  }
  return JSON.stringify(value, null, 2);
}

function pickRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" ? (value as Record<string, unknown>) : {};
}

function friendlyValue(value: unknown): string {
  if (value === null || value === undefined) {
    return "No data yet";
  }
  if (typeof value === "object") {
    const status = (value as Record<string, unknown>).status;
    if (status === "no_data") {
      return "No data yet";
    }
    if (status === "inactive") {
      return "Inactive";
    }
  }
  return compactJson(value);
}

function friendlyMessage(data: OpsSection | null, fallback: string): string {
  if (!data) {
    return fallback;
  }
  if (data.status === "unavailable") {
    return data.error ?? "Service is unavailable.";
  }
  if (data.status === "no_data") {
    return "No data yet.";
  }
  if (data.status === "inactive") {
    return data.message ?? fallback;
  }
  if (data.status === "warming_up") {
    return "Warming up.";
  }
  return fallback;
}

function operationsStatus(
  retraining: OpsSection | null,
  resilience: OpsSection | null,
  retrainingError: string | null,
  resilienceError: string | null
): string {
  if (retrainingError || resilienceError) {
    return "degraded";
  }
  const statuses = [retraining?.status, resilience?.status].filter(Boolean);
  if (statuses.includes("unavailable") || statuses.includes("stale")) {
    return "degraded";
  }
  if (statuses.length === 0) {
    return "warming_up";
  }
  return "healthy";
}
