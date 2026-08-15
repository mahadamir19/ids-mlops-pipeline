export function StatusBadge({ value }: { value?: string | null }) {
  const normalized = (value || "unavailable").toLowerCase();
  const label = statusLabel(normalized);
  const tone =
    ["healthy", "available"].includes(normalized)
      ? "good"
      : ["degraded", "pending", "promotion_pending", "stale", "warming_up"].some(
          (status) => normalized.includes(status)
        )
        ? "warn"
        : normalized.includes("inactive") || normalized.includes("no_data")
          ? "neutral"
          : "bad";
  return <span className={`status ${tone}`}>{label}</span>;
}

function statusLabel(value: string): string {
  const labels: Record<string, string> = {
    no_data: "No data yet",
    inactive: "Inactive",
    unavailable: "Unavailable",
    stale: "Stale",
    warming_up: "Warming up",
    healthy: "Healthy",
    available: "Available",
    degraded: "Degraded",
    idle: "Idle",
    cooldown: "Cooldown",
    active: "Active"
  };
  return labels[value] ?? value.replaceAll("_", " ");
}
