import { formatValue } from "../utils/format";

export function MetricCard({
  label,
  value
}: {
  label: string;
  value: unknown;
}) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{formatValue(value)}</strong>
    </div>
  );
}
