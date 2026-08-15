import type { OpsModels, OpsOverview, OpsSection } from "../types";

const API_BASE = import.meta.env.VITE_SENTINELML_API_BASE ?? "";

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return (await response.json()) as T;
}

export const api = {
  overview: () => getJson<OpsOverview>("/ops/overview"),
  models: () => getJson<OpsModels>("/ops/models"),
  monitoring: () => getJson<OpsSection>("/ops/monitoring"),
  retraining: () => getJson<OpsSection>("/ops/retraining"),
  resilience: () => getJson<OpsSection>("/ops/resilience")
};
