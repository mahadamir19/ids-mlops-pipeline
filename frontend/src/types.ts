export type Availability = "available" | "unavailable" | "healthy" | "degraded";

export interface Champion {
  available?: boolean;
  model_name?: string;
  version?: string;
  family?: string | null;
  execution_mode?: string | null;
  demo_model?: boolean | null;
  loaded_at?: string;
  source_run_id?: string | null;
}

export interface OpsOverview {
  champion?: Champion;
  health?: Record<string, { status?: string; error?: string }>;
  metrics?: Record<string, unknown>;
  latest_event?: Record<string, unknown> | null;
}

export interface OpsModels {
  status: string;
  models: Array<{
    version: string;
    model_family?: string | null;
    lifecycle_state?: string | null;
    run_id?: string | null;
    execution_mode?: string | null;
    rejection_reason?: string | null;
    created_timestamp?: string | number | null;
    promoted_timestamp?: string | null;
    source_model_uri?: string | null;
    aliases?: string[];
  }>;
  error?: string;
}

export interface OpsSection {
  status: string;
  latest?: Record<string, unknown> | null;
  heartbeat?: Record<string, unknown> | null;
  latest_decision?: Record<string, unknown> | null;
  latest_processed_monitoring?: Record<string, unknown> | null;
  enabled?: boolean;
  error?: string;
  message?: string;
}
