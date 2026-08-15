import "@testing-library/jest-dom/vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "./App";

const overview = {
  champion: {
    available: true,
    version: "4",
    family: "random_forest",
    execution_mode: "smoke",
    demo_model: true
  },
  health: {
    inference: { status: "available" },
    registry: { status: "unavailable", error: "offline" },
    monitoring: { status: "stale" }
  },
  metrics: {
    drift_share: null,
    f1: 0.91
  },
  latest_event: null
};

let operationsScenario: "normal" | "unavailable" = "normal";

vi.stubGlobal(
  "fetch",
  vi.fn((url: string) => {
    const body = url.includes("/models")
      ? { status: "available", models: [{ version: "4", model_family: "random_forest", lifecycle_state: "champion", aliases: ["champion"] }] }
      : url.includes("/monitoring")
        ? { status: "stale", latest: { performance: { attack_recall: null, f1: 0.4 }, drifting_features: [] }, heartbeat: { status: "stale" } }
        : url.includes("/retraining")
          ? operationsScenario === "unavailable"
            ? { status: "unavailable", error: "retraining API down" }
            : { status: "no_data", latest_decision: null, latest: null }
          : url.includes("/resilience")
            ? operationsScenario === "unavailable"
              ? { status: "unavailable", error: "resilience API down" }
              : { status: "inactive", latest: null }
            : overview;
    return Promise.resolve({
      ok: true,
      json: () => Promise.resolve(body)
    });
  })
);

describe("SentinelML operations dashboard", () => {
  beforeEach(() => {
    operationsScenario = "normal";
  });

  it("renders overview and N/A metrics", async () => {
    render(<App />);
    expect(await screen.findByText("Current Champion")).toBeInTheDocument();
    expect(await screen.findByText("random_forest")).toBeInTheDocument();
    expect(screen.getAllByText("N/A").length).toBeGreaterThan(0);
  });

  it("renders the models table", async () => {
    render(<App />);
    await userEvent.click(screen.getByText("Models & Lifecycle"));
    expect(await screen.findByText("Run ID")).toBeInTheDocument();
    expect(await screen.findByText("champion")).toBeInTheDocument();
  });

  it("renders monitoring states without prominent raw error JSON", async () => {
    render(<App />);
    await userEvent.click(screen.getByText("Monitoring"));
    expect(await screen.findByText("Stale")).toBeInTheDocument();
    expect(screen.getByText("No monitoring report is available yet.")).toBeInTheDocument();
    expect(screen.getByText("Monitoring Raw Details")).toBeInTheDocument();
    expect(screen.queryByText("no monitoring report found")).not.toBeInTheDocument();
  });

  it("renders no_data inactive and degraded operations state", async () => {
    render(<App />);
    await userEvent.click(screen.getByText("Operations & Resilience"));
    expect((await screen.findAllByText("No data yet")).length).toBeGreaterThan(0);
    expect(await screen.findByText("Inactive")).toBeInTheDocument();
    expect(await screen.findByText("No active probation.")).toBeInTheDocument();
  });

  it("does not show Available when operations dependencies are unavailable", async () => {
    operationsScenario = "unavailable";
    render(<App />);
    await userEvent.click(screen.getByText("Operations & Resilience"));
    expect(await screen.findByText("Degraded")).toBeInTheDocument();
    expect(screen.getAllByText("Unavailable").length).toBeGreaterThan(0);
  });
});
