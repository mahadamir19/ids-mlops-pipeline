import { describe, expect, it, vi } from "vitest";
import { api } from "./client";

describe("api client", () => {
  it("raises readable errors for failed requests", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(() => Promise.resolve({ ok: false, status: 503, statusText: "Down" }))
    );

    await expect(api.overview()).rejects.toThrow("503 Down");
  });
});
