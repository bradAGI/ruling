// Client for a ruling server. Plain ESM, no dependencies; types live in index.d.ts.

export const choice = (instructions, criteria) => ({ type: "choice", instructions, criteria });
export const score = (instructions, criteria) => ({ type: "score", instructions, criteria });
export const noul = (instructions, criteria) => ({ type: "noul", instructions, ...(criteria && { criteria }) });
export const box = (instructions) => ({ type: "box", instructions });
export const image = (data, mediaType) => ({
  type: "image", media_type: mediaType,
  data: typeof data === "string" ? data : btoa(Array.from(data, (b) => String.fromCharCode(b)).join("")),
});

export class RulingError extends Error {
  constructor(status, detail) {
    super(`ruling request failed with ${status}: ${JSON.stringify(detail)}`);
    this.status = status;
    this.detail = detail;
  }
}

export class RulingClient {
  constructor({ baseUrl = "http://127.0.0.1:8010", apiKey, fetch: fetchImpl = globalThis.fetch } = {}) {
    this.baseUrl = baseUrl.replace(/\/$/, "");
    this.apiKey = apiKey;
    this.fetch = fetchImpl;
  }

  async #request(method, path, body, extraHeaders = {}) {
    const headers = { "content-type": "application/json", ...extraHeaders };
    if (this.apiKey) headers.authorization = `Bearer ${this.apiKey}`;
    const response = await this.fetch(`${this.baseUrl}${path}`, { method, headers, body: body && JSON.stringify(body) });
    const data = await response.json();
    if (!response.ok) throw new RulingError(response.status, data.error ?? data);
    return data;
  }

  systemOne({ state, questions, model, excludeFromTraining, assets, longContext }) {
    // The flag goes only when set, so every other body is exactly System One's.
    const body = { state, questions, model: model ?? null, ...(assets && { assets }),
                   ...(excludeFromTraining && { exclude_from_training: true }) };
    return this.#request("POST", "/v1/systemone", body,
                         longContext ? { "TypeSafe-Danger-Zone-Long-Context": "true" } : {});
  }

  async outcome(decisionId, labels) {
    const body = await this.#request("POST", `/v1/decisions/${encodeURIComponent(decisionId)}/outcome`, { labels });
    return body.labels;
  }

  async models() {
    const body = await this.#request("GET", "/v1/models");
    return body.models;
  }
}
