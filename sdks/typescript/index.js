// Client for a ruling server. Plain ESM, no dependencies; types live in index.d.ts.

export const choice = (instructions, criteria) => ({ type: "choice", instructions, criteria });
export const score = (instructions, criteria) => ({ type: "score", instructions, criteria });
export const noul = (instructions) => ({ type: "noul", instructions });

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

  async #request(method, path, body) {
    const headers = { "content-type": "application/json" };
    if (this.apiKey) headers.authorization = `Bearer ${this.apiKey}`;
    const response = await this.fetch(`${this.baseUrl}${path}`, { method, headers, body: body && JSON.stringify(body) });
    const data = await response.json();
    if (!response.ok) throw new RulingError(response.status, data.detail ?? data);
    return data;
  }

  systemOne({ state, questions, model }) {
    return this.#request("POST", "/v1/systemone", { state, questions, model: model ?? null });
  }

  async models() {
    const body = await this.#request("GET", "/v1/models");
    return body.models;
  }
}
