/* ORBIT-PRISM API client. No dependencies. Every function returns parsed JSON or throws an Error whose
   message is the server's explanation (so the UI can show it). Reuse this file as-is in any page. */
const API = (() => {
  async function parse(res) {
    if (res.ok) return res.json();
    let detail = res.statusText;
    try { const j = await res.json(); detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch (_) {}
    const e = new Error(detail); e.status = res.status; throw e;
  }
  const get = (path) => fetch("/api" + path).then(parse);
  const post = (path, body) => fetch("/api" + path, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then(parse);
  const qs = (o) => { const p = new URLSearchParams(); Object.entries(o).forEach(([k, v]) => v != null && v !== "" && p.set(k, v)); const s = p.toString(); return s ? "?" + s : ""; };
  return {
    status: () => get("/status"),
    summary: () => get("/summary"),
    curves: () => get("/curves"),
    dial: () => get("/dial"),
    memory: () => get("/memory"),
    demoCase: () => get("/demo-case"),
    chips: (event, split) => get("/chips" + qs({ split, event })),
    tiles: (chip) => get("/tiles" + qs({ chip })),
    triage: (tile_id, alpha) => post("/triage", { tile_id, alpha }),
    simulate: (body) => post("/simulate", body),
    upload: (file, cloud_frac, alpha) => {
      const f = new FormData(); f.append("file", file); f.append("cloud_frac", cloud_frac); f.append("alpha", alpha);
      return fetch("/api/triage/upload", { method: "POST", body: f }).then(parse);
    },
    imageUrl: (tile_id, kind = "optical", scale = 2) => `/api/tile/${encodeURIComponent(tile_id)}/image?kind=${kind}&scale=${scale}&v=4`,
  };
})();
