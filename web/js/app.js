/* ORBIT-PRISM website logic. Every number on the page comes from the API; nothing is hard-coded except
   explanatory text. Server strings are escaped before they reach the DOM. */
const $ = (s) => document.querySelector(s);
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const F = {
  pct: (v, d = 1) => (v == null ? "–" : (v * 100).toFixed(d) + "%"),
  mb: (v) => (v == null ? "never" : v.toFixed(2)),
  n: (v, d = 3) => (v == null ? "–" : Number(v).toFixed(d)),
  kb: (b) => (b == null || b === "NOT_MEASURED" ? "–" : (b / 1024).toFixed(1)),
};
const ST = { status: null, alpha: 0.05, split: "holdout", chip: null, tiles: [], sel: null, dial: null, simTimer: null };
const STEPS = [
  ["Chip sits in storage", "A full 13-band chip is 6.8 MB, more than the 4.2 MB of working memory, so it stays in flash and streams in."],
  ["Cut into 16 tiles", "128×128 pixels each: 65,536 B optical, 32,768 B radar."],
  ["Cloud check", "Is the optical tile usable? Cloud cover of 50% or more routes the tile to radar. Clouds are simulated."],
  ["Run the small INT8 model", "The optical model on clear tiles, the radar model under cloud; both far below the 4.2 MB budget."],
  ["Subtract normal water", "A river is not a flood. In our evaluation this rule is applied to the labels (JRC permanent-water layer)."],
  ["Miss-rate dial", "Keep or skip each tile using a threshold calibrated on held-back data, separately for each path."],
  ["Priority score", "Flood probability × flood extent."],
  ["Scheduler", "Fill each pass by priority per byte; leftovers carry over to the next pass."],
  ["Send in layers", "A 32-byte alert first, then a preview, then the full tile. Minutes matter more than pixels."],
  ["Ground dashboard", "Operators see what was sent, what was skipped and what it cost."],
];

async function guard(fn, el) {                       // run an async UI action; show errors in the page, never silently
  try { return await fn(); } catch (e) { if (el) el.innerHTML = `<div class="notice err">${esc(e.message)}</div>`; console.error(e); }
}

/* ------------------------------------------------------------------ top of page */
async function init() {
  initOrbitalCanvas();
  const s = (ST.status = await guard(API.status));
  if (!s) { $("#stats").innerHTML = `<div class="notice err">The API is not running. Start it with: python -m uvicorn app.api:app --port 8000</div>`; return; }
  if (s.banner) {
    const b = $("#banner"); b.textContent = s.banner; b.style.display = "block"; document.body.classList.add("has-banner");
    const fit = () => document.documentElement.style.setProperty("--bh", b.offsetHeight + "px");   // the banner wraps on phones
    fit(); window.addEventListener("resize", fit);
  }
  $("#stats").innerHTML = [
    [s.link.pass_MB.toFixed(1) + " MB", `per ${s.link.seconds_per_pass / 60}-minute pass at ${s.link.kbps} kbit/s`],
    [s.sram_MB.toFixed(1) + " MB", "working memory on board"],
    [s.packets.L0 + " bytes", "alert sent before any tile"],
    [s.events.reduce((a, e) => a + e.tiles, 0).toLocaleString(), `real flood tiles, ${s.events.length} events`],
  ].map(([b, t]) => `<div class="stat"><b>${esc(b)}</b><span>${esc(t)}</span></div>`).join("");
  $("#steps").innerHTML = STEPS.map(([t, d], i) => `
    <li class="${i === 9 ? "ground" : ""}" data-step="${i + 1}">
      <span class="tier-badge ${i === 9 ? "tier-ground" : "tier-onboard"}">${i === 9 ? "GROUND RECEIPT" : "ONBOARD 4.2 MB"}</span>
      <b>${esc(t)}</b>
      <span>${esc(d)}</span>
      <div class="step-live" style="display:none;"></div>
    </li>`).join("");
  if (s.mode === "unavailable") { $("#try").insertAdjacentHTML("afterbegin", `<div class="notice err">No results are available yet. Run <code>python scripts/evaluate.py --allow-placeholder</code> (placeholder) or follow the runbook in README.md.</div>`); return; }

  ST.dial = await guard(API.dial);
  $("#dial").innerHTML = s.dial_targets.map((a) => `<button type="button" data-a="${a}" aria-pressed="${a === ST.alpha}">${Math.round(a * 100)}%</button>`).join("");
  $("#dial").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) setAlpha(+b.dataset.a); });
  $("#split").addEventListener("change", (e) => { ST.split = e.target.value; loadChips(); });
  $("#chip").addEventListener("change", (e) => loadTiles(e.target.value));
  $("#upcloud").addEventListener("input", (e) => ($("#upcloudv").textContent = Math.round(e.target.value * 100) + "%"));
  $("#upgo").addEventListener("click", uploadTile);
  $("#simgo").addEventListener("click", runSim);
  setAlpha(ST.alpha, true);
  loadChips();
  loadResults();
}

function setAlpha(a, quiet) {
  ST.alpha = a;
  document.querySelectorAll("#dial button").forEach((b) => b.setAttribute("aria-pressed", +b.dataset.a === a));
  const t = ST.dial && ST.dial.targets.find((x) => x.target_miss_rate === a), note = $("#dialnote");
  const zero = t && Object.entries(t.sends_everything).filter(([, v]) => v).map(([k]) => k);
  if (zero && zero.length) {
    note.hidden = false;
    note.innerHTML = `Dial target <b>${Math.round(a * 100)}%</b>: ${zero.join(" and ")} has few critical calibration tiles, so the conformal threshold is 0. <b>It keeps every tile on that path</b>, missing none.`;
  } else { note.hidden = true; note.textContent = ""; }
  paintGrid();
  if (ST.sel && !quiet) selectTile(ST.sel);
}

/* ------------------------------------------------------------------ try it */
async function loadChips() {
  const r = await guard(() => API.chips(ST.split), $("#grid"));
  if (!r) return;
  const sel = $("#chip");
  sel.innerHTML = r.chips.map((c) => `<option value="${esc(c.chip)}">${esc(c.chip)} · ${c.tiles} tiles · ${c.critical} critical · ${Math.round(c.mean_cloud * 100)}% cloud</option>`).join("");
  const best = r.chips.find((c) => c.critical > 0 && c.mean_cloud > 0.45) || r.chips.find((c) => c.critical > 0) || r.chips[0];
  if (best) { sel.value = best.chip; loadTiles(best.chip); }
}

async function loadTiles(chip) {
  ST.chip = chip; ST.sel = null;
  const r = await guard(() => API.tiles(chip), $("#grid"));
  if (!r) return;
  ST.tiles = r.tiles;
  paintGrid();
  const first = r.tiles.find((t) => t.critical) || r.tiles[0];
  if (first) selectTile(first.tile_id);
}

function paintGrid() {
  const grid = $("#grid"), by = {};
  ST.tiles.forEach((t) => (by[t.row + "," + t.col] = t));
  let h = "";
  for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) {
    const t = by[r + "," + c];
    if (!t) { h += `<div class="tile" aria-hidden="true"><div class="nopx">no valid pixels</div></div>`; continue; }
    const kept = t.kept[String(ST.alpha)];
    h += `<button type="button" class="tile ${kept ? "keep" : "skip"} ${t.tile_id === ST.sel ? "sel" : ""}" data-id="${esc(t.tile_id)}" aria-label="Tile row ${r} column ${c}, ${kept ? "kept" : "skipped"}${t.critical ? ", critical" : ""}, ${t.sensor_used}">
      <img loading="lazy" alt="" src="${API.imageUrl(t.tile_id, "optical", 1)}" onerror="this.outerHTML='<div class=&quot;nopx&quot;>pixels not built</div>'">
      <span class="badge ${t.sensor_used === "radar" ? "r" : ""}">${t.sensor_used === "radar" ? "R" : "O"}</span>${t.critical ? '<span class="crit" title="critical">★</span>' : ""}</button>`;
  }
  grid.innerHTML = h;
  grid.querySelectorAll("button.tile").forEach((b) => b.addEventListener("click", () => selectTile(b.dataset.id)));

  // Print keep/skip and critical counts as text as well as colour
  const countsEl = $("#grid-counts");
  if (countsEl && ST.tiles.length) {
    const total = ST.tiles.length;
    const keptN = ST.tiles.filter((t) => t.kept[String(ST.alpha)]).length;
    const skipN = total - keptN;
    const critN = ST.tiles.filter((t) => t.critical).length;
    countsEl.innerHTML = `${total} tiles total: <b>${keptN} kept</b> at ${Math.round(ST.alpha * 100)}% dial, <b>${skipN} skipped</b> · <b>${critN} critical</b> (≥ 30% flood)`;
  }
}

async function selectTile(id) {
  ST.sel = id;
  document.querySelectorAll("#grid .tile").forEach((b) => b.classList.toggle("sel", b.dataset && b.dataset.id === id));
  const r = await guard(() => API.triage(id, ST.alpha), $("#detail"));
  if (r) {
    renderDetail(r, $("#detail"), true);
    highlightPipelineSteps(r);
  }
}

function hexBytes(hex) {
  // layout from src/scheduler/sim.py: version+event+chip number 0-5, row/col/sensor/hazard 6-9, probability/extent/priority 10-15,
  // pass+sequence+tile size+reserved 16-29, CRC-16 30-31
  const cls = (i) => (i < 6 ? "h1" : i < 10 ? "h2" : i < 16 ? "h0" : i < 30 ? "h3" : "h1");
  let h = "";
  for (let i = 0; i < hex.length / 2; i++) h += `<span class="${cls(i)}">${hex.substr(i * 2, 2)}</span>`;
  return h;
}

function renderDetail(r, el, withImages) {
  const both = r.both_models || {}, id = r.tile_id;
  const kept = r.decision === "KEEP";
  const img = (kind, label) => `<figure><img alt="${label}" src="${API.imageUrl(id, kind, 2)}" onerror="this.outerHTML='<div class=&quot;ph&quot;>not available</div>'">${label}</figure>`;
  const modelRows = ["optical", "radar"].filter((m) => both[m]).map((m) =>
    `<dt>${m} model (${esc(both[m].model)})</dt><dd>${F.n(both[m].p_flood)} flood probability${m === r.routed_to ? " <b>(used)</b>" : ""}</dd>`).join("");
  el.innerHTML = `
    ${withImages ? `<div class="imgs">${img("optical", "optical, simulated cloud")}${img("radar", "radar (VV)")}${img("label", "label: yellow = flood")}</div>` : ""}
    <h3 style="margin-top:14px">${esc(id)}</h3>
    <p class="small">Cloud cover <b>${F.pct(r.cloud_frac, 0)}</b> ${r.routed_to === "radar" ? "≥" : "<"} ${F.pct(r.cloud_switch, 0)}, so this tile is routed to the <b>${esc(r.routed_to)}</b> model.
      A system that discards cloudy frames would <b>${esc(r.discard_cloudy_would)}</b>.</p>
    <div class="meter" aria-hidden="true"><div class="fill" style="width:${(r.p_flood * 100).toFixed(1)}%"></div><div class="thr" style="left:${(r.threshold * 100).toFixed(1)}%" title="dial threshold"></div></div>
    <p class="small">Flood probability <b>${F.n(r.p_flood)}</b>, dial threshold <b>${F.n(r.threshold)}</b> (yellow marker, ${Math.round(r.alpha * 100)}% target miss rate)${r.threshold_is_send_everything ? ": threshold 0, keeps everything" : ""}.</p>
    <p><span class="decision ${kept ? "keep" : "skip"}"><span aria-hidden="true">${kept ? "✔" : "✖"}</span> ${esc(r.decision)}</span>
      ${r.outcome ? ` <b style="margin-left:8px">${esc(r.outcome)}</b>` : ""}</p>
    <dl class="kv">${modelRows}<dt>Extent</dt><dd>${F.pct(r.extent)} of the tile</dd><dt>Priority</dt><dd>${F.n(r.priority, 4)} (probability × extent)</dd>
      <dt>Full tile</dt><dd>${r.l2_bytes.toLocaleString()} bytes</dd>${r.truth ? `<dt>Truth</dt><dd>${F.pct(r.truth.water_frac)} flood water; ${r.truth.critical ? "critical" : "not critical"} (≥ 30%)</dd>` : ""}
      <dt>Score from</dt><dd>${esc(r.score_source)}</dd></dl>
    <div class="small muted">The ${r.alert_bytes}-byte alert sent before any tile:</div>
    <div class="hex" aria-label="Alert packet in hexadecimal">${hexBytes(r.alert_hex)}</div>
    <div class="small muted"><span style="color:var(--lav)">event, chip number, checksum</span> · <span style="color:var(--lav2)">row, column, sensor, hazard</span> · <span style="color:var(--lemon)">probability, extent, priority</span> · <span>pass, sequence, tile size</span></div>
    ${r.score_source === "stored score" ? `<div class="notice">The ${esc(r.routed_to)} model could not be run here (no model file or no pixels for this tile on this machine), so the stored score is shown instead of a live model output.</div>` : ""}
    ${r.synthetic ? `<div class="notice">Placeholder mode: the dial thresholds come from synthetic scores, so this decision shows the plumbing, not a result.</div>` : ""}`;
}

/* Light up steps 3 (cloud check), 4 (model), 6 (dial) and 9 (alert) in How It Works */
function highlightPipelineSteps(r) {
  document.querySelectorAll("#steps li").forEach((li) => {
    li.classList.remove("step-lit");
    const live = li.querySelector(".step-live");
    if (live) { live.style.display = "none"; live.innerHTML = ""; }
  });

  const setStepLive = (stepNum, html) => {
    const li = $(`#steps li[data-step="${stepNum}"]`);
    if (!li) return;
    li.classList.add("step-lit");
    const live = li.querySelector(".step-live");
    if (live) {
      live.innerHTML = `<span class="step-live-val">${html}</span>`;
      live.style.display = "block";
    }
  };

  // Step 3: Cloud check
  setStepLive(3, `Cloud ${Math.round(r.cloud_frac * 100)}% (${r.cloud_frac >= r.cloud_switch ? "≥" : "<"} 50% → <b>${r.routed_to.toUpperCase()}</b>)`);
  // Step 4: Run the small INT8 model
  setStepLive(4, `${r.routed_to.toUpperCase()} model → p_flood = <b>${F.n(r.p_flood)}</b>`);
  // Step 6: Miss-rate dial
  setStepLive(6, `Dial α=${Math.round(r.alpha * 100)}% (τ=${F.n(r.threshold)}) → <b>${esc(r.decision)}</b>`);
  // Step 9: Send in layers (32-byte alert)
  setStepLive(9, `Alert packet built (${r.alert_bytes}B: 0x${r.alert_hex ? r.alert_hex.slice(0, 8) : ""}...)`);
}

async function uploadTile() {
  const f = $("#upfile").files[0], out = $("#upout");
  if (!f) { out.innerHTML = `<div class="notice err">Choose a .npz file first.</div>`; return; }
  const r = await guard(() => API.upload(f, $("#upcloud").value, ST.alpha), out);
  if (r) {
    out.innerHTML = '<div class="card" style="margin-top:12px"></div>';
    renderDetail(r, out.firstChild, false);
    highlightPipelineSteps(r);
  }
}

/* ------------------------------------------------------------------ link simulator */
async function runSim() {
  if (ST.simTimer) { clearInterval(ST.simTimer); ST.simTimer = null; }
  const out = $("#simout"); out.innerHTML = `<p class="muted">Running simulation…</p>`;
  const body = { split: ST.split, alpha: ST.alpha, chip: $("#scope").value === "chip" ? ST.chip : null };
  const r = await guard(() => API.simulate(body), out);
  if (!r) return;
  const curves = {}; Object.entries(r.systems).forEach(([n, s]) => (curves[n] = s.curve));
  const rows = Object.entries(r.systems).map(([n, s]) => ({ system: n, ...s.summary }));
  const orbit = r.systems["ORBIT-PRISM"];
  out.innerHTML = `
    <div class="notice">${esc(r.selection.tiles)} tiles, ${esc(r.selection.critical)} critical. ${esc(r.scaled_note)} (${r.scaled_budget_MB.toFixed(2)} MB). Dial ${Math.round(r.alpha * 100)}%.</div>
    ${r.selection.critical === 0 ? `<div class="notice err">No critical tiles in this selection, so recall is undefined. Choose a chip with a ★ tile or the whole split.</div>` : ""}
    <div class="card chart" id="simchart"></div>
    <div class="card" style="margin-top:14px"><div class="tablewrap">${summaryTable(rows)}</div></div>
    ${Object.keys(r.skipped).length ? `<div class="notice">Not run: ${Object.entries(r.skipped).map(([k, v]) => esc(k + " (" + v + ")")).join("; ")}</div>` : ""}
    ${orbit ? `
      <div class="card" style="margin-top:14px" id="budgetProgress">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
          <div><b>Downlink transmission sequence</b>: <span id="budgetVal">0.00 / ${r.scaled_budget_MB.toFixed(2)} MB</span></div>
          <div class="small muted" id="budgetPct">0% of scaled pass</div>
        </div>
        <div class="meter" style="height:12px; margin:8px 0 6px;"><div id="budgetFill" class="fill" style="width:0%; transition:width 0.12s linear;"></div></div>
        <div style="display:flex; gap:12px; align-items:center; margin-top:8px; flex-wrap:wrap;">
          <button type="button" class="btn play" id="playQueueBtn">▶ Play downlink sequence</button>
          <span id="playStatus" class="small muted">Press Play to animate packet transmission against the scaled pass budget.</span>
        </div>
      </div>
      <div class="card" style="margin-top:14px">
        <h3>ORBIT-PRISM send queue: the first 15 packets of each layer (of ${orbit.total_packets.toLocaleString()} in total)</h3>
        <div class="queue" id="queueTableWrap">${queueTable(orbit.queue_preview)}</div>
      </div>` : ""}`;
  recallChart($("#simchart"), curves, { scaledMB: r.scaled_budget_MB });

  if (orbit && orbit.queue_preview && $("#playQueueBtn")) {
    $("#playQueueBtn").addEventListener("click", () => playQueueAnimation(orbit.queue_preview, r.scaled_budget_MB));
  }
}

function playQueueAnimation(queue, budgetMB) {
  if (ST.simTimer) { clearInterval(ST.simTimer); ST.simTimer = null; }
  const btn = $("#playQueueBtn");
  const val = $("#budgetVal");
  const pct = $("#budgetPct");
  const fill = $("#budgetFill");
  const status = $("#playStatus");
  const rows = document.querySelectorAll("#queueTableWrap table tr");
  let idx = 0;

  btn.textContent = "⏸ Playing...";
  status.textContent = "Transmitting packets in priority order...";

  ST.simTimer = setInterval(() => {
    if (idx >= queue.length) {
      clearInterval(ST.simTimer);
      ST.simTimer = null;
      btn.textContent = "↺ Replay sequence";
      status.textContent = "Transmission complete for preview queue.";
      return;
    }
    const pkt = queue[idx];
    const cumMB = pkt.cum_bytes / 1e6;
    const ratio = Math.min(100, (cumMB / budgetMB) * 100);
    fill.style.width = ratio.toFixed(1) + "%";
    val.textContent = `${cumMB.toFixed(3)} / ${budgetMB.toFixed(2)} MB`;
    pct.textContent = `${ratio.toFixed(1)}% of scaled pass`;

    // Highlight row in table (idx + 1 because of header tr)
    rows.forEach((r) => r.classList.remove("active-pkt"));
    if (rows[idx + 1]) {
      rows[idx + 1].classList.add("active-pkt");
      rows[idx + 1].scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
    idx++;
  }, 120);
}

function queueTable(q) {
  return `<table><tr><th>#</th><th>layer</th><th>tile</th><th>bytes</th><th>cumulative MB</th><th>pass</th></tr>` + q.map((p) =>
    `<tr><td>${p.order}</td><td style="text-align:left"><span class="pill ${esc(p.layer)}">${esc(p.layer)}</span></td><td style="text-align:left">${esc(p.tile_id)}${p.critical ? " ★" : ""}</td><td>${p.bytes.toLocaleString()}</td><td>${(p.cum_bytes / 1e6).toFixed(3)}</td><td>${p.pass_index}</td></tr>`).join("") + `</table>`;
}

/* ------------------------------------------------------------------ results */
function summaryTable(rows) {
  const head = ["System", "MB to 95% recall", "Recall at scaled pass", "Recall on cloudy tiles", "Final recall", "Seconds to first alert", "MB sent"];
  return `<table><tr>${head.map((h) => `<th>${h}</th>`).join("")}</tr>` + rows.map((r) =>
    `<tr class="${r.system === "ORBIT-PRISM" ? "ours" : ""}"><td>${esc(r.system)}</td><td>${F.mb(r.bytes_to_95pct_recall_MB)}</td><td>${F.pct(r.recall_at_scaled_pass)}</td><td>${F.pct(r.cloudy_recall_at_scaled_pass)}</td><td>${F.pct(r.final_full_tile_recall)}</td><td>${F.n(r.seconds_to_first_critical_alert, 2)}</td><td>${F.n(r.total_MB_sent, 2)}</td></tr>`).join("") + `</table>`;
}

async function loadResults() {
  const syn = ST.status.synthetic;
  $("#resnote").innerHTML = syn
    ? `<b>These numbers come from synthetic placeholder scores and are NOT results.</b> They show that the evaluation runs end to end. Real numbers appear here when the trained models' scores are merged.`
    : `Held-out event Bolivia, one training run, simulated clouds. Recall is the share of critical tiles delivered as full tiles; the vertical line is the scaled pass, not the real 19.2 MB pass.`;
  const [sum, cur, dial, mem, demo] = await Promise.all([API.summary(), API.curves(), API.dial(), API.memory(), API.demoCase()].map((p) => p.catch((e) => ({ error: e.message }))));
  if (cur.curves) { const c = {}; Object.entries(cur.curves).forEach(([n, p]) => !n.includes("ablation") && (c[n] = p)); recallChart($("#chart"), c, { scaledMB: cur.scaled_pass_MB }); }
  else $("#chart").innerHTML = `<div class="notice err">${esc(cur.error)}</div>`;
  $("#summary").innerHTML = sum.rows ? summaryTable(sum.rows.filter((r) => !r.system.includes("ablation"))) : `<div class="notice err">${esc(sum.error)}</div>`;
  $("#dialtable").innerHTML = dial.measured ? (dial.measured.length ? `<table><tr><th>path</th><th>target miss rate</th><th>threshold</th><th>critical tiles</th><th>missed</th><th>achieved</th></tr>` + dial.measured.map((d) =>
    `<tr><td>${esc(d.path)}</td><td>${F.pct(d.target_miss_rate, 0)}</td><td>${F.n(d.threshold)}</td><td>${d.critical_tiles}</td><td>${d.missed}</td><td>${F.pct(d.achieved_miss_rate)}</td></tr>`).join("") + `</table><p class="small muted">A target is met only if test data resembles the calibration data; the achieved rate is whatever was measured.</p>` : "<p class='muted'>Not available yet.</p>") : `<div class="notice err">${esc(dial.error)}</div>`;
  $("#memory").innerHTML = mem.rows && mem.rows.length ? `<table><tr><th>model</th><th>dtype</th><th>weights KB</th><th>peak activation KB (est.)</th><th>total KB</th><th>% of 4.2 MB</th><th>MACs</th><th>recall</th><th>ece</th><th>basis</th></tr>` + mem.rows.map((m) =>
    `<tr><td>${esc(m.model)}</td><td>${esc(m.dtype)}</td><td>${F.kb(m.weights_bytes)}</td><td>${F.kb(m.peak_activation_bytes)}</td><td>${F.kb(m.total_bytes)}</td><td>${F.pct(m.total_bytes / mem.sram_bytes)}</td><td>${Number(m.macs).toLocaleString()}</td><td>${esc(m.recall)}</td><td>${esc(m.ece)}</td><td>${esc(m.basis)}</td></tr>`).join("") + `</table>` : `<p class="muted">The memory table has not been produced yet (it comes from the INT8 step).</p>`;
  if (demo.available) {
    const d = $("#democase"); d.hidden = false;
    d.innerHTML = `<h3>A flood hidden under cloud</h3><p>${esc(demo.note || "")}</p>` + (demo.tile_id ? `<div class="imgs"><figure><img alt="optical under cloud" src="${API.imageUrl(demo.tile_id, "optical", 2)}">optical under cloud</figure><figure><img alt="radar" src="${API.imageUrl(demo.tile_id, "radar", 2)}">radar</figure><figure><img alt="label" src="${API.imageUrl(demo.tile_id, "label", 2)}">label</figure></div>` : "");
  }
}

/* ------------------------------------------------------------------ hero orbital canvas */
function initOrbitalCanvas() {
  const canvas = $("#orbitalCanvas");
  if (!canvas || !canvas.getContext) return;
  const ctx = canvas.getContext("2d");
  const prefersReduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  let angle = 0;

  function render() {
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    // Deep space backdrop grid
    ctx.strokeStyle = "rgba(36, 91, 255, 0.12)";
    ctx.lineWidth = 1;
    for (let x = 0; x < W; x += 24) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke(); }
    for (let y = 0; y < H; y += 24) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }

    // Planetary Horizon Arc
    const cx = W / 2, cy = H * 1.6, r = H * 1.15;
    const grad = ctx.createRadialGradient(cx, cy, r * 0.7, cx, cy, r * 1.1);
    grad.addColorStop(0, "#081b3d");
    grad.addColorStop(0.85, "#0e3477");
    grad.addColorStop(1, "rgba(36, 91, 255, 0)");
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(cx, cy, r, Math.PI, 0);
    ctx.fill();

    // Atmosphere limb glow
    ctx.strokeStyle = "rgba(74, 136, 255, 0.65)";
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.arc(cx, cy, r, Math.PI * 1.15, Math.PI * 1.85);
    ctx.stroke();

    // India Nadir Ground Target
    const tx = W * 0.54, ty = H * 0.62;
    ctx.fillStyle = "#c6ff00";
    ctx.beginPath();
    ctx.arc(tx, ty, 3.5, 0, Math.PI * 2);
    ctx.fill();

    // Concentric Target Ping
    const pingR = prefersReduced ? 10 : 6 + (Math.sin(angle * 2.5) + 1) * 6;
    ctx.strokeStyle = "rgba(198, 255, 0, 0.4)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(tx, ty, pingR, 0, Math.PI * 2);
    ctx.stroke();

    // Orbiting Satellite Position
    const satOrbitR = r * 1.28;
    const satAngle = prefersReduced ? -Math.PI * 0.68 : -Math.PI * 0.68 + Math.sin(angle * 0.8) * 0.14;
    const sx = cx + Math.cos(satAngle) * satOrbitR;
    const sy = cy + Math.sin(satAngle) * satOrbitR;

    // Laser link from satellite to target
    ctx.strokeStyle = "rgba(255, 255, 255, 0.85)";
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 3]);
    ctx.lineDashOffset = prefersReduced ? 0 : -angle * 20;
    ctx.beginPath();
    ctx.moveTo(sx, sy);
    ctx.lineTo(tx, ty);
    ctx.stroke();
    ctx.setLineDash([]);

    // Traveling Photonic Packets (Acid Lime priority + White)
    if (!prefersReduced) {
      for (let p = 0; p < 3; p++) {
        const t = ((angle * 0.8 + p * 0.33) % 1.0);
        const px = sx + (tx - sx) * t;
        const py = sy + (ty - sy) * t;
        ctx.fillStyle = p === 0 ? "#c6ff00" : "#ffffff";
        ctx.beginPath();
        ctx.arc(px, py, 2, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // Satellite Bus & Solar Panels
    ctx.fillStyle = "#d99b26"; // Kapton Gold
    ctx.fillRect(sx - 5, sy - 3, 10, 6);
    ctx.fillStyle = "#245bff"; // Solar wings
    ctx.fillRect(sx - 16, sy - 2, 9, 4);
    ctx.fillRect(sx + 7, sy - 2, 9, 4);

    if (!prefersReduced) {
      angle += 0.03;
      requestAnimationFrame(render);
    }
  }

  render();
}

init();
