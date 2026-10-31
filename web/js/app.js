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
const ST = { status: null, alpha: 0.05, event: "Bolivia", view: "optical", chip: null, tiles: [], sel: null, dial: null, simTimer: null, runs: [] };
const EVENT_PLACE = {   // centre of each event's footprint (Sen1Floods11 metadata), for the marker on the Earth
  Bolivia: [-13.8, -65.02], Ghana: [8.68, -1.11], India: [26.48, 93.1], Mekong: [12.13, 105.32], Nigeria: [7.65, 5.79],
  Pakistan: [30.03, 70.75], Paraguay: [-24.74, -56.56], Somalia: [4.51, 45.26], Spain: [38.49, 0.0], "Sri-Lanka": [7.58, 81.09], USA: [39.35, -95.05],
};
const STEPS = [
  ["Chip sits in storage", "A full 13-band chip is 6.8 MB, more than the 4.2 MB of working memory, so it stays in flash and streams in."],
  ["Cut into 16 tiles", "128×128 pixels each: 65,536 B optical, 32,768 B radar."],
  ["Cloud check", "Is the optical tile usable? Cloud cover of 50% or more routes the tile to radar. Clouds are simulated."],
  ["Run the small INT8 model", "The optical model on clear tiles, the radar model under cloud; both far below the 4.2 MB budget."],
  ["Subtract normal water", "A river is not a flood. In our evaluation this rule is applied to the labels (JRC permanent-water layer)."],
  ["Miss-rate dial", "Keep or skip each tile using a cut-off calibrated on held-back data, separately for each sensor."],
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
  const s = (ST.status = await guard(API.status));
  if (!s) { $("#pill").textContent = "API offline"; $("#stats").innerHTML = `<div class="notice err">The API is not running. Start it with: python -m uvicorn app.api:app --port 8000</div>`; return; }
  const pill = $("#pill");
  pill.textContent = s.synthetic ? "PLACEHOLDER SCORES" : "REAL SCORES · " + s.headline_split.toUpperCase();
  pill.classList.add(s.synthetic ? "syn" : "real");
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
      <span class="tier-badge ${i === 9 ? "tier-ground" : "tier-onboard"}">${i === 9 ? "GROUND" : "ONBOARD"}</span>
      <b>${esc(t)}</b>
      <span>${esc(d)}</span>
      <div class="step-live" style="display:none;"></div>
    </li>`).join("");
  if (s.mode === "unavailable") { $("#try").insertAdjacentHTML("afterbegin", `<div class="notice err">No results are available yet. Run <code>python scripts/evaluate.py --allow-placeholder</code> (placeholder) or follow the runbook in README.md.</div>`); return; }

  $("#hudTarget").textContent = ST.event;
  ST.dial = await guard(API.dial);
  ST.demo = await API.demoCase().then((d) => (d.available ? d : null)).catch(() => null);   // the demo tile is the opening selection
  $("#dial").innerHTML = s.dial_targets.map((a) => `<button type="button" data-a="${a}" aria-pressed="${a === ST.alpha}">${Math.round(a * 100)}%</button>`).join("");
  $("#dial").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) setAlpha(+b.dataset.a); });
  $("#event").innerHTML = [...s.events].sort((a, b) => (b.split === "holdout") - (a.split === "holdout") || a.event.localeCompare(b.event)).map((e) =>
    `<option value="${esc(e.event)}">${esc(e.event)}${e.split === "holdout" ? " (held out, never trained on)" : ""} · ${e.tiles} tiles · ${e.critical} critical</option>`).join("");
  $("#event").addEventListener("change", (e) => { ST.event = e.target.value; loadChips(); });
  $("#gridview").addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    ST.view = b.dataset.v;
    document.querySelectorAll("#gridview button").forEach((x) => x.setAttribute("aria-pressed", x === b));
    paintGrid();
  });
  $("#budget").addEventListener("input", (e) => ($("#budgetLabel").textContent = Math.round(e.target.value * 100) + "%"));
  $("#chip").addEventListener("change", (e) => loadTiles(e.target.value));
  $("#upcloud").addEventListener("input", (e) => ($("#upcloudv").textContent = Math.round(e.target.value * 100) + "%"));
  $("#upgo").addEventListener("click", uploadTile);
  $("#simgo").addEventListener("click", () => runSim({ redraw_clouds: true, cloud_seed: 42 }));
  $("#simnew").addEventListener("click", () => runSim({ redraw_clouds: true }));
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
    note.innerHTML = `Dial target <b>${Math.round(a * 100)}%</b>: the ${zero.join(" and ")} path has too few critical calibration tiles (a 1% target needs 99), so its cut-off is 0. <b>It keeps every tile on that path</b>, missing none.`;
  } else { note.hidden = true; note.textContent = ""; }
  paintGrid();
  if (ST.sel && !quiet) selectTile(ST.sel);
}

/* ------------------------------------------------------------------ try it */
async function loadChips() {
  const r = await guard(() => API.chips(ST.event), $("#grid"));
  if (!r) return;
  const sel = $("#chip");
  sel.innerHTML = r.chips.map((c) => `<option value="${esc(c.chip)}">${esc(c.chip)} · ${esc(c.split)} · ${c.tiles} tiles · ${c.critical} critical · ${Math.round(c.mean_cloud * 100)}% cloud</option>`).join("");
  const held = r.chips.every((c) => c.split === "holdout");
  $("#eventnote").textContent = held
    ? "Bolivia is the held-out flood: no model ever trained or calibrated on it, so this is the honest test."
    : `${ST.event}: its chips were assigned to train, valid and test by the dataset, and each chip shows its split. The models trained on the train chips, so scores there are not a fair test.`;
  const place = EVENT_PLACE[ST.event];
  if (place && window.setTarget3D) window.setTarget3D(place[0], place[1]);
  if ($("#hudTarget")) $("#hudTarget").textContent = ST.event;
  const demo = ST.demo && r.chips.find((c) => c.chip === ST.demo.chip);
  const best = demo || r.chips.find((c) => c.critical > 0 && c.mean_cloud > 0.45) || r.chips.find((c) => c.critical > 0) || r.chips[0];
  if (best) { sel.value = best.chip; loadTiles(best.chip); }
}

async function loadTiles(chip) {
  ST.chip = chip; ST.sel = null;
  const r = await guard(() => API.tiles(chip), $("#grid"));
  if (!r) return;
  ST.tiles = r.tiles;
  paintGrid();
  const first = (ST.demo && chip === ST.demo.chip && r.tiles.find((t) => t.tile_id === ST.demo.tile_id)) || r.tiles.find((t) => t.critical) || r.tiles[0];
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
      <img loading="lazy" alt="" src="${API.imageUrl(t.tile_id, ST.view, 2)}" onerror="this.outerHTML='<div class=&quot;nopx&quot;>imagery not on disk</div>'">
      <span class="badge ${t.sensor_used === "radar" ? "r" : ""}">${t.sensor_used === "radar" ? "R" : "O"}</span>${t.critical ? '<span class="crit" title="critical">★</span>' : ""}</button>`;
  }
  grid.innerHTML = h;
  grid.querySelectorAll("button.tile").forEach((b) => b.addEventListener("click", () => selectTile(b.dataset.id)));
  if (window.setChip3D) window.setChip3D(ST.tiles, ST.alpha);

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
    const t = ST.tiles.find((x) => x.tile_id === id);
    if (t && window.highlight3DTile) window.highlight3DTile(t.row * 4 + t.col);
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
  const img = (kind, label) => `<figure><img alt="${label}" src="${API.imageUrl(id, kind, 3)}" onerror="this.outerHTML='<div class=&quot;ph&quot;>not available</div>'">${label}</figure>`;
  const modelRows = ["optical", "radar"].filter((m) => both[m]).map((m) =>
    `<dt>${m} model (${esc(both[m].model)})</dt><dd>${F.n(both[m].p_flood)} flood probability${m === r.routed_to ? " <b>(used)</b>" : ""}</dd>`).join("");
  el.innerHTML = `
    ${withImages ? `<div class="imgs">${img("clear", "infrared, no cloud")}${img("optical", "what the camera sees")}${img("radar", "radar (water is dark)")}${img("label", "hand label: lime = flood")}</div>` : ""}
    <h3 style="margin-top:14px">${esc(id)}</h3>
    <p class="small">Cloud cover <b>${F.pct(r.cloud_frac, 0)}</b> ${r.routed_to === "radar" ? "≥" : "<"} ${F.pct(r.cloud_switch, 0)}, so this tile is routed to the <b>${esc(r.routed_to)}</b> model.
      A system that discards cloudy frames would <b>${esc(r.discard_cloudy_would)}</b>.</p>
    <div class="meter" aria-hidden="true"><div class="fill" style="width:${(r.p_flood * 100).toFixed(1)}%"></div><div class="thr" style="left:${(r.threshold * 100).toFixed(1)}%" title="dial threshold"></div></div>
    <p class="small">Flood probability <b>${F.n(r.p_flood)}</b>, dial threshold <b>${F.n(r.threshold)}</b> (lime marker, ${Math.round(r.alpha * 100)}% target miss rate)${r.threshold_is_send_everything ? ": threshold 0, keeps everything" : ""}.</p>
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
async function runSim(opts) {
  if (ST.simTimer) { clearInterval(ST.simTimer); ST.simTimer = null; }
  const out = $("#simout"), btns = [$("#simnew"), $("#simgo")];
  btns.forEach((b) => (b.disabled = true));
  out.innerHTML = `<p class="muted glass">Simulating clouds and running the optical model on every tile…</p>`;
  const chipScope = $("#scope").value === "chip";
  const body = { event: chipScope ? null : ST.event, chip: chipScope ? ST.chip : null, alpha: ST.alpha, scaled_fraction: +$("#budget").value, ...opts };
  const r = await guard(() => API.simulate(body), out);
  btns.forEach((b) => (b.disabled = false));
  if (!r) return;
  const curves = {}; Object.entries(r.systems).forEach(([n, s]) => (curves[n] = s.curve));
  const rows = Object.entries(r.systems).map(([n, s]) => ({ system: n, ...s.summary }));
  const orbit = r.systems["ORBIT-PRISM"];
  const passes = orbit ? orbit.passes : [];
  const total = passes.reduce((a, p) => a + p.MB, 0);
  out.innerHTML = `
    <div class="notice"><b>Cloud seed ${r.weather.cloud_seed}${r.weather.cloud_seed === 42 ? " (the measured clouds)" : ""}</b> · mean cloud ${F.pct(r.weather.mean_cloud, 0)} · ${r.weather.radar_routed} of ${r.selection.tiles} tiles routed to radar · ${r.selection.critical} critical · dial ${Math.round(r.alpha * 100)}% · ${esc(r.scaled_note)} (${r.scaled_budget_MB.toFixed(2)} MB).</div>
    ${r.selection.critical === 0 ? `<div class="notice err">No critical tiles in this selection, so recall is undefined. Choose a chip with a ★ tile or the whole event.</div>` : ""}
    <div class="verdict">${r.dial.map(dialVerdict).join("")}</div>
    <div class="card chart" id="simchart"></div>
    <div class="card"><div class="tablewrap">${summaryTable(rows)}</div></div>
    ${Object.keys(r.skipped).length ? `<div class="notice">Not run: ${Object.entries(r.skipped).map(([k, v]) => esc(k + " (" + v + ")")).join("; ")}</div>` : ""}
    ${orbit && passes.length ? `
      <div class="card" id="budgetProgress">
        <div><b>Downlink sequence.</b> ORBIT-PRISM queues <b>${total.toFixed(3)} MB</b> in total. Each pass carries at most <b>${r.scaled_budget_MB.toFixed(2)} MB</b>, so it takes <b>${passes.length} pass${passes.length > 1 ? "es" : ""}</b>; what does not fit waits for the next one.</div>
        <div class="pass-bars" id="passBars">${passes.map((p) => `<div class="pass-bar" data-p="${p.pass_index}"><div class="meter"><div class="fill" style="width:0%"></div></div><span>pass ${p.pass_index + 1}: ${p.MB.toFixed(3)} MB</span></div>`).join("")}</div>
        <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap">
          <button type="button" class="btn play" id="playQueueBtn">▶ Play downlink</button>
          <span class="small"><span id="budgetVal">pass 1: 0.000 / ${r.scaled_budget_MB.toFixed(2)} MB</span> · <span class="muted" id="budgetPct">0.000 MB sent in total</span></span>
          <span id="playStatus" class="small muted"></span>
        </div>
      </div>
      <div class="card">
        <h3 class="sub first">ORBIT-PRISM send queue (${orbit.queue_preview.length === orbit.total_packets ? "all" : "sample of"} ${orbit.total_packets.toLocaleString()} packets)</h3>
        <div class="queue" id="queueTableWrap">${queueTable(orbit.queue_preview)}</div>
      </div>` : ""}`;
  recallChart($("#simchart"), curves, { scaledMB: r.scaled_budget_MB });
  if (orbit && $("#playQueueBtn")) $("#playQueueBtn").addEventListener("click", () => playQueueAnimation(orbit.queue_preview, r.scaled_budget_MB));
  addRun(r, rows);
}

function dialVerdict(d) {
  if (!d.critical) return `<div><b>${esc(d.path)} path</b>no critical tiles on this path in this run</div>`;
  const held = d.achieved <= d.target;
  return `<div class="${held ? "held" : "broke"}"><b>${esc(d.path)} path: ${held ? "promise held" : "promise broken"}</b>${d.missed} of ${d.critical} critical tiles missed = ${F.pct(d.achieved)}, target ${F.pct(d.target, 0)}. One tile is ${F.pct(1 / d.critical)} here.</div>`;
}

function addRun(r, rows) {
  const by = Object.fromEntries(rows.map((x) => [x.system, x]));
  const d = Object.fromEntries(r.dial.map((x) => [x.path, x]));
  const cell = (x) => (x && x.critical ? `<td class="${x.achieved <= x.target ? "ok" : "no"}">${x.missed}/${x.critical}</td>` : "<td>–</td>");
  ST.runs.unshift({ html: `<tr><td>${ST.runs.length + 1}</td><td>${esc(r.selection.event || r.selection.chip)}</td><td>${r.weather.cloud_seed}</td><td>${F.pct(r.weather.mean_cloud, 0)}</td><td>${Math.round(r.alpha * 100)}%</td><td>${Math.round(r.scaled_fraction * 100)}%</td><td>${F.pct((by["ORBIT-PRISM"] || {}).recall_at_scaled_pass)}</td><td>${F.pct((by["Optical-only triage"] || {}).recall_at_scaled_pass)}</td><td>${F.pct((by["Colour-index triage"] || {}).recall_at_scaled_pass)}</td><td>${F.pct((by["Discard cloudy"] || {}).recall_at_scaled_pass)}</td>${cell(d.optical)}${cell(d.radar)}</tr>` });
  $("#simhist").hidden = false;
  $("#simhisttable").innerHTML = `<table><tr><th>run</th><th>selection</th><th>cloud seed</th><th>cloud</th><th>dial</th><th>pass</th><th>ORBIT-PRISM recall</th><th>optical-only</th><th>colour index</th><th>discard cloudy</th><th>optical missed</th><th>radar missed</th></tr>${ST.runs.map((x) => x.html).join("")}</table>`;
}

function playQueueAnimation(queue, budgetMB) {
  if (ST.simTimer) { clearInterval(ST.simTimer); ST.simTimer = null; }
  const btn = $("#playQueueBtn"), val = $("#budgetVal"), pct = $("#budgetPct"), status = $("#playStatus");
  const rows = document.querySelectorAll("#queueTableWrap table tr");
  document.querySelectorAll("#passBars .fill").forEach((f) => (f.style.width = "0%"));
  const step = Math.max(1, Math.ceil(queue.length / 150));
  let idx = 0;
  btn.textContent = "⏸ Playing…";
  status.textContent = "";
  ST.simTimer = setInterval(() => {
    if (idx >= queue.length) {
      clearInterval(ST.simTimer); ST.simTimer = null;
      btn.textContent = "↺ Replay";
      status.textContent = "Done.";
      return;
    }
    idx = Math.min(queue.length, idx + step);
    const pkt = queue[idx - 1], inPass = pkt.in_pass_bytes / 1e6;
    const bar = document.querySelector(`#passBars .pass-bar[data-p="${pkt.pass_index}"]`);
    document.querySelectorAll("#passBars .pass-bar").forEach((b) => b.classList.toggle("now", b === bar));
    if (bar) bar.querySelector(".fill").style.width = Math.min(100, (inPass / budgetMB) * 100).toFixed(1) + "%";
    val.textContent = `pass ${pkt.pass_index + 1}: ${inPass.toFixed(3)} / ${budgetMB.toFixed(2)} MB`;
    pct.textContent = `${(pkt.cum_bytes / 1e6).toFixed(3)} MB sent in total`;
    rows.forEach((r) => r.classList.remove("active-pkt"));
    if (rows[idx]) { rows[idx].classList.add("active-pkt"); const w = $("#queueTableWrap"); w.scrollTop = rows[idx].offsetTop - w.clientHeight / 2; }
  }, 90);
}

function queueTable(q) {
  return `<table><tr><th>#</th><th>layer</th><th>tile</th><th>bytes</th><th>MB in this pass</th><th>MB in total</th><th>pass</th></tr>` + q.map((p) =>
    `<tr><td>${p.order}</td><td style="text-align:left"><span class="pill ${esc(p.layer)}">${esc(p.layer)}</span></td><td style="text-align:left">${esc(p.tile_id)}${p.critical ? " ★" : ""}</td><td>${p.bytes.toLocaleString()}</td><td>${(p.in_pass_bytes / 1e6).toFixed(3)}</td><td>${(p.cum_bytes / 1e6).toFixed(3)}</td><td>${p.pass_index + 1}</td></tr>`).join("") + `</table>`;
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
  const nCal = ((dial.targets || [])[0] || {}).critical_calibration_tiles || {};
  const dialRows = (rows, label) => rows.map((d) => {
    const all = Number(d.threshold) === 0, ok = d.achieved_miss_rate <= d.target_miss_rate;
    return `<tr><td>${label}</td><td>${esc(d.path)}</td><td>${F.pct(d.target_miss_rate, 0)}</td><td>${F.n(d.threshold)}</td><td>${d.critical_tiles}</td><td>${d.missed}</td><td class="${all ? "" : ok ? "ok" : "no"}">${all ? "not met: keeps every tile" : F.pct(d.achieved_miss_rate) + (ok ? " held" : " over")}</td></tr>`;
  }).join("");
  $("#dialtable").innerHTML = dial.measured ? (dial.measured.length ? `<table><tr><th>flood data</th><th>path</th><th>target miss rate</th><th>cut-off</th><th>critical tiles</th><th>missed</th><th>achieved miss rate</th></tr>` + dialRows(dial.measured, "Bolivia (unseen flood)") + dialRows(dial.measured_test || [], "test chips (seen floods)") + `</table><p class="small muted">A 1% target needs at least 99 critical calibration tiles. Radar scores do not depend on simulated cloud, so the radar dial calibrates on all ${nCal.radar} critical validation tiles. The optical model is blind under cloud, so its dial can use only the ${nCal.optical} clear ones; at 1% its cut-off is 0 and it keeps every tile (nothing missed, nothing saved).</p>` : "<p class='muted'>Not available yet.</p>") : `<div class="notice err">${esc(dial.error)}</div>`;
  if (dial.measured && dial.measured.length) {
    const lines = ["optical", "radar"].map((path) => {
      const rows = dial.measured.filter((d) => d.path === path && Number(d.threshold) > 0);
      const held = rows.every((d) => d.achieved_miss_rate <= d.target_miss_rate);
      const text = rows.map((d) => `${F.pct(d.target_miss_rate, 0)} target: ${d.missed} of ${d.critical_tiles} missed (${F.pct(d.achieved_miss_rate)})`).join("; ");
      return `<div class="${held ? "held" : "broke"}"><b>${path} path: ${held ? "held on this flood" : "did not hold on this flood"}</b>${text}.</div>`;
    });
    const n = dial.measured[0].critical_tiles;
    $("#dialverdict").innerHTML = `<div class="verdict">${lines.join("")}</div><p class="small muted">With ${n} critical tiles per path, one tile moves the rate by ${F.pct(1 / n)}, so this is thin evidence. The test chips are a second check, on floods the calibration also saw. A promise only holds when test data resembles the calibration data. Use the link simulator to see how the rates move under other clouds.</p>`;
  }
  $("#memory").innerHTML = mem.rows && mem.rows.length ? `<table><tr><th>model</th><th>dtype</th><th>weights KB</th><th>peak activation KB (est.)</th><th>total KB</th><th>% of 4.2 MB</th><th>MACs</th><th>recall</th><th>ece</th><th>basis</th></tr>` + mem.rows.map((m) =>
    `<tr><td>${esc(m.model)}</td><td>${esc(m.dtype)}</td><td>${F.kb(m.weights_bytes)}</td><td>${F.kb(m.peak_activation_bytes)}</td><td>${F.kb(m.total_bytes)}</td><td>${F.pct(m.total_bytes / mem.sram_bytes)}</td><td>${Number(m.macs).toLocaleString()}</td><td>${esc(m.recall)}</td><td>${esc(m.ece)}</td><td>${esc(m.basis)}</td></tr>`).join("") + `</table><p class="small muted">Weights are measured; peak activation is estimated from the model graph. Recall and ECE are on the validation split at the float model's 90%-recall cut-off. Radar INT8 recall of 1 means that cut-off keeps almost every tile (INT8 radar scores are compressed upward), not a better model.</p>` : `<p class="muted">The memory table has not been produced yet (it comes from the INT8 step).</p>`;
  if (demo.available) renderDemoCase(demo);
  else $("#democase").innerHTML = `<div class="muted small">The demo tile is chosen from real scores once the evaluation has been run.</div>`;
}

function renderDemoCase(d) {
  const del = d.full_tile_delivery || {}, cl = d.cloudy_critical_tiles_delivered || {};
  const sent = (n) => (del[n] ? `sent after ${del[n].cum_MB.toFixed(2)} MB` : "not delivered in the pass");
  const fig = (kind, title, note) => `<figure><img alt="${title}" src="${API.imageUrl(d.tile_id, kind, 4)}"><b>${title}</b>${note}</figure>`;
  $("#democase").innerHTML = `
    <div class="imgs">
      ${fig("clear", "Infrared, clear day", "flood water is dark blue")}
      ${fig("optical", "Camera under cloud", `${F.pct(d.cloud_frac, 0)} simulated cloud, drawn with real cloud structure`)}
      ${fig("radar", "Radar", "sees through cloud: smooth water is dark")}
      ${fig("label", "Hand label (truth)", `${F.pct(d.truth.water_frac, 0)} flood in lime`)}
    </div>
    <div class="flow">
      <div><b>${F.pct(d.cloud_frac, 0)} cloud</b>≥ 50%, so routed to ${esc(d.routed_to)}</div>
      <div><b>${F.n(d.scores.p_radar, 2)} vs ${F.n(d.thresholds.radar, 2)}</b>radar score against the ${Math.round(d.dial_target * 100)}% cut-off: ${d.kept_by_dial ? "kept" : "skipped"}</div>
      <div><b>ORBIT-PRISM</b>${sent("ORBIT-PRISM")}</div>
      <div><b>Discard cloudy</b>${sent("Discard cloudy")}</div>
      <div><b>Optical-only</b>${sent("Optical-only triage")}</div>
      <div><b>${cl["ORBIT-PRISM"]} of ${cl.total}</b>critical cloudy tiles delivered by ORBIT-PRISM; optical-only ${cl["Optical-only triage"]}, capture order ${cl["Capture order"]}, discard cloudy ${cl["Discard cloudy"]}</div>
    </div>
    <p class="small muted" style="margin-top:10px">${esc(d.tile_id)}. ${esc((d.caveats || []).join(" "))}</p>`;
}

init();
