/* Recall-versus-data chart as plain SVG (no library, works offline). Colours and dash patterns are the
   validated palette used by src/evaluation/plotting.py (dark theme), so the web chart matches the PNGs. */
const CHART_STYLE = {
  "ORBIT-PRISM":         { c: "#3987e5", d: "" },
  "Capture order":       { c: "#d95926", d: "6 3" },
  "Discard cloudy":      { c: "#199e70", d: "2 2" },
  "Optical-only triage": { c: "#c98500", d: "8 3 2 3" },
  "Colour-index triage": { c: "#d55181", d: "1 2" },
  "No radar (ablation)": { c: "#d95926", d: "6 3" },
  "No dial (ablation)":  { c: "#199e70", d: "2 2" },
};

function recallChart(el, curves, opts = {}) {
  const W = 760, H = 380, m = { l: 56, r: 24, t: 16, b: 48 };
  const names = Object.keys(curves).filter((n) => curves[n] && curves[n].length);
  const xmax = Math.max(0.001, ...names.map((n) => curves[n][curves[n].length - 1].cum_MB)) * 1.08;
  const X = (v) => m.l + (v / xmax) * (W - m.l - m.r), Y = (v) => H - m.b - v * (H - m.t - m.b);
  const ticks = (max, n) => { const raw = max / n, p = 10 ** Math.floor(Math.log10(raw)), s = [1, 2, 5, 10].find((k) => k * p >= raw) * p; const o = []; for (let v = 0; v <= max; v += s) o.push(+v.toFixed(6)); return o; };
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Critical-tile recall against cumulative data sent, one line per system" xmlns="http://www.w3.org/2000/svg">`;
  for (let v = 0; v <= 1.0001; v += 0.2) s += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(v)}" y2="${Y(v)}" class="grid"/><text x="${m.l - 8}" y="${Y(v) + 4}" text-anchor="end" class="tick">${Math.round(v * 100)}%</text>`;
  ticks(xmax, 7).forEach((v) => s += `<line x1="${X(v)}" x2="${X(v)}" y1="${m.t}" y2="${H - m.b}" class="grid"/><text x="${X(v)}" y="${H - m.b + 18}" text-anchor="middle" class="tick">${v}</text>`);
  s += `<text x="${(m.l + W - m.r) / 2}" y="${H - 8}" text-anchor="middle" class="axis">Cumulative data sent (MB)</text>`;
  s += `<text transform="rotate(-90)" x="${-(H - m.b + m.t) / 2}" y="14" text-anchor="middle" class="axis">Critical-tile recall (full tiles delivered)</text>`;
  if (opts.scaledMB && opts.scaledMB < xmax)
    s += `<line x1="${X(opts.scaledMB)}" x2="${X(opts.scaledMB)}" y1="${m.t}" y2="${H - m.b}" class="scaled"/><text x="${X(opts.scaledMB) + 6}" y="${H - m.b - 8}" class="tick">scaled pass ${opts.scaledMB.toFixed(2)} MB</text>`;
  names.forEach((n) => {
    const st = CHART_STYLE[n] || { c: "#c3c2b7", d: "" }, pts = curves[n];
    let d = `M${X(pts[0].cum_MB)},${Y(pts[0].recall)}`;
    for (let i = 1; i < pts.length; i++) d += `H${X(pts[i].cum_MB)}V${Y(pts[i].recall)}`;      // step function
    s += `<path d="${d}" fill="none" stroke="${st.c}" stroke-width="2" stroke-dasharray="${st.d}" stroke-linejoin="round"><title>${n}</title></path>`;
    if (n === "ORBIT-PRISM") { const e = pts[pts.length - 1]; s += `<circle cx="${X(e.cum_MB)}" cy="${Y(e.recall)}" r="4.5" fill="${st.c}" stroke="#1a1a19" stroke-width="2"/><text x="${X(e.cum_MB) - 6}" y="${Y(e.recall) - 10}" text-anchor="end" class="direct">ORBIT-PRISM ${Math.round(e.recall * 100)}%</text>`; }
  });
  s += "</svg>";
  const legend = names.map((n) => { const st = CHART_STYLE[n] || { c: "#c3c2b7", d: "" }; return `<li><svg width="34" height="10" aria-hidden="true"><line x1="0" x2="34" y1="5" y2="5" stroke="${st.c}" stroke-width="2" stroke-dasharray="${st.d}"/></svg>${n}</li>`; }).join("");
  el.innerHTML = s + `<ul class="legend">${legend}</ul>`;
}
