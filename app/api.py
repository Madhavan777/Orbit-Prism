"""ORBIT-PRISM web API: serves the website with real model inputs and outputs.

One process serves the JSON API under /api and the static website in web/ at /.
    python -m uvicorn app.api:app --port 8000        then open http://localhost:8000

What it serves (all read-only; nothing here trains or changes a result):
    results       summary.csv, curves.csv, dial.csv, memory_table.csv from scripts/evaluate.py
    tiles         one row per tile (data/tiles_all.csv, or data/fake_tiles.csv in placeholder mode)
    images        tile pictures rendered from data/processed/<event>.npz and the raw labels
    triage        a tile through the real models: routing by cloud, scores, the dial's keep/skip decision,
                  the 32-byte alert packet and the priority; also for an uploaded tile
    simulate      the five systems and ablations over a chip, an event or a split, with the send queue

MODES. "real" needs results/summary.csv + results/run_info.json (synthetic false) + data/tiles_all.csv.
"placeholder" uses results/synthetic/ and data/fake_tiles.csv, and EVERY response says synthetic: true.
"unavailable" means neither exists; /api/status still answers and says what to run.
"""
from __future__ import annotations

import functools
import importlib.util
import io
import json
import struct
import sys
import zlib
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.constants import CLOUD_SWITCH, L0_B, L1_BYTES, OPTICAL_B, PASS_BYTES, RADAR_B, SCALED_PASS_FRACTION, SRAM_BYTES  # noqa: E402
from src.evaluation import metrics  # noqa: E402
from src.evaluation.reference_policies import NotAvailable  # noqa: E402
from src.scheduler import sim as scheduler_sim  # noqa: E402 (the scheduler of record: policies, packets, passes)
from src.scheduler.sim import POLICIES  # noqa: E402

DIAL_TARGETS = [0.01, 0.05, 0.10]
SYSTEM_NAMES = {
    "orbit_prism": "ORBIT-PRISM", "capture_order": "Capture order", "discard_cloudy": "Discard cloudy",
    "optical_only": "Optical-only triage", "colour_index": "Colour-index triage",
    "orbit_no_radar": "No radar (ablation)", "orbit_no_dial": "No dial (ablation)",
}
MAX_UPLOAD = 2_000_000


def _clean(obj):
    """Make a structure JSON-safe: NaN and inf become null, numpy scalars become Python numbers."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _frame(df: pd.DataFrame) -> List[dict]:
    return _clean(df.astype(object).where(df.notna(), None).to_dict("records"))


def pack_alert(chip: str, row: int, col: int, sensor: str, p: float, extent: float) -> bytes:
    """The 32-byte L0 alert. The layout is owned by the scheduler (src/scheduler/sim.py,
    build_l0_alert_packet); this only calls it, so there is one definition of the packet."""
    return scheduler_sim.build_l0_alert_packet(chip, row, col, sensor, p, extent, pass_number=0, seq=0)


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Models:
    """The trained models, loaded on first use: INT8 TFLite if present, else the float Keras model."""

    def __init__(self, models_dir: Path):
        self.dir, self._cache = models_dir, {}

    def _load(self, name: str):
        if name in self._cache:
            return self._cache[name]
        found = None
        int8, flt = self.dir / f"{name}_int8.tflite", self.dir / f"{name}_float.keras"
        if int8.exists():
            from src.compress.predict_int8 import Int8ModelRunner
            runner = Int8ModelRunner(int8)
            found = ("int8", lambda tile: runner.predict_single(tile))
        elif flt.exists():
            import tensorflow as tf
            model = tf.keras.models.load_model(flt)

            def run(tile, model=model):
                p, e = model.predict(tile[None], verbose=0)
                return float(p.ravel()[0]), float(e.ravel()[0])
            found = ("float", run)
        self._cache[name] = found
        return found

    def kind(self, name: str) -> Optional[str]:
        if name in self._cache:
            return self._cache[name][0] if self._cache[name] else None
        if (self.dir / f"{name}_int8.tflite").exists():
            return "int8"
        return "float" if (self.dir / f"{name}_float.keras").exists() else None

    def predict(self, name: str, tile: np.ndarray):
        """(p_flood, extent, kind) for one tile, or None. Optical takes raw uint8 128x128x4; radar takes
        uint8 128x128x2, scaled to [0, 1] here."""
        loaded = self._load(name)
        if loaded is None:
            return None
        kind, fn = loaded
        arr = tile.astype(np.uint8) if name == "optical" else (tile.astype(np.float32) / 255.0)
        p, e = fn(arr)
        return float(np.clip(p, 0, 1)), float(np.clip(e, 0, 1)), kind


class State:
    """Everything the API reads, loaded once from the project folder."""

    def __init__(self, root: Path):
        self.root = root
        self.results, self.mode, self.synthetic, self.tiles, self.cal = root / "results", "unavailable", True, None, None
        self.run_info: dict = {}
        real_ok = ((root / "results" / "summary.csv").exists() and (root / "results" / "run_info.json").exists()
                   and (root / "data" / "tiles_all.csv").exists())
        if real_ok:
            info = json.loads((root / "results" / "run_info.json").read_text())
            if not info.get("synthetic", True):
                self.mode, self.synthetic, self.run_info = "real", False, info
        if self.mode != "real" and (root / "results" / "synthetic" / "summary.csv").exists() \
                and (root / "data" / "fake_tiles.csv").exists():
            self.mode, self.synthetic = "placeholder", True
            self.results = root / "results" / "synthetic"
            self.run_info = json.loads((self.results / "run_info.json").read_text())
        if self.mode == "unavailable":
            return
        tiles = pd.read_csv(root / "data" / ("tiles_all.csv" if self.mode == "real" else "fake_tiles.csv"))
        if self.mode == "placeholder":
            tiles = _load_script("evaluate").synthetic_extra_scores(tiles)   # invented p_optical / p_ndwi
        tiles["tile_id"] = tiles["chip"] + "_" + tiles["row"].astype(str) + "_" + tiles["col"].astype(str)
        self.tiles = tiles.set_index("tile_id", drop=False)
        self.cal = self._thresholds()
        self.models = Models(root / "models")

    def _thresholds(self) -> dict:
        if self.mode == "real":
            cal = json.loads((self.results / "calibration.json").read_text())
            src = str(cal.get("score_source", ""))
            if not src or src.startswith("synthetic") or src == "unspecified":
                raise RuntimeError(f"calibration.json score_source is '{src}': not made from real scores")
            return cal
        from src.calibration.calibrate import CalibrationPipeline
        pipe = CalibrationPipeline(target_alphas=DIAL_TARGETS)
        pipe.score_source = "synthetic_placeholder (in-process)"
        pipe.fit_and_calibrate(self.tiles.reset_index(drop=True), calib_splits=["valid"])
        return pipe.to_dict()

    def thresholds(self, alpha: float) -> Dict[str, float]:
        key = str(alpha)
        if key not in self.cal["optical"]:
            raise HTTPException(422, f"alpha must be one of {DIAL_TARGETS}")
        return {"optical": float(self.cal["optical"][key]), "radar": float(self.cal["radar"][key])}

    def table(self, name: str) -> pd.DataFrame:
        path = self.results / name
        return pd.read_csv(path) if path.exists() else pd.DataFrame()


@functools.lru_cache(maxsize=3)
def _event_arrays(npz_path: str):
    with np.load(npz_path) as z:
        ids = z["tile_id"]
        return {"index": {str(t): i for i, t in enumerate(ids)}, "optical": z["optical"], "radar": z["radar"]}


def _png(arr: np.ndarray, scale: int) -> Response:
    img = Image.fromarray(arr)
    if scale > 1:
        img = img.resize((arr.shape[1] * scale, arr.shape[0] * scale), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png", headers={"Cache-Control": "max-age=3600"})


def _optical_rgb(tile: np.ndarray) -> np.ndarray:
    """Display only: bands B4, B3, B2 (channels 2, 1, 0) with a fixed gain and gamma. Not model input."""
    rgb = tile[..., [2, 1, 0]].astype(np.float32) / 255.0
    return (np.clip((rgb * 3.0) ** 0.8, 0, 1) * 255).astype(np.uint8)


class TriageIn(BaseModel):
    tile_id: str
    alpha: float = Field(0.05, description="target miss rate: 0.01, 0.05 or 0.1")


class SimulateIn(BaseModel):
    split: str = "holdout"
    event: Optional[str] = None
    chip: Optional[str] = None
    alpha: float = 0.05
    scaled_fraction: float = Field(SCALED_PASS_FRACTION, gt=0, le=1)
    policies: List[str] = ["orbit_prism", "capture_order", "discard_cloudy", "optical_only", "colour_index"]


def create_app(root: Path = ROOT) -> FastAPI:
    app = FastAPI(title="ORBIT-PRISM", version="1.0")
    S = State(root)

    def need_data():
        if S.mode == "unavailable":
            raise HTTPException(503, "No results found. Run: python scripts/evaluate.py --allow-placeholder "
                                     "(placeholder) or the real runbook in README.md")

    def row_of(tile_id: str) -> pd.Series:
        need_data()
        if tile_id not in S.tiles.index:
            raise HTTPException(404, f"unknown tile {tile_id}")
        return S.tiles.loc[tile_id]

    def tile_arrays(row: pd.Series):
        path = root / "data" / "processed" / f"{row['event']}.npz"
        if not path.exists():
            return None
        arrays = _event_arrays(str(path))
        i = arrays["index"].get(row["tile_id"])
        return None if i is None else (arrays["optical"][i], arrays["radar"][i])

    def triage(row_meta: dict, optical: Optional[np.ndarray], radar: Optional[np.ndarray], alpha: float,
               stored: Optional[dict]) -> dict:
        sensor = "radar" if row_meta["cloud_frac"] >= CLOUD_SWITCH else "optical"
        thr = S.thresholds(alpha)[sensor]
        live = {}
        for name, tile in (("optical", optical), ("radar", radar)):
            out = S.models.predict(name, tile) if tile is not None else None
            if out:
                live[name] = {"p_flood": out[0], "extent": out[1], "model": out[2]}
        used = live.get(sensor)
        source = "live model" if used else ("stored score" if stored else None)
        if used:
            p, ext = used["p_flood"], used["extent"]
        elif stored:
            p, ext = stored["p_flood"], stored["extent"]
        else:
            raise HTTPException(409, f"no {sensor} model in models/ and no stored score for this tile")
        kept = bool(p >= thr)
        l2 = RADAR_B if sensor == "radar" else OPTICAL_B
        priority = p * ext
        alert = pack_alert(row_meta["chip"], int(row_meta["row"]), int(row_meta["col"]), sensor, p, ext)
        res = {
            "synthetic": S.synthetic, "tile_id": row_meta["tile_id"], "alpha": alpha,
            "cloud_frac": float(row_meta["cloud_frac"]), "cloud_switch": CLOUD_SWITCH, "routed_to": sensor,
            "score_source": source, "p_flood": p, "extent": ext, "threshold": thr,
            "threshold_is_send_everything": thr == 0.0, "decision": "KEEP" if kept else "SKIP",
            "priority": priority, "l2_bytes": l2, "alert_hex": alert.hex(), "alert_bytes": len(alert),
            "both_models": live,
            "discard_cloudy_would": "drop it" if row_meta["cloud_frac"] >= CLOUD_SWITCH else "send it",
            "stored": stored,
        }
        if used and stored:
            res["live_vs_stored_abs_diff"] = abs(p - stored["p_flood"])
        if "critical" in row_meta and row_meta["critical"] is not None:
            crit = int(row_meta["critical"])
            res["truth"] = {"water_frac": float(row_meta["water_frac"]), "critical": crit}
            res["outcome"] = ("caught: critical tile kept" if crit and kept else
                              "MISSED: critical tile skipped" if crit else
                              "false alarm: sent unnecessarily" if kept else "correctly skipped")
        return _clean(res)

    @app.get("/api/status")
    def status():
        info = {
            "mode": S.mode, "synthetic": S.synthetic,
            "banner": "SYNTHETIC PLACEHOLDER SCORES: NOT RESULTS" if S.synthetic else None,
            "headline_split": "holdout", "scaled_pass_fraction": SCALED_PASS_FRACTION,
            "link": {"kbps": 256, "seconds_per_pass": 600, "pass_MB": PASS_BYTES / 1e6},
            "sram_MB": SRAM_BYTES / 1e6,
            "packets": {"L0": L0_B, "L1": L1_BYTES, "L2_optical": OPTICAL_B, "L2_radar": RADAR_B},
            "dial_targets": DIAL_TARGETS, "cloud_switch": CLOUD_SWITCH, "scheduler": "src/scheduler/sim.py",
            "notes": ["Clouds are simulated.", "Simulation of the link and memory limits, not flight hardware.",
                      "Floods only; forest fire data is not loaded."],
        }
        if S.mode == "unavailable":
            return _clean({**info, "models": {}, "images": False, "events": []})
        info["score_source"] = S.run_info.get("score_source")
        info["models"] = {"optical": S.models.kind("optical"), "radar": S.models.kind("radar")}
        info["images"] = any((root / "data" / "processed").glob("*.npz")) if (root / "data" / "processed").is_dir() else False
        ev = S.tiles.groupby("event").agg(tiles=("tile_id", "size"), critical=("critical", "sum"),
                                          split=("split", lambda s: s.mode().iloc[0]))
        info["events"] = _frame(ev.reset_index())
        return _clean(info)

    @app.get("/api/summary")
    def summary():
        need_data()
        return {"synthetic": S.synthetic, "rows": _frame(S.table("summary.csv"))}

    @app.get("/api/curves")
    def curves():
        need_data()
        df, out = S.table("curves.csv"), {}
        for name, g in df.groupby("system"):
            g = g.iloc[np.unique(np.linspace(0, len(g) - 1, min(len(g), 400)).astype(int))]
            out[name] = _frame(g[["cum_MB", "recall"]])
        info = S.run_info
        return {"synthetic": S.synthetic, "scaled_pass_MB": info.get("scaled_pass_MB"), "curves": out}

    @app.get("/api/dial")
    def dial():
        need_data()
        targets = []
        for a in DIAL_TARGETS:
            t = S.thresholds(a)
            n = S.cal.get("critical_calibration_tiles", {})
            targets.append({"target_miss_rate": a, "thresholds": t, "critical_calibration_tiles": n,
                            "sends_everything": {p: t[p] == 0.0 for p in t}})
        return {"synthetic": S.synthetic, "targets": targets, "measured": _frame(S.table("dial.csv")),
                "score_source": S.cal.get("score_source")}

    @app.get("/api/memory")
    def memory():
        need_data()
        return {"rows": _frame(S.table("memory_table.csv")) if (root / "results" / "memory_table.csv").exists() else [],
                "sram_bytes": SRAM_BYTES,
                "note": "Peak activation is ESTIMATED; weights are measured. recall and ece are on the valid split."}

    @app.get("/api/chips")
    def chips(split: str = "holdout", event: Optional[str] = None):
        need_data()
        t = S.tiles[S.tiles["split"] == split]
        if event:
            t = t[t["event"] == event]
        g = t.groupby(["event", "chip"]).agg(tiles=("tile_id", "size"), critical=("critical", "sum"),
                                              mean_cloud=("cloud_frac", "mean")).reset_index()
        return {"synthetic": S.synthetic, "chips": _frame(g)}

    @app.get("/api/tiles")
    def tiles(chip: str):
        need_data()
        t = S.tiles[S.tiles["chip"] == chip]
        if t.empty:
            raise HTTPException(404, f"unknown chip {chip}")
        out = []
        for r in t.itertuples():
            thr = {a: S.thresholds(a) for a in DIAL_TARGETS}
            kept = {str(a): bool(r.p_flood >= thr[a][r.sensor_used]) for a in DIAL_TARGETS}
            out.append({"tile_id": r.tile_id, "row": r.row, "col": r.col, "split": r.split, "event": r.event,
                        "water_frac": r.water_frac, "critical": r.critical, "cloud_frac": r.cloud_frac,
                        "sensor_used": r.sensor_used, "p_flood": r.p_flood, "extent": r.extent, "kept": kept})
        return {"synthetic": S.synthetic, "chip": chip, "tiles": _clean(out)}

    @app.get("/api/tile/{tile_id}/image")
    def tile_image(tile_id: str, kind: str = Query("optical", pattern="^(optical|radar|label)$"),
                   scale: int = Query(2, ge=1, le=6)):
        row = row_of(tile_id)
        if kind == "label":
            return _label_png(root, row, scale)
        arrays = tile_arrays(row)
        if arrays is None:
            raise HTTPException(404, "tile pixels not available: build them with python scripts/build_processed.py")
        optical, radar = arrays
        return _png(_optical_rgb(optical) if kind == "optical" else radar[..., 0], scale)

    @app.post("/api/triage")
    def triage_tile(body: TriageIn):
        row = row_of(body.tile_id)
        arrays = tile_arrays(row)
        stored = {"p_flood": float(row["p_flood"]), "extent": float(row["extent"])}
        meta = {k: row[k] for k in ("tile_id", "chip", "row", "col", "cloud_frac", "water_frac", "critical")}
        return triage(meta, arrays[0] if arrays else None, arrays[1] if arrays else None, body.alpha, stored)

    @app.post("/api/triage/upload")
    async def triage_upload(file: UploadFile = File(...), cloud_frac: float = Form(0.0), alpha: float = Form(0.05)):
        need_data()
        raw = await file.read(MAX_UPLOAD + 1)
        if len(raw) > MAX_UPLOAD:
            raise HTTPException(413, f"file larger than {MAX_UPLOAD} bytes")
        if not 0.0 <= cloud_frac <= 1.0:
            raise HTTPException(422, "cloud_frac must be between 0 and 1")
        try:
            with np.load(io.BytesIO(raw), allow_pickle=False) as z:
                optical = z["optical"] if "optical" in z.files else None
                radar = z["radar"] if "radar" in z.files else None
        except Exception:
            raise HTTPException(422, "upload a .npz with arrays 'optical' (128,128,4) and/or 'radar' (128,128,2), uint8")
        for name, arr, shape in (("optical", optical, (128, 128, 4)), ("radar", radar, (128, 128, 2))):
            if arr is not None and (arr.shape != shape or arr.dtype != np.uint8):
                raise HTTPException(422, f"'{name}' must be uint8 with shape {shape}")
        if optical is None and radar is None:
            raise HTTPException(422, "the .npz needs an 'optical' and/or a 'radar' array")
        routed = "radar" if cloud_frac >= CLOUD_SWITCH else "optical"
        if (radar if routed == "radar" else optical) is None:
            raise HTTPException(422, f"cloud_frac {cloud_frac} routes this tile to the {routed} model "
                                     f"(switch at {CLOUD_SWITCH}), so the upload needs a '{routed}' array")
        meta = {"tile_id": "uploaded", "chip": "uploaded", "row": 0, "col": 0, "cloud_frac": cloud_frac,
                "water_frac": None, "critical": None}
        return triage(meta, optical, radar, alpha, None)

    @app.post("/api/simulate")
    def simulate(body: SimulateIn):
        need_data()
        sel = S.tiles[S.tiles["split"] == body.split]
        if body.event:
            sel = sel[sel["event"] == body.event]
        if body.chip:
            sel = sel[sel["chip"] == body.chip]
        if sel.empty:
            raise HTTPException(404, "no tiles match that selection")
        thr = S.thresholds(body.alpha)
        budget = body.scaled_fraction * len(sel) * OPTICAL_B
        sel = sel.reset_index(drop=True)
        out, skipped = {}, {}
        for pol in body.policies:
            if pol not in POLICIES:
                raise HTTPException(422, f"unknown policy {pol}; choose from {POLICIES}")
            try:
                log = scheduler_sim.run(sel, pol, budget, thresholds=thr)
            except NotAvailable as e:
                skipped[SYSTEM_NAMES[pol]] = str(e)
                continue
            curve = metrics.recall_curve(log, sel)
            keep = np.unique(np.linspace(0, len(curve) - 1, min(len(curve), 300)).astype(int))
            # the queue is long, so show the first packets of EACH layer (alerts, then previews, then tiles)
            sample = pd.concat([log[log["layer"] == layer].head(15) for layer in ("L0", "L1", "L2")])
            first = sample.merge(sel[["chip", "row", "col", "cloud_frac", "critical"]], on=["chip", "row", "col"], how="left")
            first["tile_id"] = first["chip"] + "_" + first["row"].astype(str) + "_" + first["col"].astype(str)
            summ = metrics.summarise(SYSTEM_NAMES[pol], log, sel, budget)
            out[SYSTEM_NAMES[pol]] = {
                "summary": summ, "total_packets": len(log),
                "curve": [{"cum_MB": float(c) / 1e6, "recall": float(r)} for c, r in
                          zip(curve["cum_bytes"].iloc[keep], curve["recall"].iloc[keep])],
                "queue_preview": first[["order", "layer", "tile_id", "bytes", "cum_bytes", "pass_index", "critical"]].to_dict("records"),
            }
        return _clean({"synthetic": S.synthetic, "selection": {"split": body.split, "event": body.event, "chip": body.chip,
                                                               "tiles": len(sel), "critical": int(sel["critical"].sum())},
                       "alpha": body.alpha, "thresholds": thr, "scaled_budget_MB": budget / 1e6,
                       "scaled_note": "scaled pass = 20% of the selection's all-optical volume, not the real 19.2 MB pass",
                       "systems": out, "skipped": skipped})

    @app.get("/api/demo-case")
    def demo_case():
        path = root / "results" / "demo_case.json"
        if not path.exists():
            return {"available": False, "message": "results/demo_case.json is produced from real scores; not yet available"}
        return {"available": True, **json.loads(path.read_text())}

    web = root / "web"
    if web.is_dir():
        app.mount("/", StaticFiles(directory=str(web), html=True), name="web")
    app.state.S = S
    return app


def _label_png(root: Path, row: pd.Series, scale: int) -> Response:
    """Colour-coded label tile from the raw chips: flood water, permanent water, not water, no data."""
    import tifffile
    base = root / "data" / "raw" / "data" / "flood_events" / "HandLabeled"
    lab, jrc = base / "LabelHand" / f"{row['chip']}_LabelHand.tif", base / "JRCWaterHand" / f"{row['chip']}_JRCWaterHand.tif"
    if not (lab.exists() and jrc.exists()):
        raise HTTPException(404, "raw label files not available: python scripts/download_flood.py")
    r, c = int(row["row"]) * 128, int(row["col"]) * 128
    label = tifffile.imread(lab)[r:r + 128, c:c + 128]
    perm = tifffile.imread(jrc)[r:r + 128, c:c + 128] == 1
    img = np.full((128, 128, 3), (42, 42, 42), dtype=np.uint8)           # not water
    img[(label == 1) & ~perm] = (255, 244, 79)                           # flood water (counts toward critical)
    img[(label == 1) & perm] = (57, 135, 229)                            # permanent water (subtracted)
    img[label == -1] = (0, 0, 0)                                         # no data
    return _png(img, scale)


app = create_app()
