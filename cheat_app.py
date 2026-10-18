"""
Demo checker: upload a CS2 .dem file in your browser and see, for every player
who got kills, how cheater-like their aim looked to each of the three models.

Setup:
  1. Run three_simple_models.ipynb once. Step 7 saves the trained models to models/.
  2. Keep these files in the same folder:
       cheat_app.py, cs2_kill_aim_telemetry.py, build_dataset.py, models/
Run:
  python cheat_app.py
  Then open http://127.0.0.1:8000 (it opens automatically).

Nothing is uploaded to the internet: the page and the analysis both run on
your own computer.
"""

import json
import os
import tempfile
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd

import cs2_kill_aim_telemetry as telemetry
from build_dataset import window_features

HOST, PORT = "127.0.0.1", 8000
MODEL_DIR = Path("models")
MODEL_FILES = {
    "Linear regression": "linear.npz",
    "Logistic regression": "logistic.npz",
    "Neural network": "network.npz",
}
TICK_RATE = 64
MIN_KILLS = 5                      # fewer kills than this = too little to judge
TOP_KILLS = 5                      # most suspicious kills listed per player
REWIND_SECONDS = 5                 # "jump to" commands start this long before the kill
MAX_UPLOAD_BYTES = 1_500_000_000   # 1.5 GB

# Must match Step 1 of three_simple_models.ipynb (features that get a log transform)
LOG = {"head_ang_err", "head_miss_units", "chest_ang_err", "spatial_dist_to_target",
       "ms_on_target", "approach_peak_ang_vel", "acquired_from_deg", "mean_ang_vel",
       "max_ang_vel", "max_abs_jerk", "head_err_last_half_s", "ang_vel_std_last_half_s",
       "head_err_early"}
YES_NO_COLUMNS = ["headshot", "penetrated", "thrusmoke", "attackerblind", "noscope",
                  "on_head_at_kill", "on_target_whole_window"]


# ---------------------------------------------------------------------------
# Models (same maths as Steps 3, 7 and 8 of three_simple_models.ipynb)
# ---------------------------------------------------------------------------
WEIGHT_KEYS = {"linear": ["w", "b"], "logistic": ["w", "b"], "network": ["W1", "b1", "w2", "b2"]}


def load_model(path):
    f = np.load(path)
    model_type = str(f["type"])
    model = {"type": model_type, "threshold": float(f["threshold"]),
             **{k: f[k] for k in WEIGHT_KEYS[model_type]}}
    prep = {"features": [str(x) for x in f["features"]],
            "median": f["median"], "mean": f["mean"], "std": f["std"]}
    return model, prep


def to_matrix(part, features):
    X = part[features].astype(float).to_numpy()
    for j, f in enumerate(features):
        if f in LOG:
            X[:, j] = np.log1p(np.clip(X[:, j], 0, None))
    return X


def apply_prep(part, prep):
    M = to_matrix(part, prep["features"])
    return (np.where(np.isnan(M), prep["median"], M) - prep["mean"]) / prep["std"]


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def relu(z):
    return np.maximum(0, z)


def predict(model, X):
    if model["type"] == "linear":
        return X @ model["w"] + model["b"]
    if model["type"] == "logistic":
        return sigmoid(X @ model["w"] + model["b"])
    return sigmoid(relu(X @ model["W1"] + model["b1"]) @ model["w2"] + model["b2"])


def load_all_models():
    missing = [f for f in MODEL_FILES.values() if not (MODEL_DIR / f).exists()]
    if missing:
        raise SystemExit(f"Missing model files in {MODEL_DIR}/: {', '.join(missing)}. "
                         "Run three_simple_models.ipynb first (Step 7 saves them).")
    return {name: load_model(MODEL_DIR / f) for name, f in MODEL_FILES.items()}


MODELS = None  # loaded at start-up


# ---------------------------------------------------------------------------
# Analysis of one demo
# ---------------------------------------------------------------------------
def verdict(n_kills, n_flagging, n_models):
    if n_kills < MIN_KILLS:
        return "few", "Too few kills to judge"
    if n_flagging == n_models:
        return "flag", "Likely cheating"
    if n_flagging == 0:
        return "clean", "No signs found"
    return "review", "Needs a human look"


def player_names(dem_path, kills):
    """steamid -> in-game name, including victims who never got a kill themselves."""
    names = dict(zip(kills["steamid"], kills["name"].astype(str)))
    try:
        events = telemetry.DemoParser(str(dem_path)).parse_event("player_death")
        for sid_col, name_col in [("user_steamid", "user_name"), ("attacker_steamid", "attacker_name")]:
            if sid_col in events.columns and name_col in events.columns:
                for sid, name in zip(telemetry._norm_sid(events[sid_col]), events[name_col]):
                    if pd.notna(name):
                        names.setdefault(sid, str(name))
    except Exception:
        pass  # names are a nice-to-have; fall back to "unknown player"
    return names


def _number(value):
    return None if pd.isna(value) else float(value)


def top_kills(player_kills, names, n=TOP_KILLS):
    """The player's n most suspicious kills, with the details needed to find them in the demo."""
    top = player_kills.sort_values(["suspicion", "head_ang_err"], ascending=[False, True]).head(n)
    out = []
    for _, k in top.iterrows():
        tick = int(k["kill_tick"])
        out.append({
            "tick": tick,
            "seconds": tick / TICK_RATE,
            "goto_tick": max(0, tick - REWIND_SECONDS * TICK_RATE),
            "victim": names.get(str(k["target_steamid"]), "unknown player"),
            "weapon": str(k.get("weapon", "")).replace("weapon_", "") if pd.notna(k.get("weapon")) else "",
            "headshot": bool(k.get("headshot", 0)),
            "suspicion": float(k["suspicion"]),
            "head_err_deg": _number(k.get("head_ang_err")),
            "ms_on_target": _number(k.get("ms_on_target")),
            "flick_deg_per_s": _number(k.get("approach_peak_ang_vel")),
        })
    return out


def analyze_demo(dem_path):
    with tempfile.TemporaryDirectory() as tmp:
        ticks_csv, summary_csv = Path(tmp) / "ticks.csv", Path(tmp) / "summary.csv"
        telemetry.export_kill_prewindow_telemetry(
            demo_path=str(dem_path), cheater_ids=[],
            output_csv_path=str(ticks_csv), summary_csv_path=str(summary_csv),
            tick_rate=TICK_RATE,
        )
        if not summary_csv.exists():
            raise ValueError("No scoreable kills were found in this demo. Grenade, knife "
                             "and fire kills are skipped, so a demo needs gun kills.")
        ticks = pd.read_csv(ticks_csv, dtype={"steamid": str, "target_steamid": str})
        summary = pd.read_csv(summary_csv, dtype={"steamid": str, "target_steamid": str})

    kills = summary.merge(window_features(ticks), on="kill_id", how="left")
    for col in YES_NO_COLUMNS:
        if col in kills.columns:
            kills[col] = kills[col].astype(str).str.lower().isin(["true", "1"]).astype(int)

    models_out = []
    for name, (model, prep) in MODELS.items():
        score = predict(model, apply_prep(kills, prep))
        if model["type"] == "linear":            # a score, not a probability: keep it in 0-1
            score = np.clip(score, 0, 1)
        kills[name] = score
        models_out.append({"name": name, "threshold": float(np.clip(model["threshold"], 0, 1))})

    # A kill's suspicion = the average of the three models' scores for it
    kills["suspicion"] = kills[[m["name"] for m in models_out]].mean(axis=1)
    names = player_names(dem_path, kills)

    players = []
    for steamid, g in kills.groupby("steamid"):
        scores = {m["name"]: float(g[m["name"]].mean()) for m in models_out}
        n_flagging = sum(scores[m["name"]] >= m["threshold"] for m in models_out)
        level, text = verdict(len(g), n_flagging, len(models_out))
        players.append({"name": str(g["name"].iloc[0]), "steamid": steamid, "kills": len(g),
                        "scores": scores, "models_flagging": int(n_flagging),
                        "verdict": level, "verdict_text": text,
                        "top_kills": top_kills(g, names)})

    order = {"flag": 0, "review": 1, "clean": 2, "few": 3}
    players.sort(key=lambda p: (order[p["verdict"]], -np.mean(list(p["scores"].values()))))
    return {"models": models_out, "players": players, "kills": int(len(kills))}


# ---------------------------------------------------------------------------
# Web server
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def _send(self, status, body, content_type):
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        else:
            self._send(404, "Not found", "text/plain")

    def do_POST(self):
        if self.path != "/analyze":
            return self._send(404, "Not found", "text/plain")
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return self._send(400, json.dumps({"error": "The file is empty."}), "application/json")
        if length > MAX_UPLOAD_BYTES:
            return self._send(413, json.dumps({"error": "The file is larger than 1.5 GB."}),
                              "application/json")
        with tempfile.NamedTemporaryFile(suffix=".dem", delete=False) as f:
            remaining = length
            while remaining > 0:
                chunk = self.rfile.read(min(1 << 20, remaining))
                if not chunk:
                    break
                f.write(chunk)
                remaining -= len(chunk)
            dem_path = Path(f.name)
        try:
            result = analyze_demo(dem_path)
            self._send(200, json.dumps(result), "application/json")
        except Exception as e:
            print(f"Analysis failed: {e!r}")
            self._send(400, json.dumps({"error": str(e) or e.__class__.__name__}), "application/json")
        finally:
            dem_path.unlink(missing_ok=True)

    def log_message(self, fmt, *args):
        pass  # keep the console quiet; analysis progress is printed instead


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Demo Checker</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Condensed:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {
  --paper: #EEF1F4; --sheet: #FFFFFF; --ink: #1C2633; --slate: #5B6878; --rule: #CDD4DC;
  --flag: #B4372F; --review: #C98A12; --clean: #2F7D6B; --track: #E2E7EC;
  --cond: "IBM Plex Sans Condensed", "Arial Narrow", sans-serif;
  --body: "IBM Plex Sans", system-ui, sans-serif;
}
@media (prefers-color-scheme: dark) {
  :root { --paper: #141B24; --sheet: #1C2530; --ink: #E6EAEF; --slate: #9AA6B4; --rule: #2E3946;
          --flag: #E06A60; --review: #E2A93A; --clean: #4FB39B; --track: #2A3440; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--paper); color: var(--ink); font: 16px/1.55 var(--body); }
main { max-width: 980px; margin: 0 auto; padding: 48px 16px 64px; }
h1 { font: 700 2.6rem/1.05 var(--cond); margin: 0 0 10px; letter-spacing: -0.01em; }
.lede { color: var(--slate); max-width: 62ch; margin: 0 0 28px; }

.drop { display: block; border: 2px dashed var(--rule); border-radius: 10px; background: var(--sheet);
        padding: 34px 24px; cursor: pointer; transition: border-color .15s, background .15s; }
.drop:hover, .drop.over { border-color: var(--ink); }
.drop:focus-within { outline: 3px solid var(--review); outline-offset: 3px; }
.drop strong { display: block; font: 600 1.35rem/1.2 var(--cond); }
.drop span { color: var(--slate); }
.drop input { position: absolute; opacity: 0; width: 1px; height: 1px; }

.status { margin: 18px 0 0; min-height: 1.5em; color: var(--slate); }
.status.error { color: var(--flag); }
.spinner { display: inline-block; width: 12px; height: 12px; margin-right: 8px; border: 2px solid var(--rule);
           border-top-color: var(--ink); border-radius: 50%; animation: spin .8s linear infinite; vertical-align: -1px; }
@keyframes spin { to { transform: rotate(360deg); } }

#results { margin-top: 40px; }
#results[hidden] { display: none; }
h2 { font: 600 1.5rem/1.2 var(--cond); margin: 0 0 4px; }
.sub { color: var(--slate); margin: 0 0 18px; }
.board { overflow-x: auto; background: var(--sheet); border: 1px solid var(--rule); border-radius: 10px; }
table { width: 100%; min-width: 720px; border-collapse: collapse; }
th { font: 600 0.95rem/1.2 var(--cond); color: var(--slate); text-align: left; padding: 14px 16px;
     border-bottom: 1px solid var(--rule); white-space: nowrap; }
td { padding: 14px 16px; border-bottom: 1px solid var(--rule); vertical-align: middle; }
tr:last-child td { border-bottom: 0; }
.player { font: 600 1.1rem/1.2 var(--cond); }
.kills { font: 500 1.1rem var(--cond); color: var(--slate); text-align: right; width: 64px; }

.score { width: 170px; }
.pct { font: 600 1.15rem/1 var(--cond); font-variant-numeric: tabular-nums; }
.bar { position: relative; height: 8px; margin-top: 7px; background: var(--track); border-radius: 4px; }
.fill { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 4px; background: var(--slate); }
.fill.over { background: var(--flag); }
.cut { position: absolute; top: -4px; bottom: -4px; width: 2px; background: var(--ink); }

.verdict { font: 600 1rem/1.2 var(--cond); white-space: nowrap; padding-left: 12px; border-left: 4px solid var(--rule); }
.verdict small { display: block; font: 400 .85rem/1.3 var(--body); color: var(--slate); }
.v-flag { border-color: var(--flag); color: var(--flag); }
.v-review { border-color: var(--review); color: var(--review); }
.v-clean { border-color: var(--clean); color: var(--clean); }
.v-few { color: var(--slate); }

.key { display: flex; flex-wrap: wrap; gap: 8px 24px; margin: 14px 0 0; color: var(--slate); font-size: .9rem; }
.key i { display: inline-block; vertical-align: middle; margin-right: 6px; }
.key .k-cut { width: 2px; height: 14px; background: var(--ink); }
.key .k-over { width: 18px; height: 8px; border-radius: 4px; background: var(--flag); }
.notes { margin-top: 28px; color: var(--slate); font-size: .92rem; max-width: 70ch; }
.notes p { margin: 0 0 8px; }
.expand { all: unset; cursor: pointer; display: inline-flex; align-items: center; gap: 8px;
          font: 600 1.1rem/1.2 var(--cond); color: var(--ink); border-radius: 4px; }
.expand:focus-visible { outline: 3px solid var(--review); outline-offset: 3px; }
.expand::before { content: ""; width: 0; height: 0; border-left: 6px solid var(--slate);
                  border-top: 5px solid transparent; border-bottom: 5px solid transparent; transition: transform .15s; }
.expand[aria-expanded="true"]::before { transform: rotate(90deg); }
.expand:hover { text-decoration: underline; text-underline-offset: 3px; }

tr.detail > td { background: var(--paper); padding: 4px 16px 18px 40px; }
tr.detail[hidden] { display: none; }
.detail-inner { position: sticky; left: 0; max-width: calc(100vw - 64px); }
.detail h3 { font: 600 1rem/1.3 var(--cond); color: var(--slate); margin: 12px 0 6px; }
.kills-list { list-style: none; margin: 0; padding: 0; }
.kill { display: grid; grid-template-columns: 76px minmax(0, 1fr) 72px auto; gap: 4px 18px; align-items: center;
        padding: 12px 0; border-top: 1px solid var(--rule); }
.kill:first-child { border-top: 0; }
.clock { display: block; font: 600 1.25rem/1 var(--cond); font-variant-numeric: tabular-nums; }
.tick { display: block; color: var(--slate); font-size: .8rem; margin-top: 3px; }
.what p { margin: 0; }
.facts { display: flex; flex-wrap: wrap; gap: 2px 16px; color: var(--slate); font-size: .88rem; margin-top: 2px !important; }
.susp { font: 600 1.15rem/1 var(--cond); text-align: right; font-variant-numeric: tabular-nums; }
.susp small { display: block; font: 400 .75rem var(--body); color: var(--slate); margin-top: 3px; }
.jump { display: flex; align-items: center; gap: 8px; }
.jump code { font: 500 .85rem/1 var(--cond); background: var(--sheet); border: 1px solid var(--rule);
             border-radius: 4px; padding: 6px 8px; white-space: nowrap; }
.copy { font: 600 .85rem/1 var(--cond); color: var(--ink); background: var(--sheet); border: 1px solid var(--ink);
        border-radius: 4px; padding: 6px 10px; cursor: pointer; min-width: 64px; }
.copy:focus-visible { outline: 3px solid var(--review); outline-offset: 2px; }
@media (max-width: 640px) {
  tr.detail > td { padding-left: 16px; }
  .kill { grid-template-columns: 64px minmax(0, 1fr) 56px; }
  .jump { grid-column: 1 / -1; }
}
@media (prefers-reduced-motion: reduce) { .spinner { animation: none; } .drop, .expand::before { transition: none; } }
</style>
</head>
<body>
<main>
  <h1>Demo checker</h1>
  <p class="lede">Upload a CS2 match demo. Every gun kill is measured — how far the crosshair was from the
  victim's head, how fast it snapped there, how long it stayed — and three models trained on labelled
  matches score each player's aim.</p>

  <label class="drop" id="drop">
    <input type="file" id="file" accept=".dem">
    <strong>Choose a .dem file or drop it here</strong>
    <span>A full match takes about a minute to analyse. The file stays on this computer.</span>
  </label>
  <p class="status" id="status" role="status" aria-live="polite"></p>

  <section id="results" hidden>
    <h2 id="title"></h2>
    <p class="sub" id="summary"></p>
    <div class="board">
      <table>
        <thead><tr id="head"></tr></thead>
        <tbody id="rows"></tbody>
      </table>
    </div>
    <div class="key">
      <span><i class="k-cut"></i>Model's cut-off for "cheater-like"</span>
      <span><i class="k-over"></i>Average past the cut-off</span>
    </div>
    <div class="notes">
      <p>Each percentage is the model's average score across that player's kills, not a proven probability
      of cheating. The models were trained to catch cheaters, so scores run high; compare players with the
      cut-off marks rather than reading the numbers on their own.</p>
      <p>"Likely cheating" means all three models put the player past their cut-off. When the models
      disagree, a person should watch the demo. Players without a gun kill aren't listed.</p>
      <p>Kill times count from the start of the demo file, warmup included. To watch a kill, paste its
      command into the CS2 console while the demo is playing; it jumps to five seconds before the kill.</p>
    </div>
  </section>
</main>
<script>
const drop = document.getElementById("drop");
const input = document.getElementById("file");
const statusEl = document.getElementById("status");

function setStatus(html, isError) {
  statusEl.innerHTML = html;
  statusEl.classList.toggle("error", !!isError);
}
function esc(s) {
  return String(s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
}

async function analyse(file) {
  if (!file) return;
  if (!file.name.toLowerCase().endsWith(".dem")) {
    setStatus("That isn't a .dem file. Choose a CS2 demo file.", true);
    return;
  }
  document.getElementById("results").hidden = true;
  setStatus(`<span class="spinner"></span>Analysing ${esc(file.name)}…`);
  try {
    const res = await fetch("/analyze", { method: "POST", body: file });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "The demo couldn't be analysed.");
    render(file.name, data);
    setStatus("");
  } catch (err) {
    setStatus(esc(err.message), true);
  } finally {
    input.value = "";
  }
}

function render(fileName, data) {
  document.getElementById("title").textContent = fileName;
  const flagged = data.players.filter(p => p.verdict === "flag").length;
  const review = data.players.filter(p => p.verdict === "review").length;
  document.getElementById("summary").textContent =
    `${data.kills} gun kills by ${data.players.length} players. ` +
    `${flagged} likely cheating, ${review} need a human look. ` +
    `Open a player to see their most suspicious kills.`;

  document.getElementById("head").innerHTML =
    `<th>Player</th><th>Verdict</th><th style="text-align:right">Kills</th>` +
    data.models.map(m => `<th>${esc(m.name)}</th>`).join("");

  const columns = 3 + data.models.length;
  document.getElementById("rows").innerHTML = data.players.map((p, i) => {
    const cells = data.models.map(m => {
      const s = p.scores[m.name];
      const over = s >= m.threshold;
      return `<td class="score">
        <div class="pct">${Math.round(s * 100)}%</div>
        <div class="bar" aria-hidden="true">
          <div class="fill${over ? " over" : ""}" style="width:${(s * 100).toFixed(1)}%"></div>
          <div class="cut" style="left:calc(${(m.threshold * 100).toFixed(1)}% - 1px)"></div>
        </div></td>`;
    }).join("");
    const detail = p.verdict === "few" ? `${p.kills} kill${p.kills === 1 ? "" : "s"}`
                 : `${p.models_flagging} of ${data.models.length} models past cut-off`;
    return `<tr><td class="player">
        <button class="expand" aria-expanded="false" aria-controls="detail-${i}">${esc(p.name)}</button></td>
      <td><div class="verdict v-${p.verdict}">${esc(p.verdict_text)}<small>${detail}</small></div></td>
      <td class="kills">${p.kills}</td>${cells}</tr>
      <tr class="detail" id="detail-${i}" hidden><td colspan="${columns}"><div class="detail-inner">${killList(p)}</div></td></tr>`;
  }).join("");
  document.getElementById("results").hidden = false;
}

function clock(seconds) {
  const m = Math.floor(seconds / 60), s = Math.floor(seconds % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

function killList(p) {
  const n = p.top_kills.length;
  const heading = n === 1 ? "Their only kill" : `${n === p.kills ? "All" : "Top"} ${n} kills, most suspicious first`;
  const items = p.top_kills.map(k => {
    const facts = [];
    if (k.head_err_deg !== null) facts.push(`${k.head_err_deg.toFixed(k.head_err_deg < 1 ? 2 : 1)}° from head centre`);
    if (k.ms_on_target !== null) facts.push(`fired ${Math.round(k.ms_on_target)} ms after reaching the head`);
    else facts.push("crosshair wasn't on the head");
    if (k.flick_deg_per_s !== null) facts.push(`${Math.round(k.flick_deg_per_s).toLocaleString()}°/s snap`);
    const how = [k.weapon && `with ${esc(k.weapon)}`, k.headshot && "headshot"].filter(Boolean).join(", ");
    const cmd = `demo_gototick ${k.goto_tick}`;
    return `<li class="kill">
      <div><span class="clock">${clock(k.seconds)}</span><span class="tick">tick ${k.tick.toLocaleString()}</span></div>
      <div class="what"><p>Killed <strong>${esc(k.victim)}</strong>${how ? " " + how : ""}</p>
        <p class="facts">${facts.map(f => `<span>${f}</span>`).join("")}</p></div>
      <div class="susp">${Math.round(k.suspicion * 100)}%<small>suspicion</small></div>
      <div class="jump"><code>${cmd}</code><button class="copy" data-cmd="${cmd}">Copy</button></div>
    </li>`;
  }).join("");
  return `<h3>${heading}</h3><ol class="kills-list">${items}</ol>`;
}

document.getElementById("rows").addEventListener("click", async e => {
  const toggle = e.target.closest(".expand");
  if (toggle) {
    const open = toggle.getAttribute("aria-expanded") === "true";
    toggle.setAttribute("aria-expanded", String(!open));
    document.getElementById(toggle.getAttribute("aria-controls")).hidden = open;
    return;
  }
  const copy = e.target.closest(".copy");
  if (copy) {
    try {
      await navigator.clipboard.writeText(copy.dataset.cmd);
      copy.textContent = "Copied";
    } catch {
      copy.textContent = "Select it";
    }
    setTimeout(() => { copy.textContent = "Copy"; }, 1500);
  }
});

input.addEventListener("change", () => analyse(input.files[0]));
["dragenter", "dragover"].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", e => analyse(e.dataTransfer.files[0]));
</script>
</body>
</html>
"""


def main():
    global MODELS
    os.chdir(Path(__file__).resolve().parent)  # relative paths start from this script's folder
    MODELS = load_all_models()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    url = f"http://{HOST}:{PORT}"
    print(f"Demo checker running at {url}  (press Ctrl+C to stop)")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()