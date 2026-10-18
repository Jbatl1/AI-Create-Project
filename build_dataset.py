"""
Step 1 of the pipeline: turn a folder of CS2 demos into ONE table for training.

    demo files  ->  cs2_kill_aim_telemetry.py (per tick, per kill)
                ->  this script (one row per kill = one training example)
                ->  data/kill_features.csv  ->  cheat_detector.ipynb

Each row is a kill. The label (y) is whether the killer is a known cheater.
List every demo you have below, with the SteamIDs (or names) of the known
cheaters in that demo. Demos with NO cheaters are just as important: the model
needs plenty of examples of normal, legit aim to compare against.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from cs2_kill_aim_telemetry import ON_HEAD_UNITS, export_kill_prewindow_telemetry

# ---------------------------------------------------------------------------
# 1. List your demos here. "cheaters" = SteamID64s or in-game names.
#    Use [] for demos where you know nobody cheated.
# ---------------------------------------------------------------------------
DEMOS = [
    {
        "path": "AI-Create-Project/clips/mouz-nxt-vs-ursa-m1-dust2.dem",
        "cheaters": ["76561198843946848", "karnez"],
    },
    # {"path": "AI-Create-Project/clips/another_match.dem", "cheaters": []},
]

OUT_DIR = Path("data")
TICK_RATE = 64
REBUILD = False  # True = re-parse demos even if their CSVs already exist

LAST_HALF_SECOND = TICK_RATE // 2  # 32 ticks
EARLY_FROM = TICK_RATE  # "early" = more than 1 s before the kill


def parse_demo(demo_path: str, cheaters: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the telemetry script on one demo (cached as CSVs)."""
    stem = Path(demo_path).stem
    ticks_csv = OUT_DIR / f"{stem}_ticks.csv"
    summary_csv = OUT_DIR / f"{stem}_summary.csv"
    if REBUILD or not (ticks_csv.exists() and summary_csv.exists()):
        export_kill_prewindow_telemetry(
            demo_path=demo_path,
            cheater_ids=cheaters,
            output_csv_path=str(ticks_csv),
            summary_csv_path=str(summary_csv),
            tick_rate=TICK_RATE,
        )
    ticks = pd.read_csv(ticks_csv, dtype={"steamid": str, "target_steamid": str})
    summary = pd.read_csv(summary_csv, dtype={"steamid": str, "target_steamid": str})
    return ticks, summary


def window_features(ticks: pd.DataFrame) -> pd.DataFrame:
    """Summarise the 3 s before each kill into a few numbers per kill."""
    t = ticks.copy()
    t["on_head"] = (t["head_miss_units"] <= ON_HEAD_UNITS).astype(float)
    t["abs_jerk"] = t["angular_jerk"].abs()
    last = t[t["ticks_before_kill"] <= LAST_HALF_SECOND]
    early = t[t["ticks_before_kill"] > EARLY_FROM]

    g = t.groupby("kill_id")
    feats = pd.DataFrame({
        # How fast and how jerky the crosshair moved over the whole window
        "mean_ang_vel": g["angular_velocity"].mean(),
        "max_ang_vel": g["angular_velocity"].max(),
        "max_abs_jerk": g["abs_jerk"].max(),
        # Share of the 3 s the crosshair spent on the victim's head
        "frac_window_on_head": g["on_head"].mean(),
    })
    # Aim in the last half second (the fight itself)
    gl = last.groupby("kill_id")
    feats["head_err_last_half_s"] = gl["head_ang_err"].mean()
    feats["ang_vel_std_last_half_s"] = gl["angular_velocity"].std()
    # Aim 1-3 s before the kill. Low error here means the crosshair was
    # already following the victim's head well before the fight; a possible
    # wallhack sign (or good pre-aim, so it's evidence, not proof).
    feats["head_err_early"] = early.groupby("kill_id")["head_ang_err"].mean()
    return feats.reset_index()


def build() -> pd.DataFrame:
    OUT_DIR.mkdir(exist_ok=True)
    tables = []
    for d in DEMOS:
        if not Path(d["path"]).exists():
            print(f"SKIP (file not found): {d['path']}")
            continue
        ticks, summary = parse_demo(d["path"], d["cheaters"])
        feats = summary.merge(window_features(ticks), on="kill_id", how="left")
        feats.insert(0, "demo", Path(d["path"]).stem)
        tables.append(feats)

    if not tables:
        raise SystemExit("No demos were parsed. Check the paths in DEMOS.")

    df = pd.concat(tables, ignore_index=True)

    # Pseudonymise players for posters and slides (keep steamid for grouping).
    ids = sorted(df["steamid"].unique())
    alias = {sid: f"P{i + 1:03d}" for i, sid in enumerate(ids)}
    df.insert(1, "player_alias", df["steamid"].map(alias))

    for col in ["headshot", "penetrated", "thrusmoke", "attackerblind", "noscope",
                "on_head_at_kill", "on_target_whole_window"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().isin(["true", "1"]).astype(int)

    out = OUT_DIR / "kill_features.csv"
    df.to_csv(out, index=False)

    n_cheat_players = df.loc[df["is_cheater"] == 1, "steamid"].nunique()
    print(f"\nSaved {out}: {len(df)} kills, {df['steamid'].nunique()} players, "
          f"{n_cheat_players} labelled cheater(s), {df['is_cheater'].sum()} cheater kills.")
    if n_cheat_players < 3:
        print("WARNING: fewer than 3 cheaters. The model can't be tested fairly on "
              "players it never saw. Add more demos with known cheaters.")
    return df


if __name__ == "__main__":
    build()
