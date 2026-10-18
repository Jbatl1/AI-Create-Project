import numpy as np
import pandas as pd
from demoparser2 import DemoParser

# ---------------------------------------------------------------------------
# Geometry constants: Hammer units above the player's origin, which is at the feet.
# Eye heights are CS2's view offsets. Hitbox centres are APPROXIMATIONS, because
# demos don't store bone/animation positions. Tune them if you calibrate in-game.
# ---------------------------------------------------------------------------
EYE_STAND, EYE_CROUCH = 64.06, 46.05
HEAD_STAND, HEAD_CROUCH = 64.0, 46.0
CHEST_STAND, CHEST_CROUCH = 50.0, 34.0  # upper chest

# Kills where the crosshair angle isn't meaningful (nades, fire, bomb, knife)
NON_HITSCAN = {
    "hegrenade", "inferno", "molotov", "incgrenade", "flashbang",
    "smokegrenade", "decoy", "planted_c4", "world",
}

# Max tick gap before a player's motion series is treated as broken
# (death, respawn, or a new round). Deltas are never taken across a break.
MAX_TICK_GAP = 8

# "On the head" = aim line passes within this many units of the head centre
# (roughly the head hitbox radius). Distance-independent, unlike a degree cutoff.
ON_HEAD_UNITS = 4.5
# How many ticks before the crosshair arrived on the head to look for the
# flick that brought it there (8 ticks = 125 ms at 64 tick).
APPROACH_TICKS = 8


def _lerp(stand, crouch, duck):
    return stand + (crouch - stand) * duck


def _norm_sid(s: pd.Series) -> pd.Series:
    # SteamID64s must stay integers: going through float loses precision
    return pd.to_numeric(s, errors="coerce").astype("Int64").astype(str)


def _forward(pitch_deg, yaw_deg):
    """Source-engine view angles -> unit forward vector.
    Yaw 0 = +X, yaw 90 = +Y, positive pitch = looking DOWN."""
    p = np.radians(np.asarray(pitch_deg, dtype=float))
    y = np.radians(np.asarray(yaw_deg, dtype=float))
    cp = np.cos(p)
    return np.column_stack([cp * np.cos(y), cp * np.sin(y), -np.sin(p)])


def _angle_between(u, v):
    """True angle (degrees) between row vectors. atan2(|u x v|, u.v) stays
    precise at tiny angles, where arccos(dot) does not, and tiny angles
    are exactly the range that matters for aim analysis."""
    cross = np.linalg.norm(np.cross(u, v), axis=1)
    dot = np.einsum("ij,ij->i", u, v)
    return np.degrees(np.arctan2(cross, dot))


def _aim_error(eye, fwd, target):
    """Returns distance, true angular error, linear miss distance and the
    ideal pitch/yaw for aiming from `eye` along `fwd` at `target`."""
    d = target - eye
    dist = np.linalg.norm(d, axis=1)
    ang = _angle_between(fwd, d)
    # Perpendicular distance from the aim ray to the target point, in units:
    # "how many units from the centre of the head the crosshair passed".
    along = np.einsum("ij,ij->i", fwd, d)
    miss = np.linalg.norm(np.cross(fwd, d), axis=1)
    miss = np.where(along < 0, dist, miss)  # target behind the player
    ideal_yaw = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
    ideal_pitch = -np.degrees(np.arctan2(d[:, 2], np.hypot(d[:, 0], d[:, 1])))
    return dist, ang, miss, ideal_pitch, ideal_yaw


def _add_motion_features(df: pd.DataFrame, tick_rate: int) -> pd.DataFrame:
    df = df.sort_values(["steamid", "tick"]).reset_index(drop=True)
    dt = df["tick"].diff()
    new_seg = (df["steamid"] != df["steamid"].shift()) | (dt > MAX_TICK_GAP) | dt.isna()
    df["segment"] = new_seg.cumsum()

    fwd = _forward(df["pitch"], df["yaw"])
    step = _angle_between(np.roll(fwd, 1, axis=0), fwd)  # deg moved since last row
    step[new_seg.to_numpy()] = np.nan
    dt_s = (dt / tick_rate).where(~new_seg)

    df["angular_velocity"] = step / dt_s  # deg/s
    df["angular_accel"] = df.groupby("segment")["angular_velocity"].diff() / dt_s  # deg/s^2
    df["angular_jerk"] = df.groupby("segment")["angular_accel"].diff() / dt_s  # deg/s^3
    df["delta_pitch"] = df.groupby("segment")["pitch"].diff()
    df["delta_yaw"] = (df.groupby("segment")["yaw"].diff() + 180) % 360 - 180
    return df


def _acquisition_stats(ticks_before, head_miss, head_ang, ang_vel, tick_rate):
    """How the crosshair arrived on the head before one kill.

    ticks_on_target: ticks between the crosshair arriving on the head and the
        kill, counting only the final unbroken stretch. 0 = it arrived on the
        kill tick itself (fired the instant it got there).
    approach_peak_ang_vel: fastest crosshair movement (deg/s) in the
        APPROACH_TICKS leading into the arrival tick, including the arrival
        move itself. High = flicked onto the head; low = target walked into a
        held crosshair.
    acquired_from_deg: head error on the last tick before arrival, i.e. how
        far away the crosshair was just before it got there.
    """
    out = {
        "on_head_at_kill": False,
        "ticks_on_target": np.nan,
        "ms_on_target": np.nan,
        "approach_peak_ang_vel": np.nan,
        "acquired_from_deg": np.nan,
        "on_target_whole_window": False,
    }
    order = np.argsort(ticks_before)  # kill tick first, then backwards
    tb = np.asarray(ticks_before)[order]
    miss = np.asarray(head_miss, dtype=float)[order]
    ang = np.asarray(head_ang, dtype=float)[order]
    vel = np.asarray(ang_vel, dtype=float)[order]

    off = np.isnan(miss) | (miss > ON_HEAD_UNITS)
    if len(tb) == 0 or tb[0] != 0 or off[0]:
        return out  # not on the head when the kill happened
    out["on_head_at_kill"] = True

    if off.any():
        first_off = int(np.argmax(off))
        arrival = first_off - 1  # earliest tick of the final on-head stretch
        out["acquired_from_deg"] = ang[first_off]
    else:
        arrival = len(tb) - 1  # on the head for the entire window
        out["on_target_whole_window"] = True

    tb_arrival = tb[arrival]
    out["ticks_on_target"] = int(tb_arrival)
    out["ms_on_target"] = tb_arrival / tick_rate * 1000.0

    # Velocity at tick t is the move from t-1 to t, so the arrival tick's own
    # velocity is the move that put the crosshair on the head.
    near = (tb >= tb_arrival) & (tb <= tb_arrival + APPROACH_TICKS)
    if np.any(near & ~np.isnan(vel)):
        out["approach_peak_ang_vel"] = float(np.nanmax(vel[near]))
    return out


def export_kill_prewindow_telemetry(
    demo_path: str,
    cheater_ids: list[str] | None = None,
    output_csv_path: str = "cs2_kill_prewindows_metrics.csv",
    summary_csv_path: str = "cs2_kill_aim_summary.csv",
    window_seconds: float = 3.0,
    tick_rate: int = 64,
    hitscan_only: bool = True,
):
    cheater_ids = {str(c).strip().lower() for c in (cheater_ids or [])}

    print(f"Parsing demo: {demo_path}...")
    parser = DemoParser(demo_path)

    # 1. Kill events
    print("Extracting kill events...")
    deaths = parser.parse_event("player_death")
    if deaths.empty:
        print("ERROR: No death events found.")
        return

    kills = deaths.dropna(subset=["attacker_steamid", "user_steamid"]).copy()
    kills["attacker_steamid"] = _norm_sid(kills["attacker_steamid"])
    kills["user_steamid"] = _norm_sid(kills["user_steamid"])
    kills = kills[
        (kills["attacker_steamid"] != kills["user_steamid"])  # suicides
        & (kills["attacker_steamid"] != "0")  # world damage
    ]
    if hitscan_only and "weapon" in kills.columns:
        w = kills["weapon"].astype(str).str.lower()
        kills = kills[~(w.isin(NON_HITSCAN) | w.str.contains("knife|bayonet"))]
    kills = kills.reset_index(drop=True)
    kills["kill_id"] = kills.index

    # 2. Tick data (duck_amount is needed for eye and hitbox heights)
    print("Parsing tick data...")
    ticks = parser.parse_ticks(
        ["tick", "steamid", "name", "pitch", "yaw", "X", "Y", "Z", "is_alive", "duck_amount"]
    )
    ticks = ticks.dropna(subset=["steamid"]).copy()
    ticks["steamid"] = _norm_sid(ticks["steamid"])
    duck = ticks["duck_amount"].fillna(0.0).clip(0.0, 1.0)
    ticks["eye_Z"] = ticks["Z"] + _lerp(EYE_STAND, EYE_CROUCH, duck)
    ticks["head_Z"] = ticks["Z"] + _lerp(HEAD_STAND, HEAD_CROUCH, duck)
    ticks["chest_Z"] = ticks["Z"] + _lerp(CHEST_STAND, CHEST_CROUCH, duck)

    # Attacker motion uses alive ticks only, split at deaths/rounds.
    alive = _add_motion_features(ticks[ticks["is_alive"] == True].copy(), tick_rate)
    attacker_frames = {sid: g for sid, g in alive.groupby("steamid")}
    # Victim positions come from ALL ticks, so the death tick itself
    # (when is_alive may already be False) still has a position.
    victim_frames = {
        sid: g.drop_duplicates("tick").set_index("tick") for sid, g in ticks.groupby("steamid")
    }

    # 3. Per-kill windows and aim error
    print("Computing crosshair-to-hitbox metrics...")
    window_ticks = int(round(window_seconds * tick_rate))
    pieces = []
    acquisitions = []

    for k in kills.itertuples(index=False):
        att = attacker_frames.get(k.attacker_steamid)
        vic = victim_frames.get(k.user_steamid)
        if att is None or vic is None:
            continue

        kt = int(k.tick)
        w = att[(att["tick"] >= kt - window_ticks) & (att["tick"] <= kt)].copy()
        if w.empty:
            continue

        v = vic.reindex(w["tick"].to_numpy())
        eye = np.column_stack([w["X"], w["Y"], w["eye_Z"]])
        fwd = _forward(w["pitch"], w["yaw"])
        head = np.column_stack([v["X"], v["Y"], v["head_Z"]])
        chest = np.column_stack([v["X"], v["Y"], v["chest_Z"]])

        dist, head_ang, head_miss, ideal_pitch, ideal_yaw = _aim_error(eye, fwd, head)
        _, chest_ang, chest_miss, _, _ = _aim_error(eye, fwd, chest)

        w["kill_id"] = k.kill_id
        w["kill_tick"] = kt
        w["ticks_before_kill"] = kt - w["tick"]
        w["target_steamid"] = k.user_steamid
        w["target_X"] = v["X"].to_numpy()
        w["target_Y"] = v["Y"].to_numpy()
        w["target_Z"] = v["Z"].to_numpy()
        w["spatial_dist_to_target"] = dist
        # Signed components show direction (yaw err + = aiming left of target).
        # Yaw degrees shrink by cos(pitch), so use the *_ang_err columns for size.
        w["target_pitch_err"] = w["pitch"].to_numpy() - ideal_pitch
        w["target_yaw_err"] = (w["yaw"].to_numpy() - ideal_yaw + 180) % 360 - 180
        w["head_ang_err"] = head_ang  # degrees off head centre
        w["head_miss_units"] = head_miss  # units off head centre
        w["chest_ang_err"] = chest_ang
        w["chest_miss_units"] = chest_miss
        pieces.append(w)

        acq = _acquisition_stats(
            w["ticks_before_kill"].to_numpy(), head_miss, head_ang,
            w["angular_velocity"].to_numpy(), tick_rate,
        )
        acq["kill_id"] = k.kill_id
        acquisitions.append(acq)

    if not pieces:
        print("No matching player telemetry found within kill windows.")
        return

    # Overlapping windows (multi-kills) keep one row per (tick, kill), so the
    # second victim's lead-up isn't thrown away.
    final = pd.concat(pieces, ignore_index=True).sort_values(["steamid", "kill_id", "tick"])

    # 4. Label cheaters
    sid_l = final["steamid"].str.strip().str.lower()
    name_l = final["name"].astype(str).str.strip().str.lower()
    final["is_cheater"] = (sid_l.isin(cheater_ids) | name_l.isin(cheater_ids)).astype(int)

    output_columns = [
        "kill_id", "kill_tick", "ticks_before_kill", "tick", "steamid", "name", "is_cheater",
        "pitch", "yaw", "X", "Y", "Z", "eye_Z", "duck_amount",
        "angular_velocity", "angular_accel", "angular_jerk", "delta_pitch", "delta_yaw",
        "target_steamid", "target_X", "target_Y", "target_Z", "spatial_dist_to_target",
        "target_pitch_err", "target_yaw_err",
        "head_ang_err", "head_miss_units", "chest_ang_err", "chest_miss_units",
    ]
    print(f"Writing per-tick dataset to {output_csv_path}...")
    final[output_columns].to_csv(output_csv_path, index=False)

    # 5. One row per kill: aim error on the tick the kill happened
    extra = [c for c in ["weapon", "headshot", "penetrated", "thrusmoke", "noscope",
                         "attackerblind", "distance"] if c in kills.columns]
    at_kill = final[final["ticks_before_kill"] == 0][[
        "kill_id", "kill_tick", "steamid", "name", "is_cheater", "target_steamid",
        "spatial_dist_to_target", "head_ang_err", "head_miss_units",
        "chest_ang_err", "chest_miss_units",
    ]]
    summary = at_kill.merge(kills[["kill_id"] + extra], on="kill_id", how="left")
    summary = summary.merge(pd.DataFrame(acquisitions), on="kill_id", how="left")
    summary.to_csv(summary_csv_path, index=False)

    print(f"Done! {len(final)} tick rows, {len(summary)} kills summarised.")


if __name__ == "__main__":
    export_kill_prewindow_telemetry(
        demo_path="AI-Create-Project/clips/mouz-nxt-vs-ursa-m1-dust2.dem",
        cheater_ids=["76561198843946848", "karnez"],
        output_csv_path="cs2_kill_prewindows_metrics.csv",
    )
