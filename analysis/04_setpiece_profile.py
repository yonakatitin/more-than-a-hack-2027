"""Step 3a: what does a set-piece window look like in the tracking? Uses NO ground truth."""
import os
import sys
import warnings
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from src.tracking import load_tracking, graded_mask, FPS  # noqa: E402
from games import GAMES, GAME  # noqa: E402
import load_smart_tagging as st  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

L, W = GAMES[GAME]["pitch"]
t = load_tracking()
ball, players = t["ball"], t["players"]
g = graded_mask()

# Physical limits, not tuned: faster than this is a tracking jump, not a real player/ball.
MAX_PLAYER_SPEED = 12.0   # m/s
MAX_BALL_SPEED = 45.0     # m/s


def speed(xy, limit):
    d = np.diff(xy, axis=0, prepend=np.nan)
    s = np.hypot(d[..., 0], d[..., 1]) * FPS
    s[s > limit] = np.nan
    return s


med_player = np.nanmedian(speed(players, MAX_PLAYER_SPEED), axis=1)   # (N,) m/s, robust to bad tracks
ball_speed = speed(ball, MAX_BALL_SPEED)
visible = ~np.isnan(ball[:, 0])
outside = visible & ((ball[:, 0] < 0) | (ball[:, 0] > L) | (ball[:, 1] < 0) | (ball[:, 1] > W))

print("== Reference over all graded frames ==")
print("median of the players' median speed: {:.2f} m/s; frames below 1 m/s: {:.1%}".format(
    np.nanmedian(med_player[g]), np.nanmean(med_player[g] < 1.0)))

sets = [e for e in st.load()
        if e["code"] == "SET PIECES" and not str(e["tags"].get("Type", "")).startswith("Kick Off")
        and g[e["start_frame"]] and g[e["end_frame"] - 1]]
BIN = 5 * FPS
for typ in sorted({e["tags"].get("Type") for e in sets}):
    evs = [e for e in sets if e["tags"].get("Type") == typ]
    nb = min((e["end_frame"] - e["start_frame"]) // BIN for e in evs)
    print("\n== {} ({} windows, profile per 5 s from window start) ==".format(typ, len(evs)))
    print("  {:>9}  {:>12}  {:>12}  {:>12}  {:>12}".format(
        "seconds", "player m/s", "ball seen", "ball m/s", "ball outside"))
    for k in range(nb):
        idx = np.concatenate([np.arange(e["start_frame"] + k * BIN, e["start_frame"] + (k + 1) * BIN)
                              for e in evs])
        print("  {:>4}-{:<4}  {:>12.2f}  {:>12.1%}  {:>12.2f}  {:>12.1%}".format(
            k * 5, (k + 1) * 5, np.nanmedian(med_player[idx]), visible[idx].mean(),
            np.nanmedian(ball_speed[idx]), outside[idx].sum() / max(1, visible[idx].sum())))
    where, low = [], []
    for e in evs:
        s = pd.Series(med_player[e["start_frame"]:e["end_frame"]]).rolling(
            FPS, min_periods=FPS // 2, center=True).mean()
        if s.notna().any():
            where.append(s.idxmin() / FPS)
            low.append(s.min())
    print("  slowest moment of the players, seconds after window start: "
          "p25 {:.0f}  median {:.0f}  p75 {:.0f}   (their speed there: median {:.2f} m/s)".format(
              *np.percentile(where, [25, 50, 75]), np.median(low)))