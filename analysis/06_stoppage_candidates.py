"""Step 3b: stoppages that the Smart Tagging set-piece windows do NOT cover.
Uses only tracking + tags, never the ground truth."""
import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from src.tracking import load_tracking, graded_mask, FPS  # noqa: E402
from src.deadball import median_player_speed  # noqa: E402
from games import GAMES, GAME  # noqa: E402
import load_smart_tagging as st  # noqa: E402

L, W = GAMES[GAME]["pitch"]
t = load_tracking()
ball, players = t["ball"], t["players"]
g = graded_mask()
n = len(g)
speed = median_player_speed(players)
visible = ~np.isnan(ball[:, 0])
outside = visible & ((ball[:, 0] < 0) | (ball[:, 0] > L) | (ball[:, 1] < 0) | (ball[:, 1] > W))

in_sp = np.zeros(n, dtype=bool)
for e in st.load():
    if e["code"] == "SET PIECES":
        in_sp[e["start_frame"]:e["end_frame"]] = True
MARGIN = 10 * FPS
near_sp = pd.Series(in_sp.astype(np.int8)).rolling(
    2 * MARGIN + 1, center=True, min_periods=1).max().to_numpy().astype(bool)


def runs(mask):
    m = np.concatenate([[0], mask.astype(np.int8), [0]])
    d = np.diff(m)
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return s, e


def report(name, mask, min_frames):
    s, e = runs(mask & g)
    keep = (e - s) >= min_frames
    s, e = s[keep], e[keep]
    cand = np.zeros(n, dtype=bool)
    for a, b in zip(s, e):
        cand[a:b] = True
    total = cand[g].sum()
    print("\n== {} (runs >= {:.0f} s) ==".format(name, min_frames / FPS))
    print("{} runs, {:.1%} of graded frames".format(len(s), total / g.sum()))
    if total == 0:
        return
    print("frames inside a set-piece window {:.0%}, within 10 s of one {:.0%}, far from any {:.0%}".format(
        (cand & in_sp)[g].sum() / total, (cand & near_sp & ~in_sp)[g].sum() / total,
        (cand & ~near_sp)[g].sum() / total))
    far = [(a, b) for a, b in zip(s, e) if not near_sp[a:b].any()]
    if far:
        sec = np.array([(b - a) / FPS for a, b in far])
        print("runs far from any set piece: {} (median {:.0f} s, p90 {:.0f} s, max {:.0f} s), "
              "{:.1%} of graded frames".format(len(far), np.median(sec), np.percentile(sec, 90),
                                               sec.max(), sec.sum() * FPS / g.sum()))
        fm = np.zeros(n, dtype=bool)
        for a, b in far:
            fm[a:b] = True
        print("  ball visible on {:.0%} of those frames".format(visible[fm].mean()))


print("reference: graded frames with players' median speed < 0.5 m/s: {:.1%}".format(
    np.nanmean(speed[g] < 0.5)))
report("Ball outside the pitch", outside, 2 * FPS)
report("All players nearly still (< 0.5 m/s)", speed < 0.5, 5 * FPS)