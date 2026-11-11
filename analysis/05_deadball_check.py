"""Layer 3 sanity check: only tracking + Smart Tagging, never the ground truth."""
import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from src.tracking import load_tracking, graded_mask, FPS  # noqa: E402
from src.deadball import (median_player_speed, dead_from_set_pieces,  # noqa: E402
                          dead_before_kickoff, MAX_DEAD_FRAMES)
import load_smart_tagging as st  # noqa: E402

t = load_tracking()
speed = median_player_speed(t["players"])
dead_sp, info = dead_from_set_pieces(st.load(), speed)
dead_ko = dead_before_kickoff()
dead = dead_sp | dead_ko
g = graded_mask()
visible = ~np.isnan(t["ball"][:, 0])

print("== 1. Share of graded frames labelled dead ==")
print("set-piece windows {:.1%}, before kick off {:.1%}, together {:.1%}".format(
    dead_sp[g].mean(), dead_ko[g].mean(), dead[g].mean()))
print("ball visible on {:.1%} of the dead frames (all graded frames: {:.1%})".format(
    visible[dead & g].mean(), visible[g].mean()))

print("\n== 2. Per set-piece type ==")
df = pd.DataFrame(info, columns=["type", "start", "dead", "never_resumed"])
df["sec"] = df.dead / FPS
print(df.groupby("type").agg(
    windows=("sec", "size"), mean_s=("sec", "mean"), median_s=("sec", "median"),
    no_dead=("dead", lambda s: (s == 0).mean()), never_resumed=("never_resumed", "mean")
).round(2).to_string())
print("windows where play never resumed within the {} s cap: {:.0%}".format(
    MAX_DEAD_FRAMES // FPS, df.never_resumed.mean()))