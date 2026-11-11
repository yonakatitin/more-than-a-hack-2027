"""Assemble the layers into one label per video frame: H / A / D / X.

    python -m src.pipeline [name]      -> evaluation/results/<name>.csv   (run from the repo root)
"""
import os
import sys
import numpy as np
import pandas as pd

from src.tracking import load_tracking, graded_mask, FPS, GRADED, N_FRAMES, ROOT  # noqa: E402
from src.control import control_from_ball, confirm_control, HOME, AWAY, NONE  # noqa: E402
from src.deadball import (median_player_speed, dead_from_set_pieces,  # noqa: E402
                          dead_before_kickoff)
import load_smart_tagging as st  # noqa: E402


def team_series(control, live):
    """Team in possession (HOME/AWAY/NONE) per frame.

    Computed separately inside every stretch of live play, so possession never carries over a
    stoppage: after a restart the first team that controls the ball owns the frames before it.
    Live stretches where nobody is ever seen controlling the ball borrow the nearest known team.
    """
    out = np.full(len(control), NONE, dtype=np.int8)
    edge = np.diff(np.concatenate([[0], live.astype(np.int8), [0]]))
    for s, e in zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1)):
        seg = confirm_control(control[s:e])
        seen = np.flatnonzero(seg != NONE)
        if len(seen):
            seg[:seen[0]] = seg[seen[0]]          # look-ahead: restart belongs to who takes the ball
        out[s:e] = seg
    known = np.flatnonzero(out != NONE)
    todo = np.flatnonzero((out == NONE) & live)
    if len(known) and len(todo):
        pos = np.searchsorted(known, todo)
        out[todo] = out[known[np.maximum(pos - 1, 0)]]
    return out


def build_labels():
    t = load_tracking()
    graded = graded_mask()
    control, _ = control_from_ball(t["ball"], t["players"], t["team"])
    speed = median_player_speed(t["players"])
    dead_sp, _ = dead_from_set_pieces(st.load(), speed)
    dead = (dead_sp | dead_before_kickoff()) & graded
    live = graded & ~dead
    team = team_series(control, live)
    labels = np.full(N_FRAMES, "X", dtype="<U1")
    labels[live & (team == HOME)] = "H"
    labels[live & (team == AWAY)] = "A"
    labels[dead] = "D"
    return labels


def main(name="v1"):
    labels = build_labels()
    path = os.path.join(ROOT, "evaluation", "results", name + ".csv")
    pd.DataFrame({"frame": np.arange(N_FRAMES), "label": labels}).to_csv(path, index=False)
    print("wrote", path, "({} rows)".format(len(labels)))
    g = graded_mask()
    print("graded frames:", {k: int((labels[g] == k).sum()) for k in "HADX"})
    for half, (s, e) in enumerate(GRADED, 1):
        seg = labels[s:e + 1]
        runs = 1 + int((seg[1:] != seg[:-1]).sum())
        print("half {}: {} label runs".format(half, runs))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "v1")