"""Layer 3: dead ball. Uses only tracking + Smart Tagging, never the ground truth.

A set-piece window from Smart Tagging says WHERE a stoppage happens; the players' speed says
WHEN play has resumed. Dead ball = from the start of the window until play has resumed.
"""
import warnings
import numpy as np
import pandas as pd

from src.tracking import FPS, GRADED, N_FRAMES  # noqa: E402  (also sets up sys.path)
from load_smart_tagging import KICKOFF_VIDEO_FRAMES  # noqa: E402

# Assumptions from football and physics, NOT tuned on this match.
RESUME_SPEED = 1.5            # m/s: a bit above relaxed walking (~1.3-1.4 m/s)
HOLD_FRAMES = 3 * FPS         # play counts as resumed once players stay that fast for 3 s
MAX_DEAD_FRAMES = 20 * FPS    # a restart rarely takes longer; also caps the damage if speed never rises
SMOOTH_FRAMES = FPS           # 1 s smoothing of the speed signal
MAX_PLAYER_SPEED = 12.0       # m/s: faster than this is a tracking jump


def median_player_speed(players):
    """(N,) smoothed median speed of the tracked players, m/s. NaN where nobody is tracked."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        d = np.diff(players, axis=0, prepend=np.nan)
        s = np.hypot(d[..., 0], d[..., 1]) * FPS
        s[s > MAX_PLAYER_SPEED] = np.nan
        med = np.nanmedian(s, axis=1)
    smooth = pd.Series(med).rolling(SMOOTH_FRAMES, min_periods=SMOOTH_FRAMES // 2, center=True).mean()
    return smooth.to_numpy()


def forward_min(speed, frames):
    """For each frame, the slowest speed over the next `frames` frames (itself included)."""
    rev = pd.Series(speed[::-1]).rolling(frames, min_periods=frames // 2).min()
    return rev.to_numpy()[::-1]


def dead_from_set_pieces(events, speed, n=N_FRAMES):
    """Dead-ball mask from set-piece windows, plus one info row per window.

    Kick offs are skipped: the tag marks the restart itself, the stoppage comes before it.
    info rows: (type, window_start_frame, dead_frames, never_resumed).
    """
    sustained = forward_min(speed, HOLD_FRAMES)
    dead = np.zeros(n, dtype=bool)
    info = []
    for e in events:
        typ = str(e["tags"].get("Type", ""))
        if e["code"] != "SET PIECES" or typ.startswith("Kick Off"):
            continue
        a = e["start_frame"]
        b = min(e["end_frame"], a + MAX_DEAD_FRAMES, n)
        hit = np.flatnonzero(sustained[a:b] > RESUME_SPEED)
        resume = a + int(hit[0]) if len(hit) else b
        dead[a:resume] = True
        info.append((typ, a, resume - a, len(hit) == 0))
    return dead, info


def dead_before_kickoff(n=N_FRAMES):
    """Ball is dead from the start of each graded block until the kick off whistle."""
    dead = np.zeros(n, dtype=bool)
    for (start, _end), half in zip(GRADED, ("1st Half", "2nd Half")):
        dead[start:KICKOFF_VIDEO_FRAMES[half]] = True
    return dead