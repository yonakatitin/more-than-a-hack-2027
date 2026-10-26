"""Parse the tactical ATD once into numpy arrays and cache them (data/ is git-ignored).

Rows are VIDEO frames (row i == submission row i). ATD frame_id = video_frame + 1.
Coordinates are returned in metres, origin at the pitch's top-left corner as in the raw file.
"""
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES  # noqa: E402
from baseline import team_of_each_track  # noqa: E402

FPS = GAMES[GAME]["fps"]
N_FRAMES = GAMES[GAME]["n_video_frames"]
# Graded windows in video frames (DATA.md): 1st half and 2nd half, inclusive.
GRADED = ((11185, 90651), (110808, 190197))
DATA = os.path.join(ROOT, "data", "tactical")
RAW = os.path.join(DATA, "BAR-ATH_tactical_atd.txt")
META = os.path.join(DATA, "BAR-ATH_tactical_atd.xml")
CACHE = os.path.join(ROOT, "data", "cache", "tactical_tracking.npz")


def graded_mask(n=N_FRAMES):
    m = np.zeros(n, dtype=bool)
    for a, b in GRADED:
        m[a:b + 1] = True
    return m


def _parse():
    length, width = GAMES[GAME]["pitch"]
    ball = np.full((N_FRAMES, 2), np.nan)
    players = np.full((N_FRAMES, 46, 2), np.nan)
    pspeed = np.full((N_FRAMES, 46), np.nan)
    with open(RAW) as handle:
        for line in handle:
            fid, pl, bl = line.rstrip("\n").split(":")
            i = int(fid) - 1
            b = np.array(bl.split(","), dtype=float)
            ball[i] = b[:2]
            arr = np.array(pl.replace(";", ",").split(","), dtype=float).reshape(-1, 3)
            players[i, :len(arr)] = arr[:46, :2]
            pspeed[i, :len(arr)] = arr[:46, 2]
    ball *= (length, width)
    players *= (length, width)
    return ball, players, pspeed


def load_tracking(rebuild=False):
    """dict(ball (N,2) m, players (N,46,2) m, pspeed (N,46), team (46,) 0=home 1=away)."""
    if not rebuild and os.path.exists(CACHE):
        z = np.load(CACHE)
        return {k: z[k] for k in z.files}
    ball, players, pspeed = _parse()
    out = dict(ball=ball, players=players, pspeed=pspeed, team=team_of_each_track(META))
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, **out)
    return out