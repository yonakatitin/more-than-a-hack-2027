"""Layer 1: which team controls the ball while the ball is visible.

Rule: the nearest tracked player to the ball controls it, but only if close enough.
If nobody is close (ball loose or in the air) or the ball is not tracked, we do NOT change
team: possession stays with the last team that controlled the ball (CHALLENGE.md).
"""
import numpy as np

HOME, AWAY, NONE = 0, 1, -1

# Domain assumption, not tuned on this match: a player can only play the ball from within
# a couple of metres, and tracking position error is around a metre.
MAX_CONTROL_DIST_M = 2.5


def control_from_ball(ball, players, team, max_dist=MAX_CONTROL_DIST_M):
    """Per frame: HOME / AWAY if a player is within max_dist of the ball, else NONE.

    ball (N,2) m, players (N,46,2) m, team (46,) 0=home 1=away.
    Returns (control (N,) int8, nearest_distance (N,) m; inf when unknown).
    """
    d = np.hypot(players[:, :, 0] - ball[:, None, 0], players[:, :, 1] - ball[:, None, 1])
    d[:, ~np.isin(team, (HOME, AWAY))] = np.inf   # tracks with no team
    d = np.where(np.isnan(d), np.inf, d)          # missing ball or missing player
    nearest = d.argmin(axis=1)
    dmin = d[np.arange(len(d)), nearest]
    out = np.full(len(d), NONE, dtype=np.int8)
    ok = dmin <= max_dist
    out[ok] = team[nearest[ok]]
    return out, dmin


def hold_last(control):
    """Carry the last known team forward over NONE frames.

    Returns (held (N,) int8, age (N,) frames since the last real observation, -1 if none yet).
    """
    n = len(control)
    pos = np.arange(n)
    idx = np.maximum.accumulate(np.where(control != NONE, pos, -1))
    seen = idx >= 0
    held = np.full(n, NONE, dtype=np.int8)
    held[seen] = control[idx[seen]]
    age = np.where(seen, pos - idx, -1)
    return held, age