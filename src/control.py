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



# Domain assumption, not tuned on this match: a real change of possession needs the new team
# to actually control the ball for about one second (25 frames), not just pass close by.
MIN_CONTROL_FRAMES = 25


def confirm_control(control, min_frames=MIN_CONTROL_FRAMES):
    """Possession series that only changes team after the new team has controlled the ball
    for at least min_frames frames (observations may be interrupted by gaps of up to
    min_frames frames). The switch is dated back to when the new team first took control,
    so confirming adds no delay. Frames before the first observation stay NONE.
    """
    n = len(control)
    events = []
    cur = NONE
    cand, cand_start, cand_count, cand_last = NONE, -1, 0, -1
    for i in range(n):
        c = control[i]
        if c == NONE:
            continue
        if cur == NONE:                      # first ever observation
            cur = c
            events.append((i, c))
            continue
        if c == cur:                         # current team seen again: drop any candidate
            cand, cand_count = NONE, 0
            continue
        if c != cand or i - cand_last > min_frames:
            cand, cand_start, cand_count = c, i, 0
        cand_count += 1
        cand_last = i
        if cand_count >= min_frames:
            cur = c
            events.append((cand_start, c))
            cand, cand_count = NONE, 0
    held = np.full(n, NONE, dtype=np.int8)
    for start, team in events:
        held[start:] = team
    return held