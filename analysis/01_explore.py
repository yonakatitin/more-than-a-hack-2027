"""Step 1: understand ball gaps and what dead ball (D) looks like in the public ground truth."""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.tracking import load_tracking, graded_mask, FPS, ROOT  # noqa: E402
from games import GAMES, GAME  # noqa: E402

L, W = GAMES[GAME]["pitch"]
t = load_tracking()
ball, players, team = t["ball"], t["players"], t["team"]
graded = graded_mask()
has_ball = ~np.isnan(ball[:, 0])


def runs(mask):
    """(start, length) of each True run."""
    m = np.concatenate([[0], mask.astype(np.int8), [0]])
    d = np.diff(m)
    s, e = np.where(d == 1)[0], np.where(d == -1)[0]
    return s, e - s


print("pitch {} x {} m".format(L, W))
print("\n== 1. Ball gaps (graded frames) ==")
print("ball present on {:.1%} of graded frames".format(has_ball[graded].mean()))
s, ln = runs(~has_ball & graded)
sec = ln / FPS
print("{} gaps; median {:.2f}s, p90 {:.1f}s, p99 {:.1f}s, max {:.1f}s".format(
    len(ln), np.median(sec), np.percentile(sec, 90), np.percentile(sec, 99), sec.max()))
for lo, hi in ((0, 0.2), (0.2, 1), (1, 5), (5, 20), (20, 1e9)):
    sel = (sec > lo) & (sec <= hi)
    print("  gaps {:>5.1f}-{:<6.1f}s: {:>5} gaps, {:>5.1%} of all missing frames".format(
        lo, min(hi, 999), sel.sum(), ln[sel].sum() / ln.sum()))

print("\n== 2. Public ground truth window ==")
gt = pd.read_csv(os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv"))
lab = gt["label"].values
win = lab != "X"
idx = np.where(win)[0]
print("window frames {}..{} ({:.0f}s)".format(idx[0], idx[-1], len(idx) / FPS))
print("label share:", (pd.Series(lab[win]).value_counts(normalize=True)).round(3).to_dict())
for k in "HAD":
    m = lab == k
    print("  {}: ball present {:.1%}, mean tracked players {:.1f}".format(
        k, has_ball[m].mean(), (~np.isnan(players[m][:, :, 0])).sum(1).mean()))

print("\n== 3. Segments in the public ground truth ==")
rows = []
cur, start = lab[idx[0]], idx[0]
for i in idx[1:]:
    if lab[i] != cur:
        rows.append((cur, start, i - 1))
        cur, start = lab[i], i
rows.append((cur, start, idx[-1]))
for k, a, b in rows:
    seg = slice(a, b + 1)
    print("  {} {:>6}-{:<6} {:>5.1f}s  ball present {:>5.1%}".format(
        k, a, b, (b - a + 1) / FPS, has_ball[seg].mean()))

print("\n== 4. Ball speed and position, by label (public window) ==")
dxy = np.diff(ball, axis=0, prepend=np.nan)
speed = np.hypot(dxy[:, 0], dxy[:, 1]) * FPS  # m/s, NaN across gaps
inside = (ball[:, 0] >= 0) & (ball[:, 0] <= L) & (ball[:, 1] >= 0) & (ball[:, 1] <= W)
for k in "HAD":
    m = (lab == k) & has_ball
    print("  {}: median speed {:.1f} m/s, p90 {:.1f}; outside pitch {:.1%}; "
          "within 2m of a line {:.1%}".format(
              k, np.nanmedian(speed[m]), np.nanpercentile(speed[m], 90), (~inside[m]).mean(),
              ((np.minimum(ball[m, 0], L - ball[m, 0]) < 2) |
               (np.minimum(ball[m, 1], W - ball[m, 1]) < 2)).mean()))