"""Layer 1 sanity check: uses only the provided tracking, never the ground truth."""
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from src.tracking import load_tracking, graded_mask, FPS, GRADED  # noqa: E402
from src.control import control_from_ball, hold_last, confirm_control, NONE  # noqa: E402


def run_lengths(x):
    change = np.flatnonzero(np.diff(x)) + 1
    edges = np.concatenate([[0], change, [len(x)]])
    return x[edges[:-1]], np.diff(edges)


t = load_tracking()
control, dmin = control_from_ball(t["ball"], t["players"], t["team"])
held, age = hold_last(control)
g = graded_mask()
visible = ~np.isnan(t["ball"][:, 0])

print("== 1. Coverage (graded frames) ==")
print("ball visible                      {:.1%}".format(visible[g].mean()))
print("ball visible AND a player <=2.5m  {:.1%}".format((control[g] != NONE).mean()))
vis_d = dmin[g & visible]
print("nearest-player distance when ball visible (m): "
      "p25 {:.1f}  median {:.1f}  p75 {:.1f}  p90 {:.1f}".format(*np.percentile(vis_d, [25, 50, 75, 90])))

print("\n== 2. How old is the information we hold on to? (graded frames) ==")
a = age[g]
for lo, hi, name in ((0, 0, "fresh (observed this frame)"), (1, FPS, "up to 1 s old"),
                     (FPS + 1, 5 * FPS, "1-5 s old"), (5 * FPS + 1, 20 * FPS, "5-20 s old"),
                     (20 * FPS + 1, 10 ** 9, "older than 20 s")):
    print("  {:<28} {:>6.1%}".format(name, ((a >= lo) & (a <= hi)).mean()))
print("  no information yet           {:>6.1%}".format((a < 0).mean()))

print("\n== 3. Possession runs of the held series, per half ==")
for name, (s, e) in zip(("1st half", "2nd half"), GRADED):
    vals, lens = run_lengths(held[s:e + 1])
    keep = vals != NONE
    sec = lens[keep] / FPS
    print("  {}: {} runs, median {:.1f} s, mean {:.1f} s, {:.0%} shorter than 2 s".format(
        name, len(sec), np.median(sec), sec.mean(), (sec < 2).mean()))

print("\n== 4. Same runs after requiring ~1 s of real control before a switch ==")
confirmed = confirm_control(control)
for name, (s, e) in zip(("1st half", "2nd half"), GRADED):
    vals, lens = run_lengths(confirmed[s:e + 1])
    keep = vals != NONE
    sec = lens[keep] / FPS
    print("  {}: {} runs, median {:.1f} s, mean {:.1f} s, {:.0%} shorter than 2 s".format(
        name, len(sec), np.median(sec), sec.mean(), (sec < 2).mean()))
print("frames whose label changed vs section 3: {:.1%}".format((confirmed[g] != held[g]).mean()))