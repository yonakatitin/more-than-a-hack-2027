"""Step 2a: what does Smart Tagging actually say, and can it help where the ball is missing?"""
import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from src.tracking import load_tracking, graded_mask, FPS, N_FRAMES  # noqa: E402
import load_smart_tagging as st  # noqa: E402

TEAM = {"Barcelona": "H", "Atletic Club": "A"}
OTHER = {"H": "A", "A": "H"}
events = st.load()

print("== 1. What is in Smart Tagging ==")
df = pd.DataFrame([{"code": e["code"], "type": e["tags"].get("Type"),
                    "dur_s": (e["end_frame"] - e["start_frame"]) / FPS} for e in events])
print(df.groupby("code").agg(n=("dur_s", "size"), mean_s=("dur_s", "mean")).round(1).to_string())
print(df[df.code == "SET PIECES"].groupby("type").agg(
    n=("dur_s", "size"), mean_s=("dur_s", "mean")).round(1).to_string())

print("\n== 2. Does the Team tag mean 'has the ball'? (public ground truth window) ==")
gt = pd.read_csv(os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv"))
lab = gt["label"].values
rows = {}
for e in events:
    team = TEAM.get(e["tags"].get("Team"))
    if team is None or e["code"] == "SET PIECES":
        continue
    seg = lab[e["start_frame"]:e["end_frame"]]
    seg = seg[np.isin(seg, ["H", "A"])]
    if len(seg) == 0:
        continue
    r = rows.setdefault(e["code"], [0, 0, 0])
    r[0] += 1
    r[1] += int((seg == team).sum())
    r[2] += int((seg != team).sum())
print("{:<30} {:>6} {:>7} {:>10}".format("code", "events", "frames", "tag == GT"))
for code, (n, ok, bad) in sorted(rows.items(), key=lambda kv: -kv[1][1] / (kv[1][1] + kv[1][2])):
    print("{:<30} {:>6} {:>7} {:>9.0%}".format(code, n, ok + bad, ok / (ok + bad)))
print("(few events per code here, so read this as a hint, not a measurement)")

POSSESSOR = {"BUILD UP", "PROGRESSION", "CREATING CHANCES", "LONG BALL", "ATTACKING TRANSITION"}
DEFENDER = {"DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD",
            "DEFENDING IN DEFENSIVE THIRD", "DEFENSIVE TRANSITION"}
signed = np.full(N_FRAMES, "", dtype=object)   # team we believe has the ball, from the tags
setpiece = np.zeros(N_FRAMES, dtype=bool)
anytag = np.zeros(N_FRAMES, dtype=bool)
for e in events:
    a, b = e["start_frame"], e["end_frame"]
    anytag[a:b] = True
    team = TEAM.get(e["tags"].get("Team"))
    if e["code"] == "SET PIECES":
        setpiece[a:b] = True
    elif team and e["code"] in POSSESSOR:
        signed[a:b] = team
    elif team and e["code"] in DEFENDER:
        signed[a:b] = OTHER[team]

graded = graded_mask()
m = lab != "X"
print("\n== 3. Tags used as 'who has the ball' (possessor codes as-is, defender codes flipped) ==")
print("covers {:.1%} of graded frames".format((signed[graded] != "").mean()))
sel = m & np.isin(lab, ["H", "A"]) & (signed != "")
print("on the public window it covers {:.1%} of H/A frames and is right on {:.1%} of those".format(
    sel.sum() / (m & np.isin(lab, ["H", "A"])).sum(), (signed[sel] == lab[sel]).mean()))

print("\n== 4. Long ball gaps (> 5 s) vs the tags ==")
t = load_tracking()
has_ball = ~np.isnan(t["ball"][:, 0])
x = np.concatenate([[0], (~has_ball & graded).astype(np.int8), [0]])
d = np.diff(x)
starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
gap = np.zeros(N_FRAMES, dtype=bool)
for a, b in zip(starts, ends):
    if (b - a) / FPS > 5:
        gap[a:b] = True
print("long gaps cover {:.1%} of graded frames".format(gap[graded].mean()))
print("  inside a set-piece window     {:.1%}".format(setpiece[gap].mean()))
print("  covered by a signed team tag  {:.1%}".format((signed[gap] != "").mean()))
print("  no tag at all                 {:.1%}".format((~anytag[gap]).mean()))