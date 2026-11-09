# Progress Notes

## Step 0: Reproduce baseline
Data placed in `data/` (git-ignored). Baseline rebuilt from raw tracking.

Public 5-minute ground truth (17 possessions, very noisy):

| submission         | macroF1 | spellF1 | poss | dur  |
|--------------------|---------|---------|------|------|
| baseline_smooth125 | 0.5812  | 0.4242  | 16   | 18.5 |
| baseline_smooth50  | 0.6207  | 0.3256  | 26   | 10.8 |
| baseline_smooth25  | 0.6200  | 0.2868  | 36   | 7.6  |
| baseline_raw       | 0.5972  | 0.0378  | 131  | 2.0  |

Published floor on the full match: spellF1 0.2433 (smooth50).

## Observations
- Baseline labels every missing-ball frame as D, so D is heavily over-predicted.
- Do not tune the smoothing window on the public 5 minutes.

## Step 1: Ball gaps and dead ball (analysis/01_explore.py)
- Ball is present on 59.3% of graded frames. 150 gaps, median 10.7 s, 44 gaps longer than 20 s
  holding 67% of all missing frames. Gaps are long, so interpolation alone will not fix them.
- Dead ball (D) is not the same as missing ball: two D segments in the public ground truth
  (20.4 s and 14.5 s) have the ball visible on 100% of frames.
- D has a lower median ball speed (2.7 m/s vs 5.3 H and 6.6 A) and 10.7% of D frames have the
  ball outside the pitch.
- The public 5 minutes is not representative: the ball is visible on ~99.9% of H frames there,
  versus 59% across the match. Missing-ball handling cannot be validated on it.

## Next
Step 2: separate dead ball from missing ball; check how far Smart Tagging can help where the
ball is missing.