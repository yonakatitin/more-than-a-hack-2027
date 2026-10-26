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

## Next
Step 1: explore ball gaps and the D pattern in the public ground truth.