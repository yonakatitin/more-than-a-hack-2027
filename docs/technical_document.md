# Technical document: ball possession (More than a Hack 2027, Phase 1)

Team: [TODO team name] · Python 3.14, numpy 2.5.4, pandas 3.1.0

## Approach
A transparent rule-based pipeline: no model, no training. Every parameter is a stated domain
assumption, not a value fitted to this match. Three layers:

1. **Control.** On each frame with a tracked ball, the nearest tracked player controls it if
   within 2.5 m; otherwise nobody does (ball loose, in the air, or untracked).
2. **Confirmation and hold.** A possession change is accepted only after the new team has controlled
   the ball for 25 frames (1 s, gaps between observations up to 1 s). The change is dated to the
   first such observation. While nobody controls the ball the last team keeps it, as the
   challenge rules say for loose balls.
3. **Dead ball.** A `SET PIECES` window from Smart Tagging (kick offs excluded) starts a dead-ball
   stretch, which ends when the players' median speed (smoothed over 1 s) stays above 1.5 m/s for
   3 s, capped at 20 s. All frames from the start of the second graded half to its kick off are
   also dead.

Assembly: possession is computed separately inside each stretch of live play, so it never carries
across a stoppage; the first team to control the ball after a restart owns the frames before it.
Frames outside the two halves are `X`.

## Assets used
- Tactical tracking (ATD): positions of the 46 track slots and the ball.
- Smart Tagging: only the `SET PIECES` windows and the two kick-off frames.
- Not used: video, broadcast feed, homography, Smart Tagging `Team` tag and phase codes.

## Training data
None. No labelled dataset was used and nothing was trained. No parameter was tuned on the
provided video or on the ground truth. [TODO: add external data here if we use any]

## Possession rules implemented
- Controlled possession needs a player within 2.5 m; passing close to the ball is not enough
  (the 1 s confirmation).
- Loose or untracked ball: possession stays with the last team that controlled it.
- Restart: ownership is not carried over a stoppage; the first team to control the ball owns it.
- Dead ball: set pieces and the pre-kick-off period only.

## Limitations
- When the ball is untracked for long stretches (about 41% of graded frames), the label only holds
  the last team. There is no model of play from the players alone.
- Dead ball is only detected around tagged set pieces; fouls, injuries, VAR and goals
  are missed, so `D` is underestimated.
- The first observation after a restart is trusted immediately.
- The thresholds are physical and football-based assumptions and were not checked on external data.
- The public 5-minute ground truth has only 17 possessions: it was used to check that the
  pipeline runs, not to choose between versions. [TODO: final public score]

## Reproduction
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell
pip install -r requirements.txt
# unpack the data bundle into ./data (or set MTAH_DATA_DIR)
python -m src.pipeline <name>        # writes evaluation/results/<name>.csv
python evaluation/validate_submission.py evaluation/results/<name>.csv
```
The first run parses the tracking file and caches it under `data/cache` (about 1-2 minutes).
Re-running gives a byte-identical file.