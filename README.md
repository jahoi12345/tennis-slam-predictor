# Grand Slam Winner & Margin Predictor

Predicts, for a given Grand Slam match: (1) who wins, and (2) by how much
(the set-score margin, e.g. 3-1). Covers both ATP and WTA.

## Status

Data ingestion, Elo/form/head-to-head features, the win classifier, the
margin classifier, end-to-end prediction, **live in-match win probability**
(notebook 06), **bookmaker odds as a feature** (notebook 07), and the
**data visualization piece** are all done — see
[Baseline Studies](https://claude.ai/code/artifact/702398f9-b979-4f00-8a49-f12a42cfd83c)
(three bespoke pieces built from this project's own data: a radial career-Elo
chart, a Grand Slam draw redrawn as a radial bracket, and a Sinner–Alcaraz
head-to-head) and
[Grand Slam Predictor Summary](https://claude.ai/code/artifact/817be16c-d366-4753-8d24-13387354e867)
(pipeline, results, findings). Only the games-differential-regression stretch
goal remains open.

### Bookmaker odds (notebook 07)

`www.tennis-data.co.uk` was down site-wide (HTTP 503) for a stretch of this
work — confirmed as a real outage (identical error via `curl` and a real
browser session, and the operator's sister site football-data.co.uk was
down too), not something specific to this project. It recovered, and
separately, the **bare domain without `www.`** (`tennis-data.co.uk`) turned
out to work throughout even when `www.` didn't — worth remembering if it
happens again.

Source: ATP season files 2000-2026, WTA 2007-2026, one Excel file per tour
per year. The real work was the join, not the download: this source
identifies players as `"Djokovic N."` (surname + initial), and — checked
directly against the actual files rather than assumed — the *same player*
is formatted inconsistently across different years (`"Del Potro J.M."`,
`"Del Potro J. M."`, and `"Del Potro J."` all appear for the same person).
`src/odds.py` matches on surname + **first initial only** (extracted via
regex, ignoring unreliable middle initials), disambiguating same-surname
collisions within a date window using tournament-name similarity
(`rapidfuzz`). Match rate: 89-90% for both tours on the actual Slam 2022+
evaluation window that matters most.

**This is a real, non-noise improvement** — a meaningfully larger gap than
anything found chasing model architecture in notebook 03's round 2:

| Tour | Our features alone | Market odds alone | **Combined** | Naive market-favorite |
|---|---|---|---|---|
| ATP | 71.1% acc, AUC 0.788 | 71.5% acc, AUC 0.808 | **73.3% acc, AUC 0.812** | 74.3% acc |
| WTA | 68.6% acc, AUC 0.759 | 69.0% acc, AUC 0.776 | **70.8% acc, AUC 0.779** | 71.2% acc |

(All three trained on the same odds-available window — 2001+/2008+ — for a
fair comparison; not the full 1968-2026 history notebook 03 uses, which
would make this look like a different, incomparable gain.) Combining closes
most of the gap to the professional betting market's own floor, and beats
"market alone" on AUC for both tours — real information in both directions,
not just the market doing all the work. Saved as a second production
artifact (`models/win_classifier_{tour}_ensemble_with_odds.joblib`) — real
future matches don't always have odds attached, so `predict.predict_matchup`
now takes optional `p1_odds`/`p2_odds` and uses this model only when
they're supplied, falling back to the original odds-free ensemble
otherwise.

### Live in-match win probability (notebook 06)

A different question from everything above: not "who wins before the
match starts," but "given the score right now, what's the win probability,
and how does that improve as the match progresses?" Uses the Match
Charting Project's point-by-point files (1.85M points across 11,600+
charted matches, both tours) — data that had been sitting in
`data/external/` since notebook 01's ingestion, previously used only for
aggregate style stats.

Two approaches, in `src/inmatch.py`: a **hierarchical Markov chain**
(Klaassen & Magnus 2003 — exact win probability from any live score, given
each player's own-serve point-win probability, no training data at all) and
a **learned model** (logistic regression / gradient boosting on live score
state plus the full pre-match feature set). The learned model beats the
analytical one at every stage of the match, and both climb well past the
pre-match ceiling once the match is underway:

| Match progress | Pre-match baseline | Analytical (Markov) | Learned (LR) |
|---|---|---|---|
| 0-10% | 66.3% (ATP) / 66.3% (WTA) | 64.9% / 67.5% | 67.6% / 68.5% |
| 50-75% | 66.4% / 66.3% | 77.2% / 79.9% | 78.2% / 80.8% |
| 90-100% | 66.4% / 66.5% | 92.0% / 94.2% | 91.6% / 94.0% |

**This is the real answer to "can we get to 90% accuracy"**: not as a flat
pre-match number (that ceiling is ~70-75%, confirmed against the sports-
prediction literature) — but as a live-updating curve that legitimately
crosses 90% in the last tenth of a match, for both tours. Two engineering
gotchas worth knowing about if extending this: the Charting Project's `Pts`
field is formatted "server's score first" (the universal tennis
broadcasting convention), not a fixed player1/player2 ordering — this was
only caught by tracing a real tiebreak point-by-point and finding
contradictions; and the Markov chain's naive recursion doesn't terminate at
a deep tie (a tiebreak stuck at 6-6+, or an old-style advantage final set
past 6-6 with no tiebreak) since ties can persist indefinitely — both
needed a closed-form absorption-probability fix instead of just recursing
deeper. Research-notebook only for now (no live score feed to drive it
automatically); a callable `predict_live()` is a natural next step.

**End-to-end prediction** (`notebooks/05_end_to_end_prediction.ipynb`,
`src/predict.py`): given two player names, a surface/round/date, returns win
probabilities plus a margin distribution. The nuance this required: the
margin model is winner-oriented, but at prediction time you don't know who
wins — so `predict_matchup` computes the margin distribution twice (once
assuming each player wins) and mixes them using the win probabilities as
weights, rather than naively calling both models independently. Example:

```
Jannik Sinner vs Carlos Alcaraz  (ATP, Clay, F)
  Jannik Sinner: 72% to win
  Carlos Alcaraz: 28% to win
  Overall margin distribution: 3-0: 40%, 3-1: 38%, 3-2: 23%
  If Sinner wins, likely: 3-0 (41%), 3-1 (38%), 3-2 (21%)
  If Alcaraz wins, likely: 3-0 (37%), 3-1 (36%), 3-2 (27%)
```

Sanity-checked against the actual 2025 Roland Garros final (predicted as of
the day before, using only information available then): gave Sinner a clear
72% edge — a reasonable pre-match view (Sinner was the higher-form, higher-
surface-Elo player) of a match Alcaraz went on to win in five sets after
trailing two sets to love.

### Trying to beat plain Elo

Round 1 (see `notebooks/03_win_classifier.ipynb` sections 3 and 8) went
after the original finding that nothing beat a bare Elo-favorite baseline:

1. **Margin-of-victory Elo** (`features.mov_multiplier`) — a 6-0 6-0 6-0
   sweep now moves ratings more than a 7-6 7-6 7-6 win of the same match,
   adapted from FiveThirtyEight's NBA/NFL Elo. Ablated directly in notebook
   02: **ATP 70.7% → 71.4%, WTA 68.5% → 69.3%** Elo-favorite accuracy.
2. **Blended surface+overall Elo** (`dataset.blended_elo`, 80/20 weighted
   toward surface) — became the single most important ATP feature by
   permutation importance, ahead of raw `diff_elo`.
3. **Hyperparameter tuning** for gradient boosting via `TimeSeriesSplit`
   CV — best-found parameters were a *simpler, more regularized* tree than
   sklearn's defaults, reinforcing rather than reversing the "this problem
   doesn't reward more model capacity" finding.

Round 2 (section 9) went broader: neural nets, and two ways of combining
what already existed.

4. **Neural network** (small MLP) — tested, didn't help, as expected: tabular
   data with one dominant near-linear signal and ~25-28k rows is exactly the
   regime where neural nets are documented to have no edge over trees/linear
   models. Included for completeness, not adopted.
5. **Explicit Elo × surface/round interaction terms** for logistic
   regression — a linear model can't discover "the Elo gap matters more in
   a final than round 1" on its own; it needs the product term spelled out.
   `elo_x_round` came out as ATP's **3rd most important feature**, and a
   round-depth calibration check confirmed the effect is real: the model's
   extra confidence in QF/SF/F matches is *better calibrated* there (higher
   AUC and lower Brier score for that subset), not just louder.
6. **Stacking ensemble** combining that interaction-aware logistic
   regression with tuned gradient boosting, via cross-validated meta-learning.

**Important caveat**: the 2022+ test set is ~2,270 matches per tour: at ~71%
accuracy, the standard error alone is about **1 percentage point**. Nearly
every number below is within noise of every other one — including the
original "does gradient boosting beat logistic regression" question. The
stacking ensemble is adopted as the new production model because it's a
theoretically-motivated, never-worse combination (ensembling reduces
variance; the interaction terms fix a real gap in what plain logistic
regression could represent) — not because any single number here proves
superiority.

**Win classifier results**, chronological split — train on 1968-2021, test on 2022+:

| Tour | Elo baseline | Logistic Reg. | Grad. Boosting (tuned) | LR + interactions | **Stacking (production)** |
|---|---|---|---|---|---|
| ATP | 71.4% acc | 71.5% acc, AUC 0.788 | 71.5% acc, AUC 0.788 | 71.6% acc, AUC 0.790 | **71.7% acc, AUC 0.790** |
| WTA | 69.3% acc | 68.8% acc, AUC 0.760 | 68.5% acc, AUC 0.759 | 68.7% acc, AUC 0.761 | **68.6% acc, AUC 0.761** |

Net effect since the original baseline: roughly +0.6-1.0 points of ATP
accuracy and +0.2 AUC; WTA gains are smaller and mostly came from the
Elo-baseline improvement itself (round 1) rather than model choice (round 2)
— consistent with the noise-floor caveat above.

**Margin classifier results** (`notebooks/04_margin_classifier.ipynb`), same
split, predicting the set-score bucket (ATP: 3-0/3-1/3-2, WTA: 2-0/2-1),
using the win classifier's probability as an input feature:

| Tour | Majority baseline | Logistic Reg. | Gradient Boosting | **LR + analytical (production)** |
|---|---|---|---|---|
| ATP | 44.5% acc, F1 0.21 | 45.0% acc, F1 0.29 | 44.0% acc, F1 0.30 | **45.2% acc, F1 0.29** |
| WTA | 68.0% acc, F1 0.40 | 68.4% acc, F1 0.46 | 68.1% acc, F1 0.45 | **68.4% acc, F1 0.43** |

Accuracy barely moves — but that's a metric artifact, not a failed model:
bucketing matches by the win classifier's probability shows P(3-0) climbing
cleanly from ~37% (close matches) to ~67% (heavy favorites), exactly the
expected direction. The problem is that 3-0 stays the single most likely
outcome across almost the whole probability range, so the *argmax*
prediction rarely flips even as the real probabilities shift a lot — visible
directly in the confusion matrix, where the model predicts "3-0" for 77-87%
of matches regardless of the true outcome. **Practical takeaway**: the useful
output here is the full probability distribution over margin buckets, not a
forced single-class prediction — a better fit for "by how much" anyway.

**Further exploration** (notebook 04 section 6), mirroring the win
classifier's round 2: a tennis-specific **analytical model** was added —
given only the match win probability, derive the exact margin distribution
from the standard race-to-N formulas assuming i.i.d. sets (no training, no
overfitting risk, but also no access to surface/H2H/style). Used alone, it
underperforms the learned models (real matches aren't perfectly i.i.d. per
set — momentum, fatigue). But feeding its three probabilities in as
*extra features* to plain logistic regression gave the best all-around
result of everything tried — beating a neural net, a stacking ensemble, and
(ATP only, since it's the one tour with 3 ordered classes) a proper ordinal
logistic regression that directly encodes 3-0 < 3-1 < 3-2. **Adopted as the
new production margin model for both tours** — the simplest model in the
whole comparison, at or near the best on every metric. Same noise-floor
caveat as the win classifier applies: these are all within roughly a
percentage point of each other at this test-set size.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then fetch the raw data (see [`data/external/README.md`](data/external/README.md)
— the Sackmann sources clone via `git`, but tennis-data.co.uk's odds files
need downloading by hand into `data/external/tennis-data-co-uk/{atp,wta}/`)
and run `notebooks/01_data_ingestion.ipynb` to produce the processed parquet
files in `data/processed/` (gitignored — regenerate locally). Notebook 07
needs the odds files; every other notebook works without them.

## Project layout

```
data/
  external/     raw cloned source repos (gitignored)
  processed/    cleaned parquet outputs from ingestion (gitignored)
  processed/viz/  small JSON exports for the Baseline Studies visualizations (gitignored)
notebooks/      the actual analysis, in order
src/            reusable loading/feature/dataset/prediction/in-match functions shared across notebooks
scripts/        one-off generator scripts (e.g. builds a notebook from source)
models/         fitted win/margin/live-win-probability pipelines (gitignored — regenerate locally)
```

## Data sources

- **Match history, rankings, players**: Jeff Sackmann's ATP/WTA archives
  (1968–2026), main tour-level singles matches. Used both for the Slam
  prediction labels and, over the full non-Slam history, for computing Elo
  and recent-form features.
- **Playing style — Match Charting Project**: volunteer-charted shot-by-shot
  data. Gives per-player forehand/backhand winner and unforced-error shares,
  rally-length preference, net-point frequency, and serve-and-volley rate.
  Coverage is volunteer-selected (skews toward top players), not a full
  census.
- **Playing style — Grand Slam point-by-point**: official ball-tracking data
  from the four majors (2011–2024). The only public source of actual serve
  speed; also gives forehand-vs-backhand winner shot type. Column
  availability varies by slam/year (two different data providers), and the
  Australian Open/French Open stopped being captured after 2022.
- **Bookmaker odds — tennis-data.co.uk**: ATP 2000-2026, WTA 2007-2026,
  free public dataset (used in prior academic work on this exact problem,
  e.g. Cornman et al. combined this same source with Sackmann's data). A
  separate license from the sources above — see `data/external/README.md`.

All of the above from Jeff Sackmann / Tennis Abstract is CC BY-NC-SA 4.0 —
**attribution required, non-commercial use only**. See
`data/external/README.md` and each source's own `UPSTREAM_README.md`.

Weather / day-night session data still aren't in any of these sources. A
**live score feed** is still explicitly out of scope — no clean free API
exists — but live in-match *modeling* turned out not to need one: notebook
06's point-by-point training data comes from the Match Charting Project's
historical charted matches, not a live feed.

## Modeling plan

1. ~~**Elo + form + head-to-head features**, computed over full match history
   (not just Slams), surface-specific.~~ Done — see `notebooks/02_elo_and_form.ipynb`
   and `src/features.py`. Elo uses an experience-adjusted K-factor (new
   players' ratings move faster); overall + per-surface ratings are tracked
   independently for ATP and WTA.
2. ~~**Win classifier**: logistic regression baseline → gradient boosting,
   trained/evaluated separately per tour, chronological (not random)
   train/test split.~~ Done — see `notebooks/03_win_classifier.ipynb`,
   `src/dataset.py` (symmetric player1/player2 framing), and results above.
3. ~~**Margin classifier**: set-score bucket (ATP: 3-0/3-1/3-2, WTA: 2-0/2-1),
   using the win model's probability as an input feature.~~ Done — see
   `notebooks/04_margin_classifier.ipynb`, `src/dataset.py`
   (`parse_set_score`, `build_margin_dataset`), and results above.
4. ~~**Combine into one end-to-end prediction**: given two players, run both
   models together and report "player X wins (68%), most likely 3-1 (35%),
   3-0 (30%), 3-2 (22%)" rather than two separate notebook outputs.~~ Done —
   see `notebooks/05_end_to_end_prediction.ipynb`, `src/predict.py`
   (`predict_matchup`), and the example above.
5. ~~**Try to beat the plain-Elo baseline**: margin-of-victory Elo, a
   blended surface+overall Elo feature, and gradient-boosting hyperparameter
   tuning.~~ Done — see "Trying to beat plain Elo" above.
6. ~~**Explore further model families**: neural nets, ensembling/stacking,
   explicit interaction features — and address whether LLMs are a fit.~~
   Done — see "Trying to beat plain Elo" (round 2) above. Neural nets don't
   help (as expected for this kind of tabular data); LLMs were assessed and
   deliberately not tried (no structural fit — see notebook 03's closing
   note); stacking + interaction terms are the small, real improvement now
   in production.
7. ~~**Same exploration for the margin classifier**: neural net, stacking,
   plus an ordinal-regression angle specific to margin's ordered classes,
   and a tennis-specific analytical (i.i.d.-sets) alternative.~~ Done — see
   "Margin classifier results" and "Further exploration" above. The
   analytical model's probabilities as extra features for plain logistic
   regression won out over everything else tried, including gradient
   boosting, a neural net, stacking, and proper ordinal logistic regression.
8. ~~**Bookmaker odds as a feature**~~ Done — see `notebooks/07_market_odds.ipynb`,
   `src/odds.py`, and "Bookmaker odds" above. +2 points of accuracy on both
   tours, a genuinely non-noise win; saved as a second production model and
   wired into `predict.predict_matchup` as an optional input.
9. ~~**Live in-match win probability, with a dataset to train on**: research
   the data landscape, build a Markov-chain analytical model and a learned
   model, evaluate as a curve over match progress rather than one
   number.~~ Done — see `notebooks/06_live_win_probability.ipynb`,
   `src/inmatch.py`, and the results above. Crosses 90% accuracy in the
   final tenth of a match, for both tours.
10. **Stretch**: games-differential regression.
11. ~~**Data viz**: a bespoke visualization now that there are real model
   outputs (Elo trajectories, win/margin probabilities) to visualize.~~
   Done — see `scripts/export_viz_data.py` (exports
   `data/processed/viz/*.json` from the existing parquet outputs, no new
   modeling) and
   [Baseline Studies](https://claude.ai/code/artifact/702398f9-b979-4f00-8a49-f12a42cfd83c):
   a radial Elo-trajectory chart (Djokovic/Swiatek career rings), a
   radial single-elimination bracket for the 2025 Roland Garros men's draw
   (wedge color flags Elo-model upsets), and a mirrored Sinner-Alcaraz
   head-to-head timeline with a current-form comparison.
