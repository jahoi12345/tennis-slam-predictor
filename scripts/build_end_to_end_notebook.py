"""Generates notebooks/05_end_to_end_prediction.ipynb."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

md("""# 05 — End-to-End Prediction

Combines the win classifier (03) and margin classifier (04) into a single
"given two players, who wins and by how much" answer.

**The subtlety this notebook exists to handle**: the margin model was
trained on WINNER-perspective features (in hindsight, "the winner won 3-1"
has no ambiguity), but at prediction time we don't know who wins yet. Just
calling both models independently and gluing the outputs together would
silently score the margin model on the wrong-oriented features half the
time. The correct combination is a mixture:

```
P(margin = m) = P(player1 wins) * P(m | assume player1 wins)
              + P(player2 wins) * P(m | assume player2 wins)
```

`src/predict.py` computes both conditional margin distributions (feeding
each player's own win probability into the margin model as its
`win_probability` feature) and mixes them with the win probabilities as
weights. See `predict_matchup` for the full implementation.
""")

code("""import sys
sys.path.insert(0, '../src')

import pandas as pd
import joblib

import features as feat
import predict as pr

pd.set_option('display.max_columns', 30)
""")

md("## 1. Load everything the predictor needs")

code("""matches = pd.read_parquet('../data/processed/matches_with_features.parquet')
matches = feat.add_rest_days(matches)
current_ratings = pd.read_parquet('../data/processed/current_ratings.parquet')

style_atp = pd.read_parquet('../data/processed/style_features_atp.parquet')
style_wta = pd.read_parquet('../data/processed/style_features_wta.parquet')

win_model_atp = joblib.load('../models/win_classifier_atp_ensemble.joblib')
win_model_wta = joblib.load('../models/win_classifier_wta_ensemble.joblib')
win_model_atp_odds = joblib.load('../models/win_classifier_atp_ensemble_with_odds.joblib')
win_model_wta_odds = joblib.load('../models/win_classifier_wta_ensemble_with_odds.joblib')
margin_model_atp = joblib.load('../models/margin_classifier_atp_lr_analytical_features.joblib')
margin_model_wta = joblib.load('../models/margin_classifier_wta_lr_analytical_features.joblib')

print('loaded')
""")

md("""## 2. A readable wrapper

Formats `predict_matchup`'s output as a short readable summary instead of a
raw dict of probabilities.
""")

code("""def show_prediction(tour, player1, player2, surface, round_, match_date, p1_odds=None, p2_odds=None, **kwargs):
    if tour == 'ATP':
        matches_, ratings_, style_, win_model, win_model_odds, margin_model, best_of = (
            matches, current_ratings, style_atp, win_model_atp, win_model_atp_odds, margin_model_atp, 5)
    else:
        matches_, ratings_, style_, win_model, win_model_odds, margin_model, best_of = (
            matches, current_ratings, style_wta, win_model_wta, win_model_wta_odds, margin_model_wta, 3)

    if p1_odds is not None and p2_odds is not None:
        kwargs.update(p1_odds=p1_odds, p2_odds=p2_odds, win_model_with_odds=win_model_odds)

    result = pr.predict_matchup(
        matches_, ratings_, style_, win_model, margin_model,
        player1=player1, player2=player2, tour=tour, surface=surface,
        round_=round_, best_of=best_of, match_date=match_date, **kwargs,
    )

    print(f\"{result['player1']} vs {result['player2']}  ({tour}, {surface}, {round_})\")
    print(f\"  {result['player1']}: {result['player1_win_prob']:.0%} to win\")
    print(f\"  {result['player2']}: {result['player2_win_prob']:.0%} to win\")
    print(f\"  Overall margin distribution:\")
    for m, p in sorted(result['overall_margin_distribution'].items()):
        print(f\"    {m}: {p:.0%}\")
    print(f\"  If {result['player1']} wins, likely: \" +
          ', '.join(f'{m} ({p:.0%})' for m, p in sorted(result['margin_if_player1_wins'].items(), key=lambda x: -x[1])))
    print(f\"  If {result['player2']} wins, likely: \" +
          ', '.join(f'{m} ({p:.0%})' for m, p in sorted(result['margin_if_player2_wins'].items(), key=lambda x: -x[1])))
    return result
""")

md("""## 3. Example matchups

A close top-of-the-game final, and a clearer favorite/underdog pairing, for
both tours.
""")

code("""_ = show_prediction('ATP', 'Jannik Sinner', 'Carlos Alcaraz', surface='Clay', round_='F', match_date='2026-06-01')
""")

code("""_ = show_prediction('ATP', 'Jannik Sinner', 'Arthur Fils', surface='Hard', round_='QF', match_date='2026-06-01')
""")

code("""_ = show_prediction('WTA', 'Aryna Sabalenka', 'Iga Swiatek', surface='Hard', round_='F', match_date='2026-06-01')
""")

md("""## 4. With bookmaker odds

If you have real odds for an upcoming match in hand, `predict_matchup`
takes them as an optional `p1_odds`/`p2_odds` pair (decimal odds) and
switches to notebook 07's odds-enhanced production model, which folds in
the devigged market-implied probability alongside everything else. Without
odds, nothing changes from section 3 above - this is purely additive.
""")

code("""_ = show_prediction('ATP', 'Jannik Sinner', 'Carlos Alcaraz', surface='Clay', round_='F',
                     match_date='2026-06-01', p1_odds=1.60, p2_odds=2.40)
""")

md("""## 5. Sanity check against a real historical result

Roland Garros 2025 men's final: Alcaraz beat Sinner in 5 sets (an epic
comeback from two sets down). Predicting it as of the day before the final
(using only information available at that point) should show Sinner as at
least a moderate favorite, given his ranking and recent form entering that
final, with a meaningfully non-trivial chance for a long/close match — not
a claim the model "called" the exact outcome, just a check that the
prediction is a reasonable pre-match view.
""")

code("""_ = show_prediction('ATP', 'Jannik Sinner', 'Carlos Alcaraz', surface='Clay', round_='F', match_date='2025-06-08')
""")

md("""## Summary

The two-stage pipeline is now a single callable: `predict.predict_matchup`
takes two player names, a surface/round/tour, and a date, and returns a full
win + conditional-margin + overall-margin distribution — pulling live Elo,
form, streak, H2H, and style-feature state directly from the processed
match history rather than requiring those to be looked up by hand. It
optionally takes bookmaker odds too, switching to the odds-enhanced
production model from notebook 07 when they're supplied.

**What's left on the original plan**: the data visualization piece (a
bespoke chart in the Visual Cinnamon style — radial bracket, Elo trajectory,
or head-to-head comparison), which was deliberately deferred until there
were real predictions and Elo histories to visualize. There now are.
""")

nb['cells'] = cells
with open('notebooks/05_end_to_end_prediction.ipynb', 'w') as f:
    nbf.write(nb, f)
print('wrote notebooks/05_end_to_end_prediction.ipynb')
