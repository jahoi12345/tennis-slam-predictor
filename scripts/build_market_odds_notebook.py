"""Generates notebooks/07_market_odds.ipynb."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

md("""# 07 — Bookmaker Odds

Every source in the earlier literature review (academic papers, Jeff
Sackmann's own analysis, the Oct-2025 GNN paper) agreed on one thing: market
odds are the single most reliable predictive signal available for tennis,
consistently landing around 75-77% accuracy versus our own ~71-72%. This
notebook brings that signal in as a feature.

Source: [tennis-data.co.uk](http://www.tennis-data.co.uk) — ATP 2000-2026,
WTA 2007-2026, one season file per tour per year. The real engineering work
here isn't the download, it's the join: this source identifies players as
`"Djokovic N."` (surname + initial), while the rest of this project uses
full names (`"Novak Djokovic"`) — see `src/odds.py`.
""")

code("""import sys
sys.path.insert(0, '../src')

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

import features as feat
import dataset as ds
import odds

pd.set_option('display.max_columns', 30)
""")

md("""## 1. Load odds and join to the Sackmann match history

The name join turned out messier than expected once checked against real
data: the *same player* is formatted inconsistently across different years
of this source (`"Del Potro J.M."`, `"Del Potro J. M."`, and just
`"Del Potro J."` all appear for Juan Martín del Potro). `src/odds.py`
handles this by matching on surname + **first initial only** (discarding
any middle initials, which aren't reliable), extracted via regex rather
than assumed to follow one fixed format, then disambiguating same-surname
collisions within a date window using tournament-name similarity.
""")

code("""%%time
matches = pd.read_parquet('../data/processed/matches_with_features.parquet')
matches = feat.add_rest_days(matches)

atp_odds = odds.load_odds_seasons('atp')
wta_odds = odds.load_odds_seasons('wta')
print(f'ATP odds: {len(atp_odds):,} rows, {atp_odds[\"Date\"].min().date()} to {atp_odds[\"Date\"].max().date()}')
print(f'WTA odds: {len(wta_odds):,} rows, {wta_odds[\"Date\"].min().date()} to {wta_odds[\"Date\"].max().date()}')

matches = odds.attach_odds(matches, atp_odds, 'ATP')
matches = odds.attach_odds(matches, wta_odds, 'WTA')
""")

code("""for tour, start_year in [('ATP', 2000), ('WTA', 2007)]:
    sub = matches[(matches['tour'] == tour) & (matches['tourney_date'].dt.year >= start_year)]
    print(f'{tour} overall match rate since {start_year}: {sub[\"winner_implied_prob\"].notna().mean():.1%} of {len(sub):,}')
    slam = matches[(matches['tour'] == tour) & (matches['tourney_level'] == 'G') & (matches['tourney_date'] >= '2022-01-01')]
    print(f'{tour} Slam 2022+ match rate: {slam[\"winner_implied_prob\"].notna().mean():.1%} of {len(slam):,} (our actual eval window)')
""")

md("""Overall match rate is lower than the Slam-specific rate (tennis-data.co.uk
has gaps for lower-tier/older events this project doesn't otherwise need),
but for the window that actually matters — Grand Slam matches in the 2022+
test period — coverage is ~89-90% for both tours.

Spot-check a handful of matched rows before trusting this further: do the
implied probabilities look sane relative to known Elo gaps?
""")

code("""sample_check = matches[
    (matches['tour'] == 'ATP') & (matches['tourney_level'] == 'G')
    & (matches['tourney_date'] >= '2024-01-01') & matches['winner_implied_prob'].notna()
].sample(6, random_state=3)
display(sample_check[['tourney_name', 'winner_name', 'loser_name', 'winner_elo_pre', 'loser_elo_pre', 'winner_implied_prob']])

agree = ((sample_check['winner_elo_pre'] > sample_check['loser_elo_pre']) == (sample_check['winner_implied_prob'] > 0.5))
print('Elo favorite matches market favorite on this sample:', agree.mean())
""")

md("""## 2. A fair comparison

Odds only exist from 2000 (ATP) / 2007 (WTA) — training the baseline "our
features alone" model on the full 1968-2026 history (like notebook 03 does)
and comparing it to an odds-only model trained on 2000+ would be comparing
different training windows, not different feature sets. All three configs
below are trained on the **same** odds-available window, tested on the same
2022+ split as every other notebook.
""")

code("""style_atp = pd.read_parquet('../data/processed/style_features_atp.parquet')
style_wta = pd.read_parquet('../data/processed/style_features_wta.parquet')

atp = ds.build_matchup_dataset(matches, style_atp, 'ATP')
wta = ds.build_matchup_dataset(matches, style_wta, 'WTA')

CUTOFF = '2022-01-01'
RESTRICT_START = {'ATP': '2001-01-01', 'WTA': '2008-01-01'}  # a year past odds availability, avoids the sparsest first year

NUMERIC_DIFFS = ['diff_elo', 'diff_surface_elo', 'diff_blended_elo', 'diff_form', 'diff_streak',
                  'diff_rest_days', 'diff_seed', 'diff_h2h'] + [f'diff_{c}' for c in ds.STYLE_COLS]
CATEGORICAL = ['surface', 'round']
PASSTHROUGH = ['best_of', 'opposite_handed']

def make_lr(numeric_cols):
    prep = ColumnTransformer([
        ('num', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL),
        ('pass', 'passthrough', PASSTHROUGH),
    ])
    return Pipeline([('prep', prep), ('clf', LogisticRegression(max_iter=1000))])

fair_results = []
for tour, df in [('ATP', atp), ('WTA', wta)]:
    train = df[(df['tourney_date'] >= RESTRICT_START[tour]) & (df['tourney_date'] < CUTOFF)]
    test = df[df['tourney_date'] >= CUTOFF]

    configs = {'Our features alone': NUMERIC_DIFFS, 'Market prob alone': ['diff_market_prob'],
               'Combined': NUMERIC_DIFFS + ['diff_market_prob']}
    for name, cols in configs.items():
        feature_cols = cols + CATEGORICAL + PASSTHROUGH
        pipe = make_lr(cols)
        pipe.fit(train[feature_cols], train['target'])
        proba = pipe.predict_proba(test[feature_cols])[:, 1]
        pred = (proba >= 0.5).astype(int)
        fair_results.append({'tour': tour, 'model': name,
                              'accuracy': accuracy_score(test['target'], pred),
                              'auc': roc_auc_score(test['target'], proba)})

    market_test = test[test['diff_market_prob'].notna()]
    naive_pred = (market_test['diff_market_prob'] > 0).astype(int)
    fair_results.append({'tour': tour, 'model': 'Naive market-favorite',
                          'accuracy': accuracy_score(market_test['target'], naive_pred), 'auc': np.nan})

fair_results_df = pd.DataFrame(fair_results)
fair_results_df
""")

md("""**This is a clear, non-noise win** — a meaningfully larger gap than
anything found chasing model architecture in notebook 03's round 2 (those
were within-noise, fractions of a point; this is +2 points of accuracy on
both tours from one added feature). Combining closes most of the distance
to the naive market-favorite floor, and beats "market alone" on AUC for
both tours — our engineered features add real information the market
price doesn't fully capture on its own, and vice versa.

**Framing this honestly**: this isn't "our model beats the market" — market
odds are still doing most of the work in the *combined* row. It's "our
features + the market's information together beat either alone," which is
exactly what you'd want from adding a genuinely independent, strong signal.
""")

md("""## 3. Build the production-quality odds-enhanced model

Same architecture as the actual production win classifier (notebook 03's
interaction-aware logistic regression stacked with tuned gradient
boosting), with `diff_market_prob` added to both base learners, trained on
the odds-available window. This becomes a *second* production artifact
(`_ensemble_with_odds.joblib`) alongside the existing odds-free one — real
future matches won't always have odds on hand, so `predict.py` needs both
and picks based on whether the caller supplies odds.
""")

code("""%%time
atp_i = ds.add_elo_interactions(atp)
wta_i = ds.add_elo_interactions(wta)

INTERACTION_COLS_WITH_ODDS = NUMERIC_DIFFS + ['diff_market_prob'] + ds.INTERACTION_COLS

odds_models = {}
odds_results = []
for tour, df_i in [('ATP', atp_i), ('WTA', wta_i)]:
    train = df_i[(df_i['tourney_date'] >= RESTRICT_START[tour]) & (df_i['tourney_date'] < CUTOFF)]
    test = df_i[df_i['tourney_date'] >= CUTOFF]
    cols = INTERACTION_COLS_WITH_ODDS + CATEGORICAL + PASSTHROUGH

    lr_prep = ColumnTransformer([
        ('num', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]),
         NUMERIC_DIFFS + ['diff_market_prob'] + ds.INTERACTION_COLS),
        ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL),
        ('pass', 'passthrough', PASSTHROUGH),
    ])
    hgb_prep = ColumnTransformer([
        ('num', 'passthrough', NUMERIC_DIFFS + ['diff_market_prob'] + ds.INTERACTION_COLS),
        ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL),
        ('pass', 'passthrough', PASSTHROUGH),
    ])
    stack = StackingClassifier(
        estimators=[('lr', Pipeline([('prep', lr_prep), ('clf', LogisticRegression(max_iter=1000))])),
                    ('hgb', Pipeline([('prep', hgb_prep), ('clf', HistGradientBoostingClassifier(random_state=0))]))],
        final_estimator=LogisticRegression(max_iter=1000), cv=5,
    )
    stack.fit(train[cols], train['target'])
    proba = stack.predict_proba(test[cols])[:, 1]
    pred = (proba >= 0.5).astype(int)
    odds_results.append({'tour': tour, 'model': 'Stacking + market odds (production)',
                          'accuracy': accuracy_score(test['target'], pred), 'auc': roc_auc_score(test['target'], proba)})
    odds_models[tour] = stack

pd.DataFrame(odds_results)
""")

md("""## 4. Save

Saved alongside (not over) the existing odds-free production model.
""")

code("""import os
os.makedirs('../models', exist_ok=True)
for tour, model in odds_models.items():
    joblib.dump(model, f'../models/win_classifier_{tour.lower()}_ensemble_with_odds.joblib')

fair_results_df.to_csv('../data/processed/market_odds_results.csv', index=False)
print('saved models/win_classifier_{atp,wta}_ensemble_with_odds.joblib')
""")

md("""## Summary

Bookmaker odds are the real, non-noise improvement this project's earlier
research promised: +2 points of accuracy on both tours from a single added
feature, closing most of the gap to the professional betting market's own
~74-75% floor. The odds-enhanced model is saved as a second production
artifact — `predict.py` now accepts an optional market odds input and uses
this model when it's supplied, falling back to the original odds-free
ensemble otherwise (real future matches don't always come with odds
attached).
""")

nb['cells'] = cells
with open('notebooks/07_market_odds.ipynb', 'w') as f:
    nbf.write(nb, f)
print('wrote notebooks/07_market_odds.ipynb')
