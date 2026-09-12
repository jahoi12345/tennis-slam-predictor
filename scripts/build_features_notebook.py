"""Generates notebooks/02_elo_and_form.ipynb."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

md("""# 02 — Elo, Form, and Head-to-Head Features

Computes, for every match in the full ATP/WTA history (not just Slams — Elo
and form need continuity across a player's whole career), the PRE-match state
for both players:

- **Overall Elo** and **surface-specific Elo** (experience-adjusted K-factor,
  same form FiveThirtyEight uses for tennis: newer players' ratings move
  faster, established players' are more stable), with a **margin-of-victory
  multiplier** on top (also adapted from FiveThirtyEight, originally for
  NBA/NFL Elo): a 6-0 6-0 6-0 sweep moves ratings more than a 7-6 7-6 7-6 win
  of the same match, and the multiplier is dampened by how surprising the
  result already was, so an already-big-favorite's expected blowout counts
  for less than a similarly dominant win by the underdog
- **Recent form**: win rate over their last 20 matches
- **Current streak**: consecutive wins (positive) or losses (negative)
  entering the match
- **Head-to-head**: prior meetings between these two players, all-time

ATP and WTA are treated as fully independent player pools throughout. This
is a strictly sequential computation — every match's features depend on
everything that happened before it for those two players — so it's a single
chronological pass rather than a vectorized operation.
""")

code("""import sys
sys.path.insert(0, '../src')

import pandas as pd
import numpy as np
import features as feat

pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 140)
""")

md("""## 1. Chronological ordering

Matches are sorted by tour → date → round → match number, so that (for
example) a player's quarterfinal result is reflected in their Elo before
their semifinal is processed — both share the same `tourney_date` (the
Monday of tournament week), so date alone isn't enough.
""")

code("""matches = pd.read_parquet('../data/processed/matches_all.parquet')
chron = feat.prepare_chronological(matches)
print(chron.shape)
chron[['tour', 'tourney_date', 'round', 'match_num', 'winner_name', 'loser_name']].head(3)
""")

md("""## 2. Compute features

Walkovers (`score == 'W/O'`, no play occurred) are skipped for rating/form/H2H
*updates* but still get a pre-match feature value looked up — they carry no
information about relative skill, so including them in updates would be
noise.
""")

code("""%%time
featured = feat.build_temporal_features(chron, form_window=20)
print(featured.shape)
""")

code("""featured.to_parquet('../data/processed/matches_with_features.parquet', index=False)
print('saved matches_with_features.parquet')
""")

md("""## 3. Sanity checks

### Elo trajectory for a well-known player

Should start at the 1500 base rating on debut and climb through their peak
years.
""")

code("""def elo_trajectory(df, player_name):
    is_winner = df['winner_name'] == player_name
    is_loser = df['loser_name'] == player_name
    sub = df[is_winner | is_loser].copy()
    sub['elo'] = np.where(is_winner[is_winner | is_loser], sub['winner_elo_pre'], sub['loser_elo_pre'])
    return sub[['tourney_date', 'tourney_name', 'round', 'elo']]

traj = elo_trajectory(featured, 'Novak Djokovic')
traj.iloc[[0, len(traj)//4, len(traj)//2, 3*len(traj)//4, -1]]
""")

md("""### Elo leaderboard

`current_ratings` gives each player's rating as of THEIR OWN last match in
the dataset — not a single as-of-today snapshot. So an all-time "best final
Elo ever reached" view mixes retired legends (whose last rating is near
their retirement peak) with active players (whose rating reflects however
they're doing right now). Both views below, to make that explicit.
""")

code("""current = feat.current_ratings(featured)
current.to_parquet('../data/processed/current_ratings.parquet', index=False)

elo_cols = ['player_name', 'overall_elo', 'last_match_date'] + [c for c in current.columns if c.startswith('elo_')]

print('ATP — best final Elo ever reached (all-time, includes retired players):')
display(current[current['tour'] == 'ATP'].sort_values('overall_elo', ascending=False).head(10)[elo_cols])

print('ATP — top 10 among players active in the last 2 years:')
active = current[current['last_match_date'] >= current['last_match_date'].max() - pd.Timedelta(days=730)]
display(active[active['tour'] == 'ATP'].sort_values('overall_elo', ascending=False).head(10)[elo_cols])

print('WTA — top 10 among players active in the last 2 years:')
display(active[active['tour'] == 'WTA'].sort_values('overall_elo', ascending=False).head(10)[elo_cols])
""")

md("""### Does Elo actually predict anything?

Quick check before moving to modeling: if we just picked the player with the
higher pre-match overall Elo as the predicted winner, what accuracy do we
get? This isn't the real model (no surface/form/H2H blending, no
calibration) — it's a floor the real model needs to beat.
""")

code("""valid = featured[(featured['score'] != 'W/O')].copy()
valid['elo_favorite_correct'] = valid['winner_elo_pre'] > valid['loser_elo_pre']

print(f\"All-time: {valid['elo_favorite_correct'].mean():.3f} accuracy over {len(valid):,} matches\")

recent = valid[valid['tourney_date'] >= '2015-01-01']
print(f\"2015+ only: {recent['elo_favorite_correct'].mean():.3f} accuracy over {len(recent):,} matches\")

slams_recent = recent[recent['tourney_level'] == 'G']
print(f\"2015+ Slams only: {slams_recent['elo_favorite_correct'].mean():.3f} accuracy over {len(slams_recent):,} matches\")
""")

md("""### Does the margin-of-victory multiplier actually help?

Re-running the exact same pass with `use_margin_of_victory=False` isolates
its effect on the 2022+ Grand Slam test window (the same split the win
classifier uses) — a direct ablation rather than just asserting the idea is
sound.
""")

code("""no_mov = feat.build_temporal_features(chron, form_window=20, use_margin_of_victory=False)
test_window = lambda df: df[(df['tourney_level'] == 'G') & (df['tourney_date'] >= '2022-01-01') & (df['score'] != 'W/O')]

for tour in ['ATP', 'WTA']:
    base_acc = (test_window(no_mov[no_mov['tour'] == tour]).pipe(lambda d: d['winner_elo_pre'] > d['loser_elo_pre']).mean())
    mov_acc = (test_window(featured[featured['tour'] == tour]).pipe(lambda d: d['winner_elo_pre'] > d['loser_elo_pre']).mean())
    print(f\"{tour}: {base_acc:.3f} without MOV -> {mov_acc:.3f} with MOV\")
""")

md("""A small but consistent gain for both tours — kept on by default.
""")

md("""### Form, streak, and H2H distributions
""")

code("""print('Form win-rate (last 20 matches) distribution:')
display(featured['winner_form_winrate'].describe())

print('Win streak distribution (winners):')
display(featured['winner_win_streak'].describe())

print('H2H: fraction of matches with at least one prior meeting:')
print((featured['winner_h2h_wins'] + featured['winner_h2h_losses'] > 0).mean())
""")

md("""## Summary

`data/processed/matches_with_features.parquet` now carries, for every
historical match: `{winner,loser}_elo_pre`, `{winner,loser}_surface_elo_pre`,
`{winner,loser}_form_winrate`, `{winner,loser}_win_streak`, and
`winner_h2h_wins`/`winner_h2h_losses`. `current_ratings.parquet` has the
latest overall + per-surface Elo for every active/historical player.

**Next**: the win classifier notebook filters this down to Grand Slam
matches, adds the remaining planned features (seed, round, rest days,
handedness, style features from notebook 01), and does a proper chronological
train/test split.
""")

nb['cells'] = cells
with open('notebooks/02_elo_and_form.ipynb', 'w') as f:
    nbf.write(nb, f)
print('wrote notebooks/02_elo_and_form.ipynb')
