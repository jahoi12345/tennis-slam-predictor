"""Generates notebooks/01_data_ingestion.ipynb. Run this to (re)build the
notebook file from source; then execute it with jupyter to produce the
processed parquet files in data/processed/."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

md("""# 01 — Data Ingestion

Loads and consolidates the raw Sackmann ATP/WTA match history, ranking, and
player files, plus two "playing style" sources (Match Charting Project and
Grand Slam point-by-point data), into clean parquet files under
`data/processed/`.

**Data license**: all sources here are Jeff Sackmann / Tennis Abstract data,
released under CC BY-NC-SA 4.0 — attribution required, non-commercial use
only. See `data/external/*/UPSTREAM_README.md` for the original notices.
""")

code("""import sys
sys.path.insert(0, '../src')

import pandas as pd
import data_loading as dl

pd.set_option('display.max_columns', 30)
""")

md("## 1. Tour-level match history (ATP + WTA, 1968–2026)")

code("""atp_matches = dl.load_tour_matches('atp')
wta_matches = dl.load_tour_matches('wta')

matches = pd.concat([atp_matches, wta_matches], ignore_index=True)
print(f"ATP: {atp_matches.shape}, WTA: {wta_matches.shape}, combined: {matches.shape}")
matches[['tour', 'tourney_date']].groupby('tour').agg(['min', 'max'])
""")

code("""matches.to_parquet('../data/processed/matches_all.parquet', index=False)
print('saved matches_all.parquet:', matches.shape)
""")

md("""### Sanity checks

Row counts by year (both tours should show a roughly stable ~2,500–3,000
matches/year at tour level in recent decades) and a null check on the columns
the modeling notebooks will depend on most.
""")

code("""matches['year'] = matches['tourney_date'].dt.year
display(matches.groupby(['year', 'tour']).size().unstack().tail(10))

key_cols = ['winner_rank', 'loser_rank', 'surface', 'best_of', 'round', 'score']
print(matches[key_cols].isna().mean().rename('null_fraction'))
""")

md("""**Note on Slam filtering**: `tourney_level == 'G'` isolates Grand Slam
matches for the winner/margin modeling target. Full match history (all
levels) is kept here because Elo and form features need continuity across a
player's whole career, not just their Slam matches.
""")

code("""slam_matches = matches[matches['tourney_level'] == 'G']
print(slam_matches.shape)
slam_matches['tourney_name'].value_counts()
""")

md("## 2. Players and rankings")

code("""atp_players = dl.load_players('atp')
wta_players = dl.load_players('wta')
players = pd.concat([atp_players, wta_players], ignore_index=True)
players.to_parquet('../data/processed/players.parquet', index=False)
print(players.shape)
players.head(3)
""")

code("""atp_rankings = dl.load_rankings('atp')
wta_rankings = dl.load_rankings('wta')
rankings = pd.concat([atp_rankings, wta_rankings], ignore_index=True)
rankings.to_parquet('../data/processed/rankings.parquet', index=False)
print(rankings.shape)
rankings.head(3)
""")

md("""## 3. Playing-style features — Match Charting Project

Forehand/backhand reliance, rally-length preference, and net game, aggregated
per player across every charted match (ATP + WTA). Coverage is volunteer-
selected — mostly top players and marquee matches, not a full census — so
expect this to be missing for many lower-ranked players.
""")

code("""def build_style_table(gender):
    shot = dl.build_shot_style_features(gender)
    rally = dl.build_rally_style_features(gender)
    net = dl.build_net_game_features(gender)

    out = shot.merge(rally.drop(columns=['tour']), on='player', how='outer')
    out = out.merge(net.drop(columns=['tour']), on='player', how='outer')
    return out

style_atp = build_style_table('m')
style_wta = build_style_table('w')
print('ATP charted players:', style_atp.shape[0], '| WTA charted players:', style_wta.shape[0])
style_atp.head(3)
""")

md("""## 4. Playing-style features — Grand Slam point-by-point (serve speed, FH/BH winners)

Serve speed and forehand-vs-backhand winner tendency from the four majors'
own ball-tracking data (2011–2024). This is the only public source of actual
serve-speed numbers; there is no public groundstroke-speed data.

Caveats (see `data/external/tennis-sackmann-archive/slam_pointbypoint/UPSTREAM_README.md`):
column availability varies by slam/year across two different data providers,
the Australian Open and French Open stopped being scraped after 2022, and
player names are matched by string (not player_id), so a handful of players
with inconsistent name formatting across years will be undercounted.
""")

code("""slam_style = dl.build_slam_style_features()
print(slam_style.shape)
slam_style['tour'].value_counts(dropna=False)
""")

code("""slam_style_atp = slam_style[slam_style['tour'] == 'ATP'].drop(columns=['tour'])
slam_style_wta = slam_style[slam_style['tour'] == 'WTA'].drop(columns=['tour'])

style_atp_full = style_atp.merge(slam_style_atp, on='player', how='outer')
style_wta_full = style_wta.merge(slam_style_wta, on='player', how='outer')

style_atp_full.to_parquet('../data/processed/style_features_atp.parquet', index=False)
style_wta_full.to_parquet('../data/processed/style_features_wta.parquet', index=False)
print('ATP style rows:', style_atp_full.shape, '| WTA style rows:', style_wta_full.shape)
""")

md("""## 5. Ingestion summary
""")

code("""import os

print('data/processed/ contents:')
for f in sorted(os.listdir('../data/processed')):
    path = os.path.join('../data/processed', f)
    print(f'  {f:35s} {os.path.getsize(path) / 1e6:8.1f} MB')
""")

nb['cells'] = cells
with open('notebooks/01_data_ingestion.ipynb', 'w') as f:
    nbf.write(nb, f)
print('wrote notebooks/01_data_ingestion.ipynb')
