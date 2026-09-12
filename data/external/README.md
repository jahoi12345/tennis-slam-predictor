# External data sources

This directory holds raw clones of the source data repos and is gitignored
(not committed — see project root `.gitignore`). To (re)fetch:

```bash
cd data/external
git clone --depth 1 https://github.com/Aneeshers/tennis-sackmann-archive.git
git clone --depth 1 https://github.com/JeffSackmann/tennis_MatchChartingProject.git
```

- **`tennis-sackmann-archive/`** — mirror of `JeffSackmann/tennis_atp`,
  `tennis_wta`, and `tennis_slam_pointbypoint` (the original repos are
  currently unreachable directly; see the mirror's own README for
  provenance). Contains `atp/`, `wta/` (match history, players, rankings)
  and `slam_pointbypoint/` (point-by-point data for the four majors,
  2011–2024).
- **`tennis_MatchChartingProject/`** — Jeff Sackmann's volunteer-charted
  shot-by-shot data (still live upstream). Used here for per-player playing
  style (forehand/backhand reliance, rally-length preference, net game).
- **`tennis-data-co-uk/{atp,wta}/`** — bookmaker odds, one Excel file per
  tour per season, downloaded by hand from
  [tennis-data.co.uk](http://www.tennis-data.co.uk/alldata.php) (its own
  HTTPS is broken - use plain `http://`, and if `www.` doesn't resolve, try
  the bare domain `http://tennis-data.co.uk` instead, which has stayed up
  when `www.` hasn't). ATP: `http://tennis-data.co.uk/{year}/{year}.xlsx`
  for 2000-2026; WTA: `http://tennis-data.co.uk/{year}w/{year}.xlsx` for
  2007-2026. See `src/odds.py` for the ingestion and name-matching join.

**License**: the Sackmann/Tennis Abstract sources above (everything except
`tennis-data-co-uk/`) are **CC BY-NC-SA 4.0** — attribution required,
non-commercial use only; see each repo's own `LICENSE`/`UPSTREAM_README.md`.
tennis-data.co.uk is a separately-licensed free public dataset with its own
terms (see the site) — not part of that CC BY-NC-SA grant.
