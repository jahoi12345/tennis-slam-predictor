"""Bookmaker odds ingestion from tennis-data.co.uk and the name-matching
join against the Sackmann-based match history. See
notebooks/07_market_odds.ipynb.

tennis-data.co.uk uses "Lastname F." (e.g. "Djokovic N."), while the rest of
this project uses full names ("Novak Djokovic") - this module's real job is
bridging that gap, not just reading the files.
"""
import glob
import os
import re

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

ODDS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "external", "tennis-data-co-uk")

# A handful of known compound surnames that would otherwise be mis-split by
# taking just the last whitespace-separated token - covers the players most
# likely to actually appear in this data (extend as needed; anyone missed
# just fails to join, the same graceful-degradation as everywhere else in
# this project's name matching).
COMPOUND_SURNAMES = [
    "del potro", "bautista agut", "auger aliassime", "ramos vinolas", "carreno busta",
    "garcia lopez", "granollers pujol", "van de zandschulp", "davidovich fokina",
    "munar clar", "struff jan", "de minaur",
]

_TRAILING_INITIALS_RE = re.compile(r"((?:[A-Z]\.\s*)+)$")


def surname_and_initial_from_full_name(full_name: str) -> tuple[str, str]:
    """'Novak Djokovic' -> ('djokovic', 'n'). Sackmann-style full names:
    surname is everything but the first token, unless it matches a known
    compound surname. Lowercased for case-insensitive matching."""
    name = full_name.replace("-", " ").strip()
    lower = name.lower()
    for compound in COMPOUND_SURNAMES:
        if lower.endswith(compound):
            first = name[: -len(compound)].strip()
            return compound, (first[0].lower() if first else "")
    parts = name.split()
    if len(parts) < 2:
        return lower, ""
    return " ".join(parts[1:]).lower(), parts[0][0].lower()


def surname_and_initial_from_odds_name(name: str) -> tuple[str, str]:
    """'Tsonga J.W.' -> ('tsonga', 'j'); 'Del Potro J. M.' -> ('del potro',
    'j'); 'Djokovic N. ' -> ('djokovic', 'n'). tennis-data.co.uk names are
    "Surname Initial(s)." - real data shows this is inconsistent for the
    same player across years (single vs. double initials, with/without
    spacing), so only the surname and the FIRST initial are kept; middle
    initials are discarded rather than relied on for matching.
    """
    name = name.strip().replace("-", " ")
    match = _TRAILING_INITIALS_RE.search(name)
    if not match:
        return name.lower(), ""
    surname = name[: match.start()].strip().lower()
    initial = match.group(1).strip()[0].lower()
    return surname, initial


def load_odds_seasons(tour: str) -> pd.DataFrame:
    """All tennis-data.co.uk season files for 'atp' or 'wta', concatenated.
    Column sets vary by year (older years have more/fewer bookmakers; 2000
    ATP has no odds at all) - concatenation just fills missing columns with
    NaN, which is the correct behavior here.
    """
    pattern = os.path.join(ODDS_DIR, tour, "*.xlsx")
    files = sorted(glob.glob(pattern)) + sorted(glob.glob(os.path.join(ODDS_DIR, tour, "*.xls")))
    frames = [pd.read_excel(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df["tour"] = tour.upper()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


def devig_implied_prob(odds_a, odds_b):
    """Devigged (overround-removed) implied win probability for the side
    priced at `odds_a`, given the opposing side is priced at `odds_b`.
    Decimal odds -> raw implied prob is 1/odds; the two raw probabilities
    sum to > 1 (the bookmaker's margin), so normalizing by their sum removes
    it, giving a fair probability estimate."""
    inv_a, inv_b = 1 / odds_a, 1 / odds_b
    return inv_a / (inv_a + inv_b)


def _best_odds_pair(row):
    """Prefer the market average (AvgW/AvgL) as the most robust consensus
    signal; fall back to Pinnacle (PSW/PSL, the single most efficient book
    per the sports-prediction literature) when Avg is missing, then Bet365
    as a last resort for the oldest years with sparser bookmaker coverage."""
    for w_col, l_col in [("AvgW", "AvgL"), ("PSW", "PSL"), ("B365W", "B365L")]:
        w, l = row.get(w_col), row.get(l_col)
        if pd.notna(w) and pd.notna(l):
            return w, l
    return np.nan, np.nan


def match_row_lookup(odds_df: pd.DataFrame, tour: str) -> dict:
    """frozenset({(surname, initial), (surname, initial)}) -> list of
    (date, row_index) for fast lookup when joining to Sackmann matches."""
    df = odds_df[odds_df["tour"] == tour]
    lookup: dict = {}
    for idx, winner, loser, date in zip(df.index, df["Winner"], df["Loser"], df["Date"]):
        if pd.isna(winner) or pd.isna(loser) or pd.isna(date):
            continue
        key = frozenset({surname_and_initial_from_odds_name(winner), surname_and_initial_from_odds_name(loser)})
        lookup.setdefault(key, []).append((date, idx))
    return lookup


def attach_odds(matches_features: pd.DataFrame, odds_df: pd.DataFrame, tour: str,
                 max_days_gap: int = 25, tournament_fuzz_threshold: int = 60) -> pd.DataFrame:
    """For each Sackmann match of `tour`, finds the matching tennis-data.co.uk
    row by (normalized surname+initial pair, nearest date within
    `max_days_gap`), disambiguating multiple candidates in that window by
    tournament-name similarity (rapidfuzz) rather than date alone - two
    players can meet more than once in a short stretch (e.g. back-to-back
    weeks), and date-only nearest-match would sometimes pick the wrong one.
    Returns matches_features with `winner_implied_prob`/`loser_implied_prob`
    columns added (NaN where no match was found, e.g. before 2000/2007 or
    any Sackmann match tennis-data.co.uk simply doesn't carry) - named to
    match the winner_X_pre/loser_X_pre convention already used throughout
    matches_with_features, so dataset.py's diff-feature builders can pick
    it up the same way as every other pre-match column.
    """
    lookup = match_row_lookup(odds_df, tour)
    df = matches_features[matches_features["tour"] == tour]

    prob_winner = pd.Series(np.nan, index=df.index)
    prob_loser = pd.Series(np.nan, index=df.index)

    for idx, winner, loser, date, tourney_name in zip(
        df.index, df["winner_name"], df["loser_name"], df["tourney_date"], df["tourney_name"]
    ):
        key = frozenset({surname_and_initial_from_full_name(winner), surname_and_initial_from_full_name(loser)})
        candidates = lookup.get(key, [])
        if not candidates:
            continue
        in_window = [(d, i) for d, i in candidates if abs((d - date).days) <= max_days_gap]
        if not in_window:
            continue
        if len(in_window) == 1:
            best_idx = in_window[0][1]
        else:
            best_idx = max(
                in_window,
                key=lambda di: fuzz.partial_ratio(str(tourney_name), str(odds_df.loc[di[1], "Tournament"])),
            )[1]
        row = odds_df.loc[best_idx]
        odds_w, odds_l = _best_odds_pair(row)
        if pd.isna(odds_w) or pd.isna(odds_l):
            continue
        p = devig_implied_prob(odds_w, odds_l)
        prob_winner.loc[idx] = p
        prob_loser.loc[idx] = 1 - p

    out = matches_features.copy()
    out.loc[df.index, "winner_implied_prob"] = prob_winner
    out.loc[df.index, "loser_implied_prob"] = prob_loser
    return out
