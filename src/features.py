"""Elo, recent-form, and head-to-head features computed via a single
chronological pass over match history. See notebooks/02_elo_and_form.ipynb.

All three feature groups are inherently sequential (each match's features
depend on everything that happened before it for those two players), so this
is a plain Python loop rather than a vectorized pandas operation - at ~360k
matches this still runs in a few seconds.
"""
from collections import deque

import numpy as np
import pandas as pd

import dataset as ds

BASE_RATING = 1500.0

# Round order within a single tourney_date, so e.g. a QF's Elo update is
# applied before the SF that depends on it (both share the same
# tourney_date, which is the Monday of the tournament week).
ROUND_ORDER = {
    "ER": 0, "RR": 1, "R128": 1, "R64": 2, "R32": 3, "R16": 4,
    "QF": 5, "SF": 6, "BR": 6, "F": 7,
}


def prepare_chronological(matches: pd.DataFrame) -> pd.DataFrame:
    """Sort matches into the order they need to be replayed in for a
    sequential Elo/form/H2H pass: by tour (ATP/WTA are independent player
    pools), then date, then round, then match_num as a final tiebreak."""
    df = matches.copy()
    df["surface"] = df["surface"].str.capitalize()
    df["_round_order"] = df["round"].map(ROUND_ORDER).fillna(3)
    df = df.sort_values(["tour", "tourney_date", "_round_order", "match_num"])
    return df.drop(columns=["_round_order"]).reset_index(drop=True)


def _elo_k(matches_played: int) -> float:
    """Experience-adjusted K-factor: new/inexperienced players' ratings move
    faster, established players' ratings are more stable. Same form used by
    FiveThirtyEight's tennis Elo model."""
    return 250.0 / (matches_played + 5) ** 0.4


def mov_multiplier(games_winner: int, games_loser: int, winner_elo_pre: float, loser_elo_pre: float) -> float:
    """Margin-of-victory multiplier on the Elo update, adapted from
    FiveThirtyEight's NBA/NFL Elo (log-scaled margin, dampened by how
    surprising the result already was). A 6-0 6-0 6-0 sweep should move
    ratings more than a 7-6 7-6 7-6 win of the same match, and an underdog
    winning big should count for more than a big favorite winning as
    expected. `elo_diff` here is signed from the WINNER's perspective:
    positive means the winner was already favored (blowout wins by an
    already-big-favorite get dampened); negative means the winner was the
    underdog (dominant upsets get amplified).
    """
    margin = games_winner - games_loser
    if margin <= 0:
        return 1.0
    elo_diff = winner_elo_pre - loser_elo_pre
    return np.log(margin + 1) * (2.2 / (elo_diff * 0.001 + 2.2))


def build_temporal_features(matches: pd.DataFrame, form_window: int = 20,
                             use_margin_of_victory: bool = True) -> pd.DataFrame:
    """Adds, for every match, the PRE-match state for both players:
    - overall_elo, surface_elo (Elo on that match's surface)
    - form_winrate_{form_window} (win rate over their last N matches)
    - win_streak (current streak entering the match: positive=winning, negative=losing)
    - h2h_wins / h2h_losses (this pair's prior meetings, all-time)

    Must be called on output of prepare_chronological (one tour at a time is
    handled internally - ATP and WTA player pools are kept separate).
    Walkovers ('W/O' score, no play occurred) are skipped for rating/form/H2H
    updates but still get pre-match feature values looked up.
    """
    df = matches.reset_index(drop=True)
    n = len(df)

    overall_elo, elo_count = {}, {}
    surf_elo, surf_count = {}, {}
    recent_results = {}  # player_id -> deque of 1/0, most recent last
    streak = {}  # player_id -> int, positive=win streak, negative=loss streak
    h2h = {}  # (min_id, max_id) -> {player_id: wins}

    out = {
        "winner_elo_pre": np.empty(n), "loser_elo_pre": np.empty(n),
        "winner_surface_elo_pre": np.empty(n), "loser_surface_elo_pre": np.empty(n),
        "winner_form_winrate": np.full(n, np.nan), "loser_form_winrate": np.full(n, np.nan),
        "winner_win_streak": np.zeros(n, dtype=int), "loser_win_streak": np.zeros(n, dtype=int),
        "winner_h2h_wins": np.zeros(n, dtype=int), "winner_h2h_losses": np.zeros(n, dtype=int),
    }

    current_tour = None
    for i, row in enumerate(df.itertuples(index=False)):
        if row.tour != current_tour:
            # ATP and WTA are independent player pools - reset all state
            overall_elo, elo_count = {}, {}
            surf_elo, surf_count = {}, {}
            recent_results, streak, h2h = {}, {}, {}
            current_tour = row.tour

        w, l, surface = row.winner_id, row.loser_id, row.surface

        rw = overall_elo.get(w, BASE_RATING)
        rl = overall_elo.get(l, BASE_RATING)
        out["winner_elo_pre"][i] = rw
        out["loser_elo_pre"][i] = rl

        rsw = surf_elo.get((w, surface), BASE_RATING)
        rsl = surf_elo.get((l, surface), BASE_RATING)
        out["winner_surface_elo_pre"][i] = rsw
        out["loser_surface_elo_pre"][i] = rsl

        w_hist = recent_results.get(w, deque(maxlen=form_window))
        l_hist = recent_results.get(l, deque(maxlen=form_window))
        out["winner_form_winrate"][i] = np.mean(w_hist) if w_hist else np.nan
        out["loser_form_winrate"][i] = np.mean(l_hist) if l_hist else np.nan

        out["winner_win_streak"][i] = streak.get(w, 0)
        out["loser_win_streak"][i] = streak.get(l, 0)

        pair_key = (min(w, l), max(w, l))
        pair = h2h.get(pair_key, {})
        out["winner_h2h_wins"][i] = pair.get(w, 0)
        out["winner_h2h_losses"][i] = pair.get(l, 0)

        if row.score == "W/O":
            continue  # no play occurred - don't update any rating/form/H2H state

        mult = 1.0
        if use_margin_of_victory:
            gw, gl = ds.parse_game_totals(row.score)
            if gw is not None and gw > gl:
                mult = mov_multiplier(gw, gl, rw, rl)

        # --- Elo update (overall) ---
        exp_w = 1.0 / (1.0 + 10 ** ((rl - rw) / 400))
        mw, ml = elo_count.get(w, 0), elo_count.get(l, 0)
        overall_elo[w] = rw + _elo_k(mw) * mult * (1 - exp_w)
        overall_elo[l] = rl + _elo_k(ml) * mult * (exp_w - 1)
        elo_count[w], elo_count[l] = mw + 1, ml + 1

        # --- Elo update (surface-specific) ---
        if pd.notna(surface):
            exp_w_s = 1.0 / (1.0 + 10 ** ((rsl - rsw) / 400))
            msw, msl = surf_count.get((w, surface), 0), surf_count.get((l, surface), 0)
            surf_elo[(w, surface)] = rsw + _elo_k(msw) * mult * (1 - exp_w_s)
            surf_elo[(l, surface)] = rsl + _elo_k(msl) * mult * (exp_w_s - 1)
            surf_count[(w, surface)], surf_count[(l, surface)] = msw + 1, msl + 1

        # --- form update ---
        w_hist.append(1); l_hist.append(0)
        recent_results[w], recent_results[l] = w_hist, l_hist

        # --- streak update ---
        streak[w] = streak.get(w, 0) + 1 if streak.get(w, 0) >= 0 else 1
        streak[l] = streak.get(l, 0) - 1 if streak.get(l, 0) <= 0 else -1

        # --- H2H update ---
        pair[w] = pair.get(w, 0) + 1
        h2h[pair_key] = pair

    for col, values in out.items():
        df[col] = values
    return df


def add_rest_days(features_df: pd.DataFrame) -> pd.DataFrame:
    """Days since each player's previous tour-level match, added as
    winner_rest_days / loser_rest_days. Must be called on the FULL
    chronological match history (not pre-filtered to Slams), since a
    player's rest before a Slam is set by whatever they played right before
    it, Slam or not.

    Caveat: `tourney_date` is the Monday of tournament week, not the actual
    day of any given match - so this captures gaps BETWEEN tournaments (e.g.
    a long injury layoff, or jumping straight from clay to grass), not
    within-tournament fatigue (back-to-back rounds within the same event all
    share one date and so show 0 rest days by construction).
    """
    df = features_df.reset_index(drop=True)
    n = len(df)
    winner_rest = np.full(n, np.nan)
    loser_rest = np.full(n, np.nan)

    last_date = {}
    current_tour = None
    for i, row in enumerate(df.itertuples(index=False)):
        if row.tour != current_tour:
            last_date = {}
            current_tour = row.tour
        w, l, date = row.winner_id, row.loser_id, row.tourney_date
        if w in last_date:
            winner_rest[i] = (date - last_date[w]).days
        if l in last_date:
            loser_rest[i] = (date - last_date[l]).days
        if row.score != "W/O":
            last_date[w] = date
            last_date[l] = date

    df["winner_rest_days"] = winner_rest
    df["loser_rest_days"] = loser_rest
    return df


def current_ratings(features_df: pd.DataFrame) -> pd.DataFrame:
    """Final-career overall + per-surface Elo for every player, taken from
    the last match each appears in (as winner or loser).

    Note: this is each player's rating as of their OWN last match in the
    dataset, not a single as-of-today snapshot. A legend's final rating (near
    their retirement peak) sits alongside an active player's present-day
    rating with no adjustment - so this ranks "best final Elo ever reached",
    not "best right now". Filter on `last_match_date` for a genuinely
    current leaderboard (e.g. players active in the last 2 years).
    """
    w = features_df[["tour", "tourney_date", "winner_id", "winner_name", "surface",
                      "winner_elo_pre", "winner_surface_elo_pre"]].rename(columns={
        "winner_id": "player_id", "winner_name": "player_name",
        "winner_elo_pre": "overall_elo", "winner_surface_elo_pre": "surface_elo",
    })
    l = features_df[["tour", "tourney_date", "loser_id", "loser_name", "surface",
                      "loser_elo_pre", "loser_surface_elo_pre"]].rename(columns={
        "loser_id": "player_id", "loser_name": "player_name",
        "loser_elo_pre": "overall_elo", "loser_surface_elo_pre": "surface_elo",
    })
    long = pd.concat([w, l], ignore_index=True)
    long = long.sort_values("tourney_date")
    latest_overall = long.drop_duplicates(subset=["tour", "player_id"], keep="last")
    latest_overall = latest_overall.rename(columns={"tourney_date": "last_match_date"})
    latest_overall = latest_overall[["tour", "player_id", "player_name", "overall_elo", "last_match_date"]]

    latest_surface = long.drop_duplicates(subset=["tour", "player_id", "surface"], keep="last")
    surf_pivot = latest_surface.pivot_table(
        index=["tour", "player_id"], columns="surface", values="surface_elo", aggfunc="last"
    ).add_prefix("elo_").add_suffix("").reset_index()

    return latest_overall.merge(surf_pivot, on=["tour", "player_id"], how="left")
