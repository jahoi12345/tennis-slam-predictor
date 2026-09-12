"""End-to-end prediction: given two players and a match context, combines
the win classifier (notebook 03) and margin classifier (notebook 04) into a
single "who wins, and by how much" answer.

The subtlety this module exists to handle: the margin model was trained on
WINNER-perspective features (there's no ambiguity in hindsight - "the winner
won 3-1"), but at prediction time we don't know who wins. So the margin
distribution has to be computed twice - once assuming player1 wins, once
assuming player2 wins - and combined with the win probability as mixture
weights: P(margin=m) = P(p1 wins)*P(m | p1 wins) + P(p2 wins)*P(m | p2 wins).
Gluing the two models together without this step would silently evaluate the
margin model on the wrong-oriented features half the time.
"""
import numpy as np
import pandas as pd

import dataset as ds
import odds as odds_mod

BASE_FEATURE_COLS = (
    ["diff_elo", "diff_surface_elo", "diff_blended_elo", "diff_form", "diff_streak", "diff_rest_days",
     "diff_seed", "diff_h2h"] + [f"diff_{c}" for c in ds.STYLE_COLS]
    + ["surface", "round", "best_of", "opposite_handed"]
)
# The win model (notebook 03's stacking ensemble) additionally needs the
# explicit Elo x surface/round interaction terms. The margin model (notebook
# 04's production model, "LR + analytical features") needs the win
# probability plus the analytical i.i.d.-sets margin probabilities derived
# from it - the column count differs by best_of (3 buckets for best-of-5,
# 2 for best-of-3), hence a function rather than a constant.
WIN_FEATURE_COLS = BASE_FEATURE_COLS + ds.INTERACTION_COLS
# Used only when the caller supplies both players' current odds - see
# predict_matchup's p1_odds/p2_odds args and notebook 07's odds-enhanced
# production model.
WIN_FEATURE_COLS_WITH_ODDS = WIN_FEATURE_COLS + ["diff_market_prob"]


def margin_feature_cols(best_of: int) -> list:
    return BASE_FEATURE_COLS + ["win_probability"] + ds.analytical_feature_cols(best_of)


def player_snapshot(matches_features: pd.DataFrame, current_ratings: pd.DataFrame,
                     player_name: str, tour: str, as_of_date, form_window: int = 20) -> dict:
    """Current Elo/form/streak/hand for a player, as of `as_of_date`.

    Elo comes straight from current_ratings.parquet (each player's rating
    entering their own last recorded match - see features.current_ratings
    for why that's the right semantics here, not a live daily rating).
    Form/streak are recomputed directly from the actual last-20 results
    rather than incrementally, since the exact match sequence is available
    and that's simpler than replaying the training-time state machine.
    """
    hist = matches_features[
        (matches_features["tour"] == tour)
        & ((matches_features["winner_name"] == player_name) | (matches_features["loser_name"] == player_name))
        & (matches_features["tourney_date"] < as_of_date)
    ].sort_values("tourney_date")
    if hist.empty:
        raise ValueError(f"No matches found for {player_name!r} in {tour} before {as_of_date}")

    last = hist.iloc[-1]
    won_last = last["winner_name"] == player_name
    hand = last["winner_hand"] if won_last else last["loser_hand"]

    recent = hist.tail(form_window)
    results = (recent["winner_name"] == player_name).astype(int).to_numpy()
    form_winrate = results.mean()

    streak = 0
    for r in results[::-1]:
        won = r == 1
        if streak == 0:
            streak = 1 if won else -1
        elif (streak > 0) == won:
            streak += 1 if won else -1
        else:
            break

    elo_row = current_ratings[(current_ratings["tour"] == tour) & (current_ratings["player_name"] == player_name)]
    if elo_row.empty:
        raise ValueError(f"No Elo rating found for {player_name!r} in {tour}")
    elo_row = elo_row.iloc[0]

    return {
        "overall_elo": elo_row["overall_elo"],
        "elo_by_surface": {s: elo_row.get(f"elo_{s}", np.nan) for s in ["Hard", "Clay", "Grass", "Carpet"]},
        "form_winrate": form_winrate,
        "win_streak": streak,
        "last_match_date": last["tourney_date"],
        "hand": hand,
    }


def head_to_head(matches_features: pd.DataFrame, player1: str, player2: str, tour: str,
                  as_of_date) -> tuple[int, int]:
    """(player1_wins, player2_wins) prior to as_of_date, all match levels."""
    mask = (
        (matches_features["tour"] == tour)
        & (matches_features["tourney_date"] < as_of_date)
        & (
            ((matches_features["winner_name"] == player1) & (matches_features["loser_name"] == player2))
            | ((matches_features["winner_name"] == player2) & (matches_features["loser_name"] == player1))
        )
    )
    meetings = matches_features[mask]
    p1_wins = (meetings["winner_name"] == player1).sum()
    p2_wins = (meetings["winner_name"] == player2).sum()
    return int(p1_wins), int(p2_wins)


def _diff_features(a_state, b_state, a_style, b_style, a_seed, b_seed,
                    a_h2h_wins, b_h2h_wins, surface, round_, best_of, as_of_date) -> dict:
    a_surface_elo = a_state["elo_by_surface"].get(surface, np.nan)
    b_surface_elo = b_state["elo_by_surface"].get(surface, np.nan)
    row = {
        "diff_elo": a_state["overall_elo"] - b_state["overall_elo"],
        "diff_surface_elo": a_surface_elo - b_surface_elo,
        "diff_blended_elo": ds.blended_elo(a_state["overall_elo"], a_surface_elo) - ds.blended_elo(b_state["overall_elo"], b_surface_elo),
        "diff_form": a_state["form_winrate"] - b_state["form_winrate"],
        "diff_streak": a_state["win_streak"] - b_state["win_streak"],
        "diff_rest_days": (as_of_date - a_state["last_match_date"]).days - (as_of_date - b_state["last_match_date"]).days,
        "diff_seed": (ds.parse_seed(b_seed) - ds.parse_seed(a_seed)) if (b_seed or a_seed) else np.nan,
        "diff_h2h": a_h2h_wins - b_h2h_wins,
        "surface": surface, "round": round_, "best_of": best_of,
        "opposite_handed": int((a_state["hand"] == "L") != (b_state["hand"] == "L")),
    }
    for col in ds.STYLE_COLS:
        av = a_style[col] if a_style is not None and col in a_style else np.nan
        bv = b_style[col] if b_style is not None and col in b_style else np.nan
        row[f"diff_{col}"] = av - bv

    row["elo_x_hard"] = row["diff_elo"] * (surface == "Hard")
    row["elo_x_clay"] = row["diff_elo"] * (surface == "Clay")
    row["elo_x_grass"] = row["diff_elo"] * (surface == "Grass")
    row["round_num"] = ds.ROUND_RANK.get(round_, 3)
    row["elo_x_round"] = row["diff_elo"] * row["round_num"]
    return row


def predict_matchup(matches_features: pd.DataFrame, current_ratings: pd.DataFrame,
                     style: pd.DataFrame, win_model, margin_model,
                     player1: str, player2: str, tour: str, surface: str, round_: str,
                     best_of: int, match_date=None, p1_seed=None, p2_seed=None,
                     p1_odds=None, p2_odds=None, win_model_with_odds=None) -> dict:
    """Full prediction for a hypothetical (or historical) matchup.

    Returns win probabilities for both players, the margin distribution
    conditional on each player winning, and the overall margin distribution
    (marginalized over who wins) - see module docstring for why that mixture
    step is necessary rather than just calling both models independently.

    `p1_odds`/`p2_odds` (decimal odds, e.g. 1.85) are optional - when both
    are supplied, `win_model_with_odds` (notebook 07's odds-enhanced
    ensemble) is used instead of `win_model` for the win probability, with
    the devigged implied probability folded in as `diff_market_prob`. Real
    future matches won't always have odds on hand, hence this being
    optional rather than required - `win_model` alone still works exactly
    as before when odds aren't supplied.
    """
    match_date = pd.Timestamp(match_date) if match_date is not None else pd.Timestamp.now().normalize()
    use_odds = p1_odds is not None and p2_odds is not None
    if use_odds and win_model_with_odds is None:
        raise ValueError("p1_odds/p2_odds were given but win_model_with_odds was not")

    p1_state = player_snapshot(matches_features, current_ratings, player1, tour, match_date)
    p2_state = player_snapshot(matches_features, current_ratings, player2, tour, match_date)
    p1_h2h, p2_h2h = head_to_head(matches_features, player1, player2, tour, match_date)

    style_lookup = style.set_index("player")
    p1_style = style_lookup.loc[player1] if player1 in style_lookup.index else None
    p2_style = style_lookup.loc[player2] if player2 in style_lookup.index else None

    row_p1 = _diff_features(p1_state, p2_state, p1_style, p2_style, p1_seed, p2_seed,
                             p1_h2h, p2_h2h, surface, round_, best_of, match_date)
    if use_odds:
        p1_implied = odds_mod.devig_implied_prob(p1_odds, p2_odds)
        row_p1["diff_market_prob"] = p1_implied - (1 - p1_implied)
        X_win = pd.DataFrame([row_p1])[WIN_FEATURE_COLS_WITH_ODDS]
        p1_win_prob = float(win_model_with_odds.predict_proba(X_win)[0, 1])
    else:
        X_win = pd.DataFrame([row_p1])[WIN_FEATURE_COLS]
        p1_win_prob = float(win_model.predict_proba(X_win)[0, 1])
    p2_win_prob = 1.0 - p1_win_prob

    margin_cols = margin_feature_cols(best_of)

    def add_analytical(row, win_prob):
        analytical = ds.analytical_margin_probs(win_prob, best_of)
        return dict(row, win_probability=win_prob,
                    **{f"analytical_{key}": value for key, value in analytical.items()})

    row_p1_margin = add_analytical(row_p1, p1_win_prob)
    margin_if_p1_wins = dict(zip(
        margin_model.classes_,
        margin_model.predict_proba(pd.DataFrame([row_p1_margin])[margin_cols])[0],
    ))

    row_p2 = _diff_features(p2_state, p1_state, p2_style, p1_style, p2_seed, p1_seed,
                             p2_h2h, p1_h2h, surface, round_, best_of, match_date)
    row_p2_margin = add_analytical(row_p2, p2_win_prob)
    margin_if_p2_wins = dict(zip(
        margin_model.classes_,
        margin_model.predict_proba(pd.DataFrame([row_p2_margin])[margin_cols])[0],
    ))

    all_classes = set(margin_if_p1_wins) | set(margin_if_p2_wins)
    overall_margin = {
        m: p1_win_prob * margin_if_p1_wins.get(m, 0.0) + p2_win_prob * margin_if_p2_wins.get(m, 0.0)
        for m in all_classes
    }

    return {
        "player1": player1, "player2": player2,
        "player1_win_prob": p1_win_prob,
        "player2_win_prob": p2_win_prob,
        "margin_if_player1_wins": margin_if_p1_wins,
        "margin_if_player2_wins": margin_if_p2_wins,
        "overall_margin_distribution": overall_margin,
    }
