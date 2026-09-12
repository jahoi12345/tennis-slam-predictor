"""Assembles the symmetric player1-vs-player2 modeling table for the Slam
win classifier. See notebooks/03_win_classifier.ipynb.

Raw match rows are winner/loser-labeled, which a classifier can't be trained
on directly (there's no "winner" before the match happens, and always
putting the actual winner in the same column would just teach the model to
read the label). So each match is randomly reassigned to player1/player2 and
every player-specific feature becomes a signed difference (p1 - p2), with
target = 1 if player1 actually won.
"""
import re

import numpy as np
import pandas as pd
from scipy.optimize import brentq

SURFACE_ELO_WEIGHT = 0.8  # weight on surface-specific Elo vs. overall Elo in the blend


def _bo5_match_win_prob(p: float) -> float:
    """P(win a best-of-5 match) given a constant per-set win probability p,
    assuming sets are i.i.d. Bernoulli trials (no momentum/fatigue)."""
    q = 1 - p
    return p**3 * (1 + 3 * q + 6 * q**2)


def _bo3_match_win_prob(p: float) -> float:
    return p**2 * (3 - 2 * p)


def analytical_margin_probs(match_win_prob: float, best_of: int) -> dict:
    """Derives the set-score margin distribution purely from the match win
    probability, via an i.i.d.-per-set model: find the per-set win
    probability p whose implied match win probability matches
    `match_win_prob` (root-finding, since the map is monotonic), then use
    the standard negative-binomial race-to-N formulas for exactly how many
    sets a race-to-3 (best of 5) or race-to-2 (best of 3) match takes.

    The margin label doesn't care who won ("the match went 3-1" either way),
    so this returns win-agnostic probabilities: P(margin=k-0) = p^k + q^k,
    not just the favorite's side.

    This is a theory-only alternative to the learned classifier - no
    training data, no overfitting risk, but also no access to anything
    beyond the win probability (surface, H2H, seed, style all discarded).
    Tested in notebook 04 as both a standalone baseline and a source of
    extra input features for the learned models - it underperforms alone
    (real matches aren't perfectly i.i.d. per set) but adding its outputs
    as features to logistic regression gave the best all-around result of
    everything tried, since the two encode genuinely different information.
    """
    f = _bo5_match_win_prob if best_of == 5 else _bo3_match_win_prob
    mwp = np.clip(match_win_prob, 1e-9, 1 - 1e-9)
    p = brentq(lambda x: f(x) - mwp, 1e-9, 1 - 1e-9)
    q = 1 - p
    if best_of == 5:
        return {"p_sweep": p**3 + q**3, "p_four": 3 * p * q * (p**2 + q**2), "p_five": 6 * p**2 * q**2}
    return {"p_sweep": p**2 + q**2, "p_three": 2 * p * q}


def analytical_feature_cols(best_of: int) -> list:
    return ["analytical_p_sweep", "analytical_p_four", "analytical_p_five"] if best_of == 5 \
        else ["analytical_p_sweep", "analytical_p_three"]


def add_analytical_margin_features(df: pd.DataFrame, best_of: int) -> pd.DataFrame:
    """Adds the analytical_margin_probs() outputs as columns, keyed off the
    existing win_probability column. See analytical_margin_probs docstring."""
    df = df.copy()
    probs = df["win_probability"].apply(lambda p: analytical_margin_probs(p, best_of))
    for col in analytical_feature_cols(best_of):
        key = col.replace("analytical_", "")
        df[col] = [p[key] for p in probs]
    return df


def blended_elo(overall_elo, surface_elo, weight: float = SURFACE_ELO_WEIGHT):
    """A single Elo number blending surface-specific and overall form,
    weighted toward the surface (matching FiveThirtyEight's tennis Elo
    approach). Given as its own feature because a linear model can't
    discover this weighted combination on its own from two separate diff
    columns - it would need an explicit interaction term to do so."""
    return weight * surface_elo + (1 - weight) * overall_elo


# Round difficulty as a plain number (deeper rounds = bigger favorites tend to
# hold serve/win more predictably) - used only for the elo x round interaction
# term below, not the Elo computation itself (see features.ROUND_ORDER for
# that, which this mirrors).
ROUND_RANK = {"R128": 1, "R64": 2, "R32": 3, "R16": 4, "QF": 5, "SF": 6, "F": 7, "RR": 3}

INTERACTION_COLS = ["elo_x_hard", "elo_x_clay", "elo_x_grass", "elo_x_round", "round_num"]


def add_elo_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Explicit diff_elo x surface and diff_elo x round terms, for the
    logistic regression component of the win classifier. A linear model
    can't discover "the Elo gap matters more/less on this surface, or this
    deep into the draw" on its own from diff_elo and surface/round as
    separate columns - it needs the product term spelled out. Tree models
    (HistGradientBoosting) don't need this: they can already split on
    surface/round and then on diff_elo within that split, which is an
    interaction by construction.
    """
    df = df.copy()
    df["elo_x_hard"] = df["diff_elo"] * (df["surface"] == "Hard")
    df["elo_x_clay"] = df["diff_elo"] * (df["surface"] == "Clay")
    df["elo_x_grass"] = df["diff_elo"] * (df["surface"] == "Grass")
    df["round_num"] = df["round"].map(ROUND_RANK).fillna(3)
    df["elo_x_round"] = df["diff_elo"] * df["round_num"]
    return df


STYLE_COLS = [
    "fh_winner_share", "bh_winner_share", "fh_unforced_share", "bh_unforced_share",
    "pts_short_rally_share", "pts_long_rally_share", "pts_very_long_rally_share",
    "net_point_win_rate", "snv_rate", "avg_serve_speed_kmh",
]


_INCOMPLETE_MATCH_RE = re.compile(r"RET|W/O|DEF|ABD")
_SET_TOKEN_RE = re.compile(r"^(\d+)-(\d+)$")


def parse_set_score(score) -> tuple[int, int] | None:
    """Returns (winner_sets, loser_sets) for a genuinely completed match, or
    None for anything that didn't finish normally (retirement, walkover,
    default, abandonment) or fails to parse. Those are excluded from margin
    modeling entirely: a 3-0 via retirement isn't the same signal as a 3-0
    played to completion.
    """
    if pd.isna(score) or _INCOMPLETE_MATCH_RE.search(score):
        return None
    winner_sets = loser_sets = 0
    for token in score.split():
        core = re.sub(r"\(.*?\)", "", token)  # strip tiebreak score, e.g. '7-6(4)' -> '7-6'
        m = _SET_TOKEN_RE.match(core)
        if not m:
            return None
        a, b = int(m.group(1)), int(m.group(2))
        if a == b:
            return None
        winner_sets += a > b
        loser_sets += a < b
    return winner_sets, loser_sets


def parse_game_totals(score) -> tuple[int, int] | tuple[None, None]:
    """Total games won by the winner/loser, as lenient as possible (skips
    any token it can't parse instead of failing the whole match) - unlike
    parse_set_score, this is meant to degrade gracefully across the FULL,
    messier match history (not just the already-vetted Slam subset), for
    use as an Elo margin-of-victory signal. Retirements/defaults still
    return whatever games were actually completed up to that point, which
    is a real (if partial) dominance signal; walkovers return (None, None)
    since no games were played at all.
    """
    if pd.isna(score):
        return None, None
    total_w = total_l = 0
    any_valid = False
    for token in score.split():
        if token in ("RET", "W/O", "DEF", "ABD"):
            continue
        core = re.sub(r"\(.*?\)", "", token)
        m = _SET_TOKEN_RE.match(core)
        if not m:
            continue
        total_w += int(m.group(1))
        total_l += int(m.group(2))
        any_valid = True
    return (total_w, total_l) if any_valid else (None, None)


def parse_seed(seed) -> float:
    """Numeric seed value where parseable ('1' -> 1, '3F' -> 3 for
    round-robin-style seeding); NaN for 'Q'/'WC'/'ALT'/etc (unseeded)."""
    if pd.isna(seed):
        return np.nan
    m = re.match(r"^\d+", str(seed))
    return float(m.group()) if m else np.nan


def build_matchup_dataset(matches_features: pd.DataFrame, style: pd.DataFrame,
                           tour: str, random_state: int = 0) -> pd.DataFrame:
    """matches_features: output of features.add_rest_days(features.build_temporal_features(...)).
    style: style_features_{atp,wta}.parquet for the matching tour.
    Filters to Grand Slam matches for `tour`, drops walkovers (no play
    occurred - nothing to predict), and returns one row per match with
    symmetric p1/p2 diff features and a random target.
    """
    df = matches_features[
        (matches_features["tour"] == tour)
        & (matches_features["tourney_level"] == "G")
        & (matches_features["score"] != "W/O")
    ].copy().reset_index(drop=True)

    df["winner_seed_num"] = df["winner_seed"].map(parse_seed)
    df["loser_seed_num"] = df["loser_seed"].map(parse_seed)
    df["winner_blended_elo"] = blended_elo(df["winner_elo_pre"], df["winner_surface_elo_pre"])
    df["loser_blended_elo"] = blended_elo(df["loser_elo_pre"], df["loser_surface_elo_pre"])

    style_lookup = style.set_index("player")[STYLE_COLS]
    winner_style = style_lookup.reindex(df["winner_name"]).reset_index(drop=True)
    loser_style = style_lookup.reindex(df["loser_name"]).reset_index(drop=True)
    for col in STYLE_COLS:
        df[f"winner_{col}"] = winner_style[col]
        df[f"loser_{col}"] = loser_style[col]

    rng = np.random.RandomState(random_state)
    flip = rng.random(len(df)) < 0.5  # True: player1 is the actual winner

    out = pd.DataFrame({
        "tourney_date": df["tourney_date"],
        "tourney_name": df["tourney_name"],
        "surface": df["surface"],
        "round": df["round"],
        "best_of": df["best_of"],
        "p1_name": np.where(flip, df["winner_name"], df["loser_name"]),
        "p2_name": np.where(flip, df["loser_name"], df["winner_name"]),
        "p1_hand": np.where(flip, df["winner_hand"], df["loser_hand"]),
        "p2_hand": np.where(flip, df["loser_hand"], df["winner_hand"]),
        "target": flip.astype(int),
    })
    out["opposite_handed"] = (
        (out["p1_hand"] == "L") != (out["p2_hand"] == "L")
    ).astype(int)

    sign = np.where(flip, 1.0, -1.0)  # +1 when p1=winner, so (winner - loser) already equals (p1 - p2)

    def diff(winner_col, loser_col):
        return sign * (df[winner_col] - df[loser_col])

    out["diff_elo"] = diff("winner_elo_pre", "loser_elo_pre")
    out["diff_surface_elo"] = diff("winner_surface_elo_pre", "loser_surface_elo_pre")
    out["diff_blended_elo"] = diff("winner_blended_elo", "loser_blended_elo")
    out["diff_form"] = diff("winner_form_winrate", "loser_form_winrate")
    out["diff_streak"] = diff("winner_win_streak", "loser_win_streak")
    out["diff_rest_days"] = diff("winner_rest_days", "loser_rest_days")
    # seed: LOWER is better, so swap operand order to keep "positive favors p1"
    out["diff_seed"] = diff("loser_seed_num", "winner_seed_num")
    out["diff_h2h"] = sign * (df["winner_h2h_wins"] - df["winner_h2h_losses"])
    if "winner_implied_prob" in df.columns:
        # only present when matches_features has been through odds.attach_odds();
        # NaN before 2000 (ATP) / 2007 (WTA) or for any unmatched match, same
        # graceful-degradation as every other optional feature in this project.
        out["diff_market_prob"] = diff("winner_implied_prob", "loser_implied_prob")
    for col in STYLE_COLS:
        out[f"diff_{col}"] = diff(f"winner_{col}", f"loser_{col}")

    return out


def build_margin_dataset(matches_features: pd.DataFrame, style: pd.DataFrame,
                          tour: str, best_of: int) -> pd.DataFrame:
    """Set-score margin target, from the WINNER's perspective (unlike the
    win classifier, this target already has a canonical orientation - "the
    winner won 3-1" - so there's no need for the p1/p2 symmetric framing).

    Only genuinely completed matches at the given `best_of` are included
    (see parse_set_score); ATP has a small number of legacy best-of-3 Slam
    matches from the early Open era that are excluded by filtering to
    best_of == 5, matching standard practice for this analysis.
    """
    df = matches_features[
        (matches_features["tour"] == tour)
        & (matches_features["tourney_level"] == "G")
        & (matches_features["best_of"] == best_of)
    ].copy().reset_index(drop=True)

    parsed = df["score"].map(parse_set_score)
    df = df[parsed.notna()].copy()
    parsed = parsed[parsed.notna()]
    df["winner_sets"] = [p[0] for p in parsed]
    df["loser_sets"] = [p[1] for p in parsed]

    expected_winner_sets = best_of // 2 + 1
    consistent = df["winner_sets"] == expected_winner_sets
    if not consistent.all():
        df = df[consistent]  # a handful of data-entry errors where recorded sets don't match a real match win
    df["margin"] = df["winner_sets"].astype(str) + "-" + df["loser_sets"].astype(str)

    df["winner_seed_num"] = df["winner_seed"].map(parse_seed)
    df["loser_seed_num"] = df["loser_seed"].map(parse_seed)

    style_lookup = style.set_index("player")[STYLE_COLS]
    winner_style = style_lookup.reindex(df["winner_name"]).reset_index(drop=True)
    loser_style = style_lookup.reindex(df["loser_name"]).reset_index(drop=True)
    df = df.reset_index(drop=True)
    for col in STYLE_COLS:
        df[f"winner_{col}"] = winner_style[col]
        df[f"loser_{col}"] = loser_style[col]

    out = pd.DataFrame({
        "tourney_date": df["tourney_date"],
        "tourney_name": df["tourney_name"],
        "surface": df["surface"],
        "round": df["round"],
        "winner_name": df["winner_name"],
        "loser_name": df["loser_name"],
        "margin": df["margin"],
        "best_of": best_of,
    })
    out["opposite_handed"] = ((df["winner_hand"] == "L") != (df["loser_hand"] == "L")).astype(int)

    winner_blended = blended_elo(df["winner_elo_pre"], df["winner_surface_elo_pre"])
    loser_blended = blended_elo(df["loser_elo_pre"], df["loser_surface_elo_pre"])

    out["diff_elo"] = df["winner_elo_pre"] - df["loser_elo_pre"]
    out["diff_surface_elo"] = df["winner_surface_elo_pre"] - df["loser_surface_elo_pre"]
    out["diff_blended_elo"] = winner_blended - loser_blended
    out["diff_form"] = df["winner_form_winrate"] - df["loser_form_winrate"]
    out["diff_streak"] = df["winner_win_streak"] - df["loser_win_streak"]
    out["diff_rest_days"] = df["winner_rest_days"] - df["loser_rest_days"]
    out["diff_seed"] = df["loser_seed_num"] - df["winner_seed_num"]
    out["diff_h2h"] = df["winner_h2h_wins"] - df["winner_h2h_losses"]
    if "winner_implied_prob" in df.columns:
        out["diff_market_prob"] = df["winner_implied_prob"] - df["loser_implied_prob"]
    for col in STYLE_COLS:
        out[f"diff_{col}"] = df[f"winner_{col}"] - df[f"loser_{col}"]

    return out
