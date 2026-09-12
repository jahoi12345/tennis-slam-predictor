"""Live in-match win probability: reconstructing point-by-point state from
the Match Charting Project, and a hierarchical Markov-chain model (Klaassen
& Magnus 2003 / the standard tennis win-probability recursion) that computes
exact win probability from any live score, given each player's probability
of winning a point on their own serve. See notebooks/06_live_win_probability.ipynb.

Scope note: only the Match Charting Project's point-by-point files are used
here, not the Slam point-by-point files also in data/external/ - Charting
Project already has explicit cumulative Set1/Set2 columns and covers far
more matches (11,601 vs ~2,000, and both tours vs Slams-only), so it's the
better primary source. The Slam files' richer per-point columns (serve
speed, rally length) are a documented future enrichment, not included here.
"""
import glob
import os
import re
from functools import lru_cache

import numpy as np
import pandas as pd

import dataset as ds

MCP_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "external", "tennis_MatchChartingProject")

_GAME_POINT_MAP = {"0": 0, "15": 1, "30": 2, "40": 3, "AD": 4}


def parse_point_score(pts: str, is_tiebreak: bool) -> tuple[int, int]:
    """'15-30' -> (1, 2); tiebreak scores ('0-1', '6-4') are already plain
    integers. 'AD' maps to 4 (one past 40), which combines correctly with
    game_win_prob's diff-based deuce handling (see that function's
    docstring) without needing a separate deuce/advantage encoding.

    Returns (server_points, receiver_points) - NOT (p1, p2). Pts follows
    the universal tennis convention of server's score first ("40-15" always
    means the server has 40), regardless of which numbered player is
    serving - confirmed by tracing a real tiebreak point-by-point, where
    interpreting Pts as a fixed player1-player2 ordering produced scores
    that contradicted the next row's. Caller must reorient using Svr.
    """
    a, b = pts.split("-")
    if is_tiebreak:
        return int(a), int(b)
    return _GAME_POINT_MAP[a], _GAME_POINT_MAP[b]


def load_point_sequences(gender: str) -> pd.DataFrame:
    """All charted points for 'm' or 'w', across the to-2009/2010s/2020s
    files, with Pts parsed into numeric (points_p1, points_p2) - the score
    ENTERING each point (i.e. before PtWinner's point is applied).

    `TbSet` is a whole-SET-level flag ("this set is decided by a tiebreak"),
    not a per-point one - and small scores like "1-0"/"1-1" are ambiguous
    between game and tiebreak notation. The reliable signal that a specific
    point's Pts field uses tiebreak (plain integer) notation rather than
    game (0/15/30/40/AD) notation is that both players have reached 6
    games in that set: scoring only switches format once the breaker
    actually starts. Everything else - including the games column
    occasionally running past 6-6 without ever reaching a tiebreak, in
    older advantage-final-set matches - is regular game notation.
    """
    pattern = os.path.join(MCP_DIR, f"charting-{gender}-points-*.csv")
    frames = [pd.read_csv(f, low_memory=False) for f in sorted(glob.glob(pattern))]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["Gm1"].notna() & df["Gm2"].notna()].copy()  # a handful of rows have corrupt/missing game state
    df["TbSet"] = df["TbSet"].fillna(False).astype(bool)
    is_tiebreak_point = df["TbSet"] & (df["Gm1"] == 6) & (df["Gm2"] == 6)

    split = df["Pts"].str.split("-", n=1, expand=True)
    game_server = split[0].map(_GAME_POINT_MAP)
    game_receiver = split[1].map(_GAME_POINT_MAP)
    tb_server = pd.to_numeric(split[0], errors="coerce")
    tb_receiver = pd.to_numeric(split[1], errors="coerce")
    df["is_tiebreak_point"] = is_tiebreak_point
    points_server = np.where(is_tiebreak_point, tb_server, game_server)
    points_receiver = np.where(is_tiebreak_point, tb_receiver, game_receiver)

    # Pts is server-first; reorient to (p1, p2) using who's actually serving.
    p1_serves = df["Svr"] == 1
    points_p1 = np.where(p1_serves, points_server, points_receiver)
    points_p2 = np.where(p1_serves, points_receiver, points_server)

    unparsed = pd.isna(points_p1) | pd.isna(points_p2)
    if unparsed.any():
        df = df[~unparsed].copy()
        points_p1, points_p2 = points_p1[~unparsed], points_p2[~unparsed]
    df["points_p1"] = points_p1.astype(int)
    df["points_p2"] = points_p2.astype(int)
    return df


def load_charting_matches(gender: str) -> pd.DataFrame:
    path = os.path.join(MCP_DIR, f"charting-{gender}-matches.csv")
    df = pd.read_csv(path, low_memory=False)
    df["tour"] = "ATP" if gender == "m" else "WTA"
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d", errors="coerce")
    df["Best of"] = pd.to_numeric(df["Best of"], errors="coerce")
    return df


def _normalize_name(name: str) -> str:
    """Sackmann hyphenates compound names ('Jo-Wilfried Tsonga'); the
    Charting Project uses spaces ('Jo Wilfried Tsonga'). Normalizing both
    to spaces before joining fixes this cheaply without risking false
    matches (compound hyphenated names are rare enough that colliding with
    an unrelated player is not a realistic concern)."""
    return name.replace("-", " ")


def match_row_lookup(matches_features: pd.DataFrame, tour: str) -> dict:
    """(tour, frozenset({name1, name2})) -> list of (tourney_date, row_index)
    for every Sackmann match of that tour - used to attach the full
    pre-match feature row (Elo, seed, H2H, ...) to charted matches by
    player-pair + nearest date. Charting Project names are already full
    names matching Sackmann's format, so this is a name-exact
    (post-normalization) join, unlike the tennis-data.co.uk odds join,
    which needs surname+initial matching.
    """
    df = matches_features[matches_features["tour"] == tour]
    lookup: dict = {}
    for idx, winner, loser, date in zip(df.index, df["winner_name"], df["loser_name"], df["tourney_date"]):
        key = frozenset({_normalize_name(winner), _normalize_name(loser)})
        lookup.setdefault(key, []).append((date, idx))
    return lookup


def attach_sackmann_match(charting_matches: pd.DataFrame, matches_features: pd.DataFrame,
                           tour: str, min_days_before: int = 3, max_days_after: int = 25) -> pd.DataFrame:
    """Adds a `_sackmann_idx` column to charting_matches: the row index into
    matches_features holding that match's full pre-match feature set,
    resolved by player-pair + nearest date (charted matches are dated the
    day actually played; Sackmann's tourney_date is nominally the Monday of
    tournament week, but is occasionally a day or two after the actual
    first day of play - e.g. Slams starting on a Sunday - hence allowing a
    small negative gap too). Rows with no match in the window are dropped.
    """
    lookup = match_row_lookup(matches_features, tour)
    indices = []
    for p1, p2, date in zip(charting_matches["Player 1"], charting_matches["Player 2"], charting_matches["Date"]):
        candidates = lookup.get(frozenset({_normalize_name(p1), _normalize_name(p2)}), [])
        best_idx, best_gap = None, None
        for tourney_date, idx in candidates:
            gap = (date - tourney_date).days
            if -min_days_before <= gap <= max_days_after and (best_gap is None or abs(gap) < abs(best_gap)):
                best_idx, best_gap = idx, gap
        indices.append(best_idx)
    out = charting_matches.copy()
    out["_sackmann_idx"] = indices
    return out[out["_sackmann_idx"].notna()].reset_index(drop=True)


STYLE_COLS = [
    "fh_winner_share", "bh_winner_share", "fh_unforced_share", "bh_unforced_share",
    "pts_short_rally_share", "pts_long_rally_share", "pts_very_long_rally_share",
    "net_point_win_rate", "snv_rate", "avg_serve_speed_kmh",
]


def build_match_context(gender: str, matches_features: pd.DataFrame, style: pd.DataFrame) -> pd.DataFrame:
    """One row per successfully-joined charted match, with Player1-oriented
    diff features (positive favors Player 1) pulled from the matched
    Sackmann row's pre-match state, plus style diffs looked up by name -
    the same feature set and sign convention as dataset.build_matchup_dataset,
    just keyed to charting match_id instead of the win classifier's random
    player1/player2 framing (there's no leakage risk here from always using
    "Player 1"/"Player 2" as charted, since that assignment was made by
    whoever charted the match, not derived from the outcome).
    """
    tour = "ATP" if gender == "m" else "WTA"
    cm = load_charting_matches(gender)
    cm = attach_sackmann_match(cm, matches_features, tour)

    sackmann = matches_features.loc[cm["_sackmann_idx"]].reset_index(drop=True)
    cm = cm.reset_index(drop=True)

    p1_is_winner = np.array([
        _normalize_name(p1) == _normalize_name(w)
        for p1, w in zip(cm["Player 1"], sackmann["winner_name"])
    ])
    sign = np.where(p1_is_winner, 1.0, -1.0)

    def diff(winner_col, loser_col):
        return sign * (sackmann[winner_col].to_numpy() - sackmann[loser_col].to_numpy())

    style_lookup = style.set_index("player")[STYLE_COLS]
    p1_style = style_lookup.reindex(cm["Player 1"]).reset_index(drop=True)
    p2_style = style_lookup.reindex(cm["Player 2"]).reset_index(drop=True)

    out = pd.DataFrame({
        "match_id": cm["match_id"],
        "tour": tour,
        "tourney_date": sackmann["tourney_date"],
        "surface": sackmann["surface"],
        "round": sackmann["round"],
        "best_of": sackmann["best_of"],
        "p1_name": cm["Player 1"],
        "p2_name": cm["Player 2"],
        "p1_wins_match": p1_is_winner.astype(int),
    })
    out["diff_elo"] = diff("winner_elo_pre", "loser_elo_pre")
    out["diff_surface_elo"] = diff("winner_surface_elo_pre", "loser_surface_elo_pre")
    winner_blended = ds.blended_elo(sackmann["winner_elo_pre"], sackmann["winner_surface_elo_pre"])
    loser_blended = ds.blended_elo(sackmann["loser_elo_pre"], sackmann["loser_surface_elo_pre"])
    out["diff_blended_elo"] = sign * (winner_blended.to_numpy() - loser_blended.to_numpy())
    out["diff_form"] = diff("winner_form_winrate", "loser_form_winrate")
    out["diff_streak"] = diff("winner_win_streak", "loser_win_streak")
    if "winner_rest_days" in sackmann.columns:
        out["diff_rest_days"] = diff("winner_rest_days", "loser_rest_days")
    winner_seed_num = sackmann["winner_seed"].map(ds.parse_seed)
    loser_seed_num = sackmann["loser_seed"].map(ds.parse_seed)
    out["diff_seed"] = sign * (loser_seed_num.to_numpy() - winner_seed_num.to_numpy())  # lower seed is better
    out["diff_h2h"] = sign * (sackmann["winner_h2h_wins"].to_numpy() - sackmann["winner_h2h_losses"].to_numpy())
    for col in STYLE_COLS:
        out[f"diff_{col}"] = p1_style[col].to_numpy() - p2_style[col].to_numpy()
    return out


def player_serve_win_rate(gender: str) -> pd.Series:
    """Per-player P(win a point on own serve), from Match Charting
    Project's Overview.csv (serve_pts, first_won, second_won) - not
    carried into style_features_{atp,wta}.parquet by notebook 01, which
    only kept the winner/unforced-error columns from that file, so this
    reads Overview.csv directly rather than duplicating it there for a
    single new use. Falls back to the tour-wide average for anyone with
    too few charted serve points to trust individually.
    """
    path = os.path.join(MCP_DIR, f"charting-{gender}-stats-Overview.csv")
    overview = pd.read_csv(path, low_memory=False)
    total = overview[overview["set"] == "Total"]
    agg = total.groupby("player")[["serve_pts", "first_won", "second_won"]].sum()
    agg = agg[agg["serve_pts"] >= 50]  # drop players with too little charted serve data to be reliable
    rate = (agg["first_won"] + agg["second_won"]) / agg["serve_pts"]
    return rate


WIN_MODEL_BASE_COLS = (
    ["diff_elo", "diff_surface_elo", "diff_blended_elo", "diff_form", "diff_streak", "diff_rest_days",
     "diff_seed", "diff_h2h"] + [f"diff_{c}" for c in STYLE_COLS]
    + ["surface", "round", "best_of"]
)


def build_inmatch_dataset(gender: str, matches_features: pd.DataFrame, style: pd.DataFrame, win_model) -> pd.DataFrame:
    """One row per point across every successfully-joined charted match:
    running score state (sets/games/points, who's serving), match-progress
    fraction, and the Player1-oriented pre-match context (including our own
    pre-match win_probability from the production win classifier) - the
    full training set for the live in-match model. `win_model` is the
    win_classifier_{tour}_ensemble.joblib pipeline; it needs the Elo x
    surface/round interaction terms as additional inputs (opposite_handed
    isn't available here since Charting Project's hand columns are
    per-match, not attached in build_match_context - passed as 0, a minor
    simplification since it's a weak feature for the win model anyway).
    """
    ctx = build_match_context(gender, matches_features, style)
    ctx["opposite_handed"] = 0
    ctx_i = ds.add_elo_interactions(ctx)
    win_cols = WIN_MODEL_BASE_COLS + ["opposite_handed"] + ds.INTERACTION_COLS
    ctx["win_probability"] = win_model.predict_proba(ctx_i[win_cols])[:, 1]

    points = load_point_sequences(gender)
    points["total_points"] = points.groupby("match_id")["Pt"].transform("max")
    points["match_progress"] = points["Pt"] / points["total_points"]

    merged = points.merge(ctx, on="match_id", how="inner")
    return merged

# ---------------------------------------------------------------------------
# Hierarchical Markov chain: point -> game -> tiebreak -> set -> match
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def game_win_prob(p: float, a: int = 0, b: int = 0) -> float:
    """P(server wins the game), given p = P(server wins a point on serve),
    starting from a score of `a` points (server) to `b` points (returner).
    Score is in raw point counts (0,1,2,3 = 0/15/30/40); anything from
    (3, 3) onward is resolved via the closed-form deuce/advantage formula
    (gambler's ruin) rather than recursing indefinitely.
    """
    q = 1 - p
    if a >= 4 and a - b >= 2:
        return 1.0
    if b >= 4 and b - a >= 2:
        return 0.0
    if a >= 3 and b >= 3:
        deuce = p**2 / (p**2 + q**2)
        diff = a - b
        if diff == 0:
            return deuce
        if diff == 1:  # server has advantage
            return p + q * deuce
        if diff == -1:  # returner has advantage
            return p * deuce
    return p * game_win_prob(p, a + 1, b) + q * game_win_prob(p, a, b + 1)


def _tiebreak_server(point_index: int) -> str:
    """Which player serves point `point_index` (0-indexed) of a tiebreak:
    'X' served point 1, then serve alternates every 2 points."""
    if point_index == 0:
        return "X"
    block = (point_index - 1) // 2
    return "X" if block % 2 == 1 else "O"


@lru_cache(maxsize=None)
def _deep_tiebreak_states(p_x: float, p_o: float) -> dict:
    """Absorption probabilities for the 6 recurring 'fresh 2-point serve
    block' states once a tiebreak is tied at or beyond target-1 points
    each: FreshBlock(server, diff) for server in {X,O}, diff in {-1,0,1}
    (diff = X's points minus O's points since the tie).

    This exists because naive recursion never terminates here: at a
    perfect tie, both players can in principle stay tied forever (a
    symmetric random walk with no drift), so plain recursion on (a, b)
    blows Python's call stack. But the tiebreak's serve pattern (2 points
    per server, alternating) means only 6 distinct *relative* states ever
    recur once tied this deep - solving them as one small linear system
    (each state's value is a known combination of itself and the other 5,
    since a block's outcome either terminates the tiebreak outright or
    lands back in one of these same 6 states) replaces the infinite
    recursion with an O(6x6) solve done once per (p_x, p_o) pair.
    """
    states = [("X", -1), ("X", 0), ("X", 1), ("O", -1), ("O", 0), ("O", 1)]
    index = {s: i for i, s in enumerate(states)}
    A = np.zeros((6, 6))
    rhs = np.zeros(6)
    for server, diff in states:
        i = index[(server, diff)]
        r = p_x if server == "X" else (1 - p_o)  # P(X wins a point served by `server`)
        other = "O" if server == "X" else "X"
        A[i, i] = 1.0
        for delta, weight in [(2, r * r), (0, 2 * r * (1 - r)), (-2, (1 - r) * (1 - r))]:
            new_diff = diff + delta
            if new_diff >= 2:
                rhs[i] += weight * 1.0
            elif new_diff <= -2:
                rhs[i] += weight * 0.0
            else:
                A[i, index[(other, new_diff)]] -= weight
    solution = np.linalg.solve(A, rhs)
    return dict(zip(states, solution))


@lru_cache(maxsize=None)
def tiebreak_win_prob(p_x: float, p_o: float, a: int = 0, b: int = 0, target: int = 7) -> float:
    """P(player X wins the tiebreak), given P(X wins a point on X's own
    serve) = p_x and P(O wins a point on O's own serve) = p_o, from a
    current tiebreak score of a-b (X-O). Handles the alternating-serve
    pattern (1 point, then alternating 2-point blocks) explicitly.
    """
    if a >= target and a - b >= 2:
        return 1.0
    if b >= target and b - a >= 2:
        return 0.0
    if a == b and a >= target - 1:
        # Tied deep enough that either side could stay tied indefinitely -
        # resolve via the closed-form deep-tiebreak states rather than
        # recursing. We're mid the current server's 2-point block (ties
        # only occur at even total points played, which always falls
        # between a block's two points); after this one point resolves,
        # the OTHER player's fresh 2-point block begins from a +/-1 diff.
        server = _tiebreak_server(a + b)
        p_x_wins_point = p_x if server == "X" else (1 - p_o)
        next_block_server = "O" if server == "X" else "X"
        deep = _deep_tiebreak_states(p_x, p_o)
        return p_x_wins_point * deep[(next_block_server, 1)] + \
            (1 - p_x_wins_point) * deep[(next_block_server, -1)]
    server = _tiebreak_server(a + b)
    p_x_wins_point = p_x if server == "X" else (1 - p_o)
    return p_x_wins_point * tiebreak_win_prob(p_x, p_o, a + 1, b, target) + \
        (1 - p_x_wins_point) * tiebreak_win_prob(p_x, p_o, a, b + 1, target)


@lru_cache(maxsize=None)
def _deep_advantage_set_tied(p_x: float, p_o: float) -> float:
    """P(X wins an advantage-scoring set | currently tied beyond 6-6, no
    tiebreak). Closed form, same reasoning as _deep_tiebreak_states but
    simpler: games (unlike tiebreak points) alternate server one at a time,
    so there are only 2 recurring states once tied (X-serves-tied,
    O-serves-tied) rather than 6 - solved by hand below rather than via a
    linear-algebra solve, and it comes out server-independent (D_x0 ==
    D_o0), consistent with the proven fact that set-win probability doesn't
    depend on who serves first. Exists because real historical data
    occasionally has advantage-format deciding sets (games continuing past
    6-6 with no tiebreak - e.g. pre-2019 Wimbledon, pre-2022 French Open,
    some Davis Cup ties), which the standard tiebreak-at-6-6 recursion
    below can't resolve (naive recursion at a deep tie never terminates,
    for the same reason the tiebreak case originally didn't).
    """
    gx, go = game_win_prob(p_x), game_win_prob(p_o)
    denom = gx + go - 2 * gx * go
    return gx * (1 - go) / denom


def _deep_advantage_set_prob(p_x: float, p_o: float, diff: int, x_serves_next: bool) -> float:
    """P(X wins), for diff in {-1, 0, 1} once already in advantage-set
    territory (see _deep_advantage_set_tied). diff = games_x - games_o."""
    tied = _deep_advantage_set_tied(p_x, p_o)
    gx, go = game_win_prob(p_x), game_win_prob(p_o)
    if diff == 0:
        return tied
    if diff == 1:
        return gx * 1.0 + (1 - gx) * tied if x_serves_next else (1 - go) * 1.0 + go * tied
    if diff == -1:
        return gx * tied if x_serves_next else (1 - go) * tied
    raise ValueError(f"diff must be -1, 0, or 1 once already tied deep; got {diff}")


@lru_cache(maxsize=None)
def set_win_prob(p_x: float, p_o: float, games_x: int = 0, games_o: int = 0, x_serves_next: bool = True) -> float:
    """P(player X wins the set), given each player's own-serve point-win
    probability, from a current game score of games_x-games_o, with
    `x_serves_next` indicating who serves the next game. Sets go to 6 games
    (win by 2), or a 7-point tiebreak at 6-6 - unless already deeper than
    6-6 with no tiebreak (an advantage-scoring set, see
    _deep_advantage_set_tied), in which case that closed form is used
    instead of recursing indefinitely.
    """
    if games_x >= 6 and games_x - games_o >= 2:
        return 1.0
    if games_o >= 6 and games_o - games_x >= 2:
        return 0.0
    if games_x == 6 and games_o == 6:
        return tiebreak_win_prob(p_x, p_o)
    if games_x >= 6 and games_o >= 6:
        return _deep_advantage_set_prob(p_x, p_o, games_x - games_o, x_serves_next)
    p_x_wins_game = game_win_prob(p_x) if x_serves_next else (1 - game_win_prob(p_o))
    return p_x_wins_game * set_win_prob(p_x, p_o, games_x + 1, games_o, not x_serves_next) + \
        (1 - p_x_wins_game) * set_win_prob(p_x, p_o, games_x, games_o + 1, not x_serves_next)


def match_win_prob(p_x: float, p_o: float, sets_x: int, sets_o: int,
                    games_x: int, games_o: int, pts_x: int, pts_o: int,
                    x_serves: bool, best_of: int,
                    in_tiebreak: bool = False) -> float:
    """P(player X wins the match) from any live score. Not itself cached
    (the point-score args make the state space too large to usefully
    memoize at this level; the layers below are cached and do the heavy
    lifting). `pts_x`/`pts_o` are the current game's point score (or
    tiebreak score if `in_tiebreak`); (0,0) if a new game/tiebreak hasn't
    started yet.
    """
    sets_to_win = best_of // 2 + 1
    if sets_x >= sets_to_win:
        return 1.0
    if sets_o >= sets_to_win:
        return 0.0

    if in_tiebreak:
        p_x_wins_this = tiebreak_win_prob(p_x, p_o, pts_x, pts_o)
    else:
        p_x_wins_this = game_win_prob(p_x, pts_x, pts_o) if x_serves else (1 - game_win_prob(p_o, pts_o, pts_x))

    # Winning the current game/tiebreak advances games (or wins the set outright if this was a 6-6 tiebreak)
    if in_tiebreak:
        games_x_win, games_o_win = 7, games_o  # set won 7-6
        games_x_lose, games_o_lose = games_x, 7
    else:
        games_x_win, games_o_win = games_x + 1, games_o
        games_x_lose, games_o_lose = games_x, games_o + 1

    # A tiebreak win/loss always ends the set outright (7-6) - set_win_prob
    # can't be handed that score (its own game-count logic would try to
    # keep playing games past 6-6), so both terminal cases are short-
    # circuited here instead of delegated.
    if in_tiebreak:
        p_x_wins_set_if_wins_this = 1.0
        p_x_wins_set_if_loses_this = 0.0
    else:
        p_x_wins_set_if_wins_this = 1.0 if (games_x_win >= 6 and games_x_win - games_o_win >= 2) \
            else set_win_prob(p_x, p_o, games_x_win, games_o_win, not x_serves)
        p_x_wins_set_if_loses_this = 0.0 if (games_o_lose >= 6 and games_o_lose - games_x_lose >= 2) \
            else set_win_prob(p_x, p_o, games_x_lose, games_o_lose, not x_serves)

    p_x_wins_set = p_x_wins_this * p_x_wins_set_if_wins_this + (1 - p_x_wins_this) * p_x_wins_set_if_loses_this

    # Whoever wins/loses this set, the next set (if any) starts 0-0. Serve
    # alternates continuously through the whole match regardless of set/
    # tiebreak boundaries (a tiebreak counts as one game in the rotation),
    # so the next set's first server is just the opposite of who's serving
    # the current game/tiebreak - the same "not x_serves" used above.
    next_set_server_is_x = not x_serves
    p_win_match_if_x_wins_set = match_win_prob(p_x, p_o, sets_x + 1, sets_o, 0, 0, 0, 0, next_set_server_is_x, best_of)
    p_win_match_if_x_loses_set = match_win_prob(p_x, p_o, sets_x, sets_o + 1, 0, 0, 0, 0, next_set_server_is_x, best_of)

    return p_x_wins_set * p_win_match_if_x_wins_set + (1 - p_x_wins_set) * p_win_match_if_x_loses_set
