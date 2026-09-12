"""Exports lightweight JSON datasets for the three data-viz Artifacts:
radial Elo trajectory, tournament bracket with retrospective predictions,
and a Sinner/Alcaraz head-to-head comparison. Reads only from the already
-built data/processed/*.parquet outputs of notebooks 01-02.
"""
import json
import math
import os

import pandas as pd

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "viz")
os.makedirs(OUT_DIR, exist_ok=True)


def elo_win_prob(elo_a, elo_b):
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def player_trajectory(matches, player_name, tour):
    sub = matches[
        (matches["tour"] == tour)
        & ((matches["winner_name"] == player_name) | (matches["loser_name"] == player_name))
    ].sort_values("tourney_date")
    rows = []
    for _, r in sub.iterrows():
        won = r["winner_name"] == player_name
        elo = r["winner_elo_pre"] if won else r["loser_elo_pre"]
        surf_elo = r["winner_surface_elo_pre"] if won else r["loser_surface_elo_pre"]
        opp = r["loser_name"] if won else r["winner_name"]
        if pd.isna(elo):
            continue
        rows.append({
            "date": r["tourney_date"].strftime("%Y-%m-%d"),
            "tourney": r["tourney_name"],
            "level": r["tourney_level"],
            "round": r["round"],
            "surface": r["surface"],
            "opponent": opp,
            "won": bool(won),
            "is_title": bool(won and r["round"] == "F"),
            "score": r["score"],
            "elo": None if pd.isna(elo) else round(float(elo), 1),
            "surface_elo": None if pd.isna(surf_elo) else round(float(surf_elo), 1),
        })
    return rows


def build_bracket(matches, tourney_id, tour):
    sub = matches[(matches["tourney_id"] == tourney_id) & (matches["tour"] == tour)].copy()
    round_order = ["R128", "R64", "R32", "R16", "QF", "SF", "F"]
    sub = sub[sub["round"].isin(round_order)]
    matches_by_round = {rnd: [] for rnd in round_order}
    for _, r in sub.iterrows():
        w_elo, l_elo = r["winner_elo_pre"], r["loser_elo_pre"]
        pred_prob_winner = None
        if pd.notna(w_elo) and pd.notna(l_elo):
            pred_prob_winner = round(elo_win_prob(w_elo, l_elo), 4)
        matches_by_round[r["round"]].append({
            "round": r["round"],
            "winner": r["winner_name"],
            "loser": r["loser_name"],
            "winner_seed": None if pd.isna(r["winner_seed"]) else str(r["winner_seed"]),
            "loser_seed": None if pd.isna(r["loser_seed"]) else str(r["loser_seed"]),
            "score": r["score"],
            "winner_elo_pre": None if pd.isna(w_elo) else round(float(w_elo), 1),
            "loser_elo_pre": None if pd.isna(l_elo) else round(float(l_elo), 1),
            "elo_model_prob_winner": pred_prob_winner,
            "elo_model_favored_winner": (pred_prob_winner is not None and pred_prob_winner >= 0.5),
        })
    meta = sub.iloc[0]
    return {
        "tourney_id": tourney_id,
        "tourney_name": meta["tourney_name"],
        "surface": meta["surface"],
        "date": meta["tourney_date"].strftime("%Y-%m-%d"),
        "tour": tour,
        "rounds": round_order,
        "matches_by_round": matches_by_round,
    }


def head_to_head(matches, style, p1, p2, tour):
    mask = (
        (matches["tour"] == tour)
        & (
            ((matches["winner_name"] == p1) & (matches["loser_name"] == p2))
            | ((matches["winner_name"] == p2) & (matches["loser_name"] == p1))
        )
    )
    meetings = matches[mask].sort_values("tourney_date")
    meeting_rows = []
    for _, r in meetings.iterrows():
        p1_won = r["winner_name"] == p1
        p1_elo = r["winner_elo_pre"] if p1_won else r["loser_elo_pre"]
        p2_elo = r["loser_elo_pre"] if p1_won else r["winner_elo_pre"]
        meeting_rows.append({
            "date": r["tourney_date"].strftime("%Y-%m-%d"),
            "tourney": r["tourney_name"],
            "level": r["tourney_level"],
            "round": r["round"],
            "surface": r["surface"],
            "score": r["score"],
            "p1_won": bool(p1_won),
            "p1_elo_pre": None if pd.isna(p1_elo) else round(float(p1_elo), 1),
            "p2_elo_pre": None if pd.isna(p2_elo) else round(float(p2_elo), 1),
        })

    style_lookup = style.set_index("player")
    style_cols = [c for c in style.columns if c not in ("player", "tour")]

    def style_row(name):
        if name not in style_lookup.index:
            return None
        row = style_lookup.loc[name]
        return {c: (None if pd.isna(row[c]) else round(float(row[c]), 3)) for c in style_cols}

    def latest_state(name):
        hist = matches[
            (matches["tour"] == tour)
            & ((matches["winner_name"] == name) | (matches["loser_name"] == name))
        ].sort_values("tourney_date")
        last = hist.iloc[-1]
        won = last["winner_name"] == name
        return {
            "overall_elo": round(float(last["winner_elo_pre"] if won else last["loser_elo_pre"]), 1),
            "surface_elo": round(float(last["winner_surface_elo_pre"] if won else last["loser_surface_elo_pre"]), 1),
            "as_of": last["tourney_date"].strftime("%Y-%m-%d"),
            "hand": last["winner_hand"] if won else last["loser_hand"],
        }

    p1_wins = sum(1 for m in meeting_rows if m["p1_won"])
    p2_wins = len(meeting_rows) - p1_wins
    return {
        "player1": p1, "player2": p2, "tour": tour,
        "record": {"p1_wins": p1_wins, "p2_wins": p2_wins},
        "meetings": meeting_rows,
        "style": {"p1": style_row(p1), "p2": style_row(p2)},
        "current": {"p1": latest_state(p1), "p2": latest_state(p2)},
    }


RIVALRY_PLAYERS = [
    ("Roger Federer", "Golden Generation"), ("Rafael Nadal", "Golden Generation"),
    ("Novak Djokovic", "Golden Generation"), ("Andy Murray", "Golden Generation"),
    ("Stan Wawrinka", "Golden Generation"), ("Juan Martin del Potro", "Golden Generation"),
    ("Tomas Berdych", "Golden Generation"), ("David Ferrer", "Golden Generation"),
    ("Jo-Wilfried Tsonga", "Golden Generation"),
    ("Stefanos Tsitsipas", "Lost Generation"), ("Alexander Zverev", "Lost Generation"),
    ("Daniil Medvedev", "Lost Generation"), ("Dominic Thiem", "Lost Generation"),
    ("Kei Nishikori", "Lost Generation"), ("Andrey Rublev", "Lost Generation"),
    ("Jannik Sinner", "Next Gen"), ("Carlos Alcaraz", "Next Gen"),
    ("Casper Ruud", "Next Gen"), ("Hubert Hurkacz", "Next Gen"), ("Taylor Fritz", "Next Gen"),
]


def build_rivalry_network(matches, players, min_meetings=3):
    names = [p for p, _ in players]
    cohort = dict(players)
    sub = matches[
        (matches["tour"] == "ATP")
        & (matches["winner_name"].isin(names))
        & (matches["loser_name"].isin(names))
    ]

    def peak_elo(name):
        hist = matches[
            (matches["tour"] == "ATP")
            & ((matches["winner_name"] == name) | (matches["loser_name"] == name))
        ]
        elos = pd.concat([
            hist.loc[hist["winner_name"] == name, "winner_elo_pre"],
            hist.loc[hist["loser_name"] == name, "loser_elo_pre"],
        ])
        return round(float(elos.max()), 1) if len(elos) else None

    nodes = [{"id": name, "cohort": cohort[name], "peak_elo": peak_elo(name)} for name in names]

    pair_counts = {}
    for _, r in sub.iterrows():
        pair = tuple(sorted([r["winner_name"], r["loser_name"]]))
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
    edges = [
        {"source": a, "target": b, "meetings": n}
        for (a, b), n in pair_counts.items() if n >= min_meetings
    ]
    return {"nodes": nodes, "edges": edges}


TRAJECTORY_PLAYERS = [
    ("atp_federer", "Roger Federer", "ATP"),
    ("atp_nadal", "Rafael Nadal", "ATP"),
    ("atp_djokovic", "Novak Djokovic", "ATP"),
    ("atp_murray", "Andy Murray", "ATP"),
    ("wta_serena", "Serena Williams", "WTA"),
    ("wta_venus", "Venus Williams", "WTA"),
    ("wta_swiatek", "Iga Swiatek", "WTA"),
]


def all_slam_tourneys(matches, min_date="2022-01-01"):
    slams = matches[
        (matches["tourney_level"] == "G") & (matches["tourney_date"] >= min_date) & (matches["draw_size"] == 128)
    ]
    combos = slams[["tourney_id", "tourney_name", "tourney_date", "tour"]].drop_duplicates()
    return combos.sort_values("tourney_date")


if __name__ == "__main__":
    matches = pd.read_parquet("data/processed/matches_with_features.parquet")
    matches["tourney_date"] = pd.to_datetime(matches["tourney_date"])
    style_atp = pd.read_parquet("data/processed/style_features_atp.parquet")

    trajectories = {
        key: player_trajectory(matches, name, tour) for key, name, tour in TRAJECTORY_PLAYERS
    }
    with open(os.path.join(OUT_DIR, "elo_trajectories.json"), "w") as f:
        json.dump(trajectories, f)
    print("trajectories:", {k: len(v) for k, v in trajectories.items()})

    combos = all_slam_tourneys(matches)
    brackets = {}
    for _, row in combos.iterrows():
        key = f"{row['tourney_id']}_{row['tour']}"
        brackets[key] = build_bracket(matches, row["tourney_id"], row["tour"])
    with open(os.path.join(OUT_DIR, "brackets.json"), "w") as f:
        json.dump(brackets, f)
    print("brackets:", len(brackets), "tournaments")

    h2h = head_to_head(matches, style_atp, "Jannik Sinner", "Carlos Alcaraz", "ATP")
    with open(os.path.join(OUT_DIR, "h2h_sinner_alcaraz.json"), "w") as f:
        json.dump(h2h, f)
    print("h2h meetings:", len(h2h["meetings"]), "record:", h2h["record"])

    rivalry_network = build_rivalry_network(matches, RIVALRY_PLAYERS)
    with open(os.path.join(OUT_DIR, "rivalry_network.json"), "w") as f:
        json.dump(rivalry_network, f)
    print("rivalry network:", len(rivalry_network["nodes"]), "nodes,", len(rivalry_network["edges"]), "edges")

    slam_finals = matches[(matches["tourney_level"] == "G") & (matches["round"] == "F")]
    SLAM_COUNT_PLAYERS = [
        ("Roger Federer", "ATP"), ("Rafael Nadal", "ATP"), ("Novak Djokovic", "ATP"), ("Andy Murray", "ATP"),
        ("Jannik Sinner", "ATP"), ("Carlos Alcaraz", "ATP"),
        ("Serena Williams", "WTA"), ("Venus Williams", "WTA"), ("Iga Swiatek", "WTA"),
    ]
    slam_counts = [
        {"name": name, "tour": tour, "titles": int((slam_finals["winner_name"] == name).sum())}
        for name, tour in SLAM_COUNT_PLAYERS
    ]
    with open(os.path.join(OUT_DIR, "slam_counts.json"), "w") as f:
        json.dump(slam_counts, f)
    print("slam counts:", slam_counts)
