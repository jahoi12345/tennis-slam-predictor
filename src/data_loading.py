"""Loaders for the Sackmann ATP/WTA match history, Match Charting Project,
and Grand Slam point-by-point data. See notebooks/01_data_ingestion.ipynb
for the end-to-end pipeline that calls these.

Data sources (CC BY-NC-SA 4.0, Jeff Sackmann / Tennis Abstract, non-commercial
use only, attribution required):
- data/external/tennis-sackmann-archive  (mirror of tennis_atp + tennis_wta + tennis_slam_pointbypoint)
- data/external/tennis_MatchChartingProject
"""
import glob
import os
import re

import numpy as np
import pandas as pd

EXTERNAL_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "external")
ARCHIVE_DIR = os.path.join(EXTERNAL_DIR, "tennis-sackmann-archive")
MCP_DIR = os.path.join(EXTERNAL_DIR, "tennis_MatchChartingProject")


def load_tour_matches(tour: str) -> pd.DataFrame:
    """Load main tour-level singles matches for 'atp' or 'wta', all years."""
    pattern = os.path.join(ARCHIVE_DIR, tour, f"{tour}_matches_[0-9]*.csv")
    files = sorted(glob.glob(pattern))
    frames = [pd.read_csv(f, low_memory=False) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df["tour"] = tour.upper()
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d", errors="coerce")
    # draw_size is numeric except for a handful of 'exho' (exhibition) rows;
    # seed columns are genuinely mixed (ints, 'Q'/'WC'/'ALT', or '1F'-style
    # round-robin seeds) so they're kept as strings rather than coerced.
    df["draw_size"] = pd.to_numeric(df["draw_size"], errors="coerce")
    for col in ["winner_seed", "loser_seed"]:
        df[col] = df[col].astype("string")
    # a handful of rows have lowercase 'clay' instead of 'Clay'
    df["surface"] = df["surface"].str.capitalize()
    return df


def load_players(tour: str) -> pd.DataFrame:
    path = os.path.join(ARCHIVE_DIR, tour, f"{tour}_players.csv")
    df = pd.read_csv(path, low_memory=False)
    df["tour"] = tour.upper()
    df["dob"] = pd.to_datetime(df["dob"], format="%Y%m%d", errors="coerce")
    return df


def load_rankings(tour: str) -> pd.DataFrame:
    """Concatenate all decade ranking files plus the current one."""
    pattern = os.path.join(ARCHIVE_DIR, tour, f"{tour}_rankings_*.csv")
    files = sorted(glob.glob(pattern))
    frames = [pd.read_csv(f, low_memory=False) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df["tour"] = tour.upper()
    df["ranking_date"] = pd.to_datetime(df["ranking_date"], format="%Y%m%d", errors="coerce")
    df = df.rename(columns={"player": "player_id"})
    return df.drop_duplicates(subset=["tour", "ranking_date", "player_id"])


# ---------------------------------------------------------------------------
# Match Charting Project: shot-type / rally / net-point style features
# ---------------------------------------------------------------------------

def _mcp_gender_files(gender: str, stat: str) -> str:
    return os.path.join(MCP_DIR, f"charting-{gender}-stats-{stat}.csv")


def load_charting_matches(gender: str) -> pd.DataFrame:
    path = os.path.join(MCP_DIR, f"charting-{gender}-matches.csv")
    df = pd.read_csv(path, low_memory=False)
    df["tour"] = "ATP" if gender == "m" else "WTA"
    return df


def _match_to_players(charting_matches: pd.DataFrame) -> pd.DataFrame:
    """match_id -> {player_1, player_2} long form, one row per (match_id, player)."""
    p1 = charting_matches[["match_id", "Player 1"]].rename(columns={"Player 1": "player"})
    p2 = charting_matches[["match_id", "Player 2"]].rename(columns={"Player 2": "player"})
    return pd.concat([p1, p2], ignore_index=True)


def build_shot_style_features(gender: str) -> pd.DataFrame:
    """Per-player forehand/backhand reliance: share of winners and unforced
    errors coming off each wing, aggregated across all charted matches.
    Source: charting-*-stats-Overview.csv, row == 'Total' (whole-match row).
    """
    overview = pd.read_csv(_mcp_gender_files(gender, "Overview"), low_memory=False)
    total = overview[overview["set"] == "Total"]
    agg = (
        total.groupby("player")[
            ["serve_pts", "return_pts", "winners", "winners_fh", "winners_bh",
             "unforced", "unforced_fh", "unforced_bh"]
        ]
        .sum()
        .reset_index()
    )
    agg["fh_winner_share"] = agg["winners_fh"] / (agg["winners_fh"] + agg["winners_bh"])
    agg["bh_winner_share"] = agg["winners_bh"] / (agg["winners_fh"] + agg["winners_bh"])
    agg["fh_unforced_share"] = agg["unforced_fh"] / (agg["unforced_fh"] + agg["unforced_bh"])
    agg["bh_unforced_share"] = agg["unforced_bh"] / (agg["unforced_fh"] + agg["unforced_bh"])
    agg["tour"] = "ATP" if gender == "m" else "WTA"
    return agg


def build_rally_style_features(gender: str) -> pd.DataFrame:
    """Per-player share of points won at short/medium/long rally lengths."""
    rally = pd.read_csv(_mcp_gender_files(gender, "Rally"), low_memory=False)
    bucket_map = {
        "1-3": "short", "4-6": "medium", "7-9": "long", "10": "very_long",
    }
    rally = rally[rally["row"].isin(bucket_map)].copy()
    rally["bucket"] = rally["row"].map(bucket_map)

    long_rows = []
    for _, r in rally.iterrows():
        long_rows.append({"player": r["server"], "bucket": r["bucket"], "pts": r["pts"], "pts_won": r["pl1_won"]})
        long_rows.append({"player": r["returner"], "bucket": r["bucket"], "pts": r["pts"], "pts_won": r["pl2_won"]})
    long_df = pd.DataFrame(long_rows)
    agg = long_df.groupby(["player", "bucket"])[["pts", "pts_won"]].sum().reset_index()
    pivot = agg.pivot(index="player", columns="bucket", values="pts").fillna(0)
    pivot.columns = [f"pts_{c}_rally" for c in pivot.columns]
    total = pivot.sum(axis=1)
    share = pivot.div(total, axis=0).add_suffix("_share")
    out = pd.concat([pivot, share], axis=1).reset_index()
    out["tour"] = "ATP" if gender == "m" else "WTA"
    return out


def build_net_game_features(gender: str) -> pd.DataFrame:
    """Per-player net-point frequency and win rate, plus serve-and-volley rate
    (share of service points where the player served-and-volleyed)."""
    net = pd.read_csv(_mcp_gender_files(gender, "NetPoints"), low_memory=False)
    net_total = net[net["row"] == "NetPoints"]
    net_agg = net_total.groupby("player")[["net_pts", "pts_won"]].sum().reset_index()
    net_agg["net_point_win_rate"] = net_agg["pts_won"] / net_agg["net_pts"]

    snv = pd.read_csv(_mcp_gender_files(gender, "SnV"), low_memory=False)
    snv_pts = snv[snv["row"] == "SnV"].groupby("player")["snv_pts"].sum()
    nonsnv_pts = snv[snv["row"] == "nonSnV"].groupby("player")["snv_pts"].sum()
    snv_agg = pd.concat([snv_pts.rename("snv_pts"), nonsnv_pts.rename("nonsnv_pts")], axis=1).reset_index()
    snv_agg["snv_rate"] = snv_agg["snv_pts"] / (snv_agg["snv_pts"] + snv_agg["nonsnv_pts"])

    out = net_agg[["player", "net_point_win_rate"]].merge(
        snv_agg[["player", "snv_rate"]], on="player", how="outer"
    )
    out["tour"] = "ATP" if gender == "m" else "WTA"
    return out


# ---------------------------------------------------------------------------
# Grand Slam point-by-point: serve speed + forehand/backhand winner tendency
# ---------------------------------------------------------------------------

SLAM_PBP_DIR = os.path.join(ARCHIVE_DIR, "slam_pointbypoint")

_POINTS_COLS = ["match_id", "PointServer", "Speed_KMH", "ServeIndicator",
                "P1Winner", "P2Winner", "WinnerShotType"]


def build_slam_style_features() -> pd.DataFrame:
    """Aggregate serve speed and forehand/backhand winner tendency per player
    across all Grand Slam point-by-point files (2011-2024).

    Caveats (see slam_pointbypoint UPSTREAM_README.md):
    - Column availability/consistency varies by slam and year (IBM Slamtracker
      vs. Infosys MatchBeats providers); AO/FO extraction stopped after 2022.
    - Speed_KMH of 0 means "not recorded", not an actual 0 kph serve - treated
      as missing here.
    - Player join key is name-string matching, not player_id - accented
      characters/suffix variants can cause misses.
    """
    matches_files = sorted(glob.glob(os.path.join(SLAM_PBP_DIR, "*-matches.csv")))
    matches_files = [f for f in matches_files if "-doubles" not in f and "-mixed" not in f]

    speed_sum, speed_n = {}, {}
    fh_wins, bh_wins = {}, {}
    tour_of = {}

    for mf in matches_files:
        pf = mf.replace("-matches.csv", "-points.csv")
        if not os.path.exists(pf):
            continue
        try:
            matches = pd.read_csv(mf, usecols=["match_id", "match_num", "player1", "player2"], low_memory=False)
            points = pd.read_csv(pf, usecols=lambda c: c in _POINTS_COLS, low_memory=False)
        except (ValueError, pd.errors.EmptyDataError):
            continue
        if points.empty or "PointServer" not in points.columns:
            continue

        matches["match_num"] = pd.to_numeric(matches["match_num"], errors="coerce")
        matches = matches.dropna(subset=["match_num"])
        matches["tour"] = np.where(matches["match_num"] // 1000 == 1, "ATP",
                            np.where(matches["match_num"] // 1000 == 2, "WTA", None))
        matches = matches.dropna(subset=["tour"])

        merged = points.merge(matches, on="match_id", how="inner")
        if merged.empty:
            continue

        server_name = np.where(merged["PointServer"] == 1, merged["player1"],
                        np.where(merged["PointServer"] == 2, merged["player2"], None))
        merged["server_name"] = server_name

        if "Speed_KMH" in merged.columns:
            valid_speed = merged[(merged["Speed_KMH"].notna()) & (merged["Speed_KMH"] > 0) & (merged["server_name"].notna())]
            for name, tour, speed in zip(valid_speed["server_name"], valid_speed["tour"], valid_speed["Speed_KMH"]):
                speed_sum[name] = speed_sum.get(name, 0.0) + speed
                speed_n[name] = speed_n.get(name, 0) + 1
                tour_of[name] = tour

        if "P1Winner" in merged.columns and "WinnerShotType" in merged.columns:
            winner_name = np.where(merged["P1Winner"] == 1, merged["player1"],
                            np.where(merged["P2Winner"] == 1, merged["player2"], None))
            merged["winner_name"] = winner_name
            shot_winners = merged[(merged["winner_name"].notna()) & (merged["WinnerShotType"].isin(["F", "B"]))]
            for name, tour, shot in zip(shot_winners["winner_name"], shot_winners["tour"], shot_winners["WinnerShotType"]):
                tour_of[name] = tour
                if shot == "F":
                    fh_wins[name] = fh_wins.get(name, 0) + 1
                else:
                    bh_wins[name] = bh_wins.get(name, 0) + 1

    players = set(speed_sum) | set(fh_wins) | set(bh_wins)
    rows = []
    for p in players:
        fh, bh = fh_wins.get(p, 0), bh_wins.get(p, 0)
        rows.append({
            "player": p,
            "tour": tour_of.get(p),
            "avg_serve_speed_kmh": speed_sum[p] / speed_n[p] if p in speed_sum else np.nan,
            "serve_speed_sample_size": speed_n.get(p, 0),
            "fh_winners": fh,
            "bh_winners": bh,
            "fh_winner_share_slam": fh / (fh + bh) if (fh + bh) > 0 else np.nan,
        })
    return pd.DataFrame(rows)
