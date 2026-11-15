"""
sgp_quote_collector.py — Automated FanDuel SGP quote collection.

Reads FanDuel's public pricing-only endpoint (no auth, no bets placed)
to collect per-archetype adjustment factors across line depths.

Usage:
    python sgp_quote_collector.py                # collect quotes for current NFL games
    python sgp_quote_collector.py --dry-run      # show what would be collected, don't write CSV

FanDuel API (public, read-only):
    Event listing:  GET  api.sportsbook.fanduel.com/sbapi/content-managed-page
    Event markets:  GET  api.sportsbook.fanduel.com/sbapi/event-page
    SGP pricing:    POST sib.nj.sportsbook.fanduel.com/api/sports/fixedodds/transactional/v1/implyBets
"""
from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

_WEB_KEY = "FhMFpcPWXMeyZxOx"
_SB_BASE = "https://sbapi.nj.sportsbook.fanduel.com/api"
_SGP_BASE = "https://sib.nj.sportsbook.fanduel.com"
_HEADERS = {
    "Origin": "https://sportsbook.fanduel.com",
    "Referer": "https://sportsbook.fanduel.com/",
    "x-sportsbook-region": "NJ",
    "Accept": "application/json",
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
}

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "sgp_quotes.csv"

# Rate limiting: minimum seconds between requests
_MIN_DELAY = 1.5

# Archetype definitions
ARCHETYPES = {
    "RB_QB_sameteam": {
        "stat_a": "rush_yds",
        "stat_b": "pass_yds",
        "same_player": False,
        "market_a_pattern": "Rushing Yds",
        "market_b_pattern": "Passing Yds",
        "priority": 1,
    },
    "QB_passyds_passtds": {
        "stat_a": "pass_yds",
        "stat_b": "pass_tds",
        "same_player": True,
        "market_a_pattern": "Passing Yds",
        "market_b_pattern": "Passing TDs",
        "priority": 2,
    },
    "WR_recyds_receptions": {
        "stat_a": "rec_yds",
        "stat_b": "receptions",
        "same_player": True,
        "market_a_pattern": "Receiving Yds",
        "market_b_pattern": "Receptions",
        "priority": 3,
    },
}

# Line depth categories
DEPTH_LABELS = {"chalky": 0, "moderate": 1, "aggressive": 2}


# ─── API Helpers ──────────────────────────────────────────────────────────────

def _get(path: str, params: dict | None = None) -> dict | None:
    """GET from FanDuel sportsbook API with rate limiting."""
    url = f"{_SB_BASE}/{path}"
    p = {"_ak": _WEB_KEY}
    if params:
        p.update(params)
    time.sleep(_MIN_DELAY)
    try:
        r = requests.get(url, params=p, headers=_HEADERS, timeout=15)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        log.error("GET %s failed: %s", path, e)
        return None
    except json.JSONDecodeError:
        log.error("GET %s returned non-JSON", path)
        return None


def _post_sgp(legs: list[dict]) -> dict | None:
    """POST to FanDuel SGP pricing endpoint."""
    url = f"{_SGP_BASE}/api/sports/fixedodds/transactional/v1/implyBets"
    body = {"betLegs": legs}
    params = {"_ak": _WEB_KEY}
    time.sleep(_MIN_DELAY)
    try:
        r = requests.post(url, json=body, params=params, headers=_HEADERS, timeout=15)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        log.error("SGP pricing failed: %s", e)
        return None
    except json.JSONDecodeError:
        log.error("SGP pricing returned non-JSON")
        return None


# ─── Event Discovery ──────────────────────────────────────────────────────────

def get_nfl_events() -> list[dict]:
    """Get current NFL game event IDs from FanDuel's content page."""
    data = _get("content-managed-page", {
        "page": "CUSTOM",
        "customPageId": "nfl",
    })
    if not data:
        log.error("Failed to fetch NFL content page")
        return []

    events = []
    att = data.get("attachments", {})
    for eid, ev in att.get("events", {}).items():
        name = ev.get("name", "")
        # Filter for actual game matchups (contain "@")
        if "@" not in name:
            continue
        events.append({
            "event_id": eid,
            "name": name,
            "open_date": ev.get("openDate", ""),
            "competition": ev.get("competitionId", ""),
        })

    # Sort by date (upcoming first)
    events.sort(key=lambda e: e["open_date"])
    log.info("Found %d NFL game events", len(events))
    return events


def get_event_markets(event_id: str) -> dict:
    """Get all markets for an event, finding SGP-eligible player prop markets.

    Loads passing-props, receiving-props, rushing-props tabs separately since
    the default event page only returns game-level markets.

    Returns dict of {player_name: {stat_type: [{marketId, selectionId, line, ...}]}}
    """
    players: dict[str, dict[str, list]] = {}

    # Load each prop tab
    for tab_slug in ["passing-props", "receiving-props", "rushing-props"]:
        data = _get("event-page", {"eventId": event_id, "tab": tab_slug})
        if not data:
            log.warning("Failed to fetch %s for event %s", tab_slug, event_id)
            continue

        att = data.get("attachments", {})
        markets_raw = att.get("markets", {})

        for mid, mkt in markets_raw.items():
            if not mkt.get("sgmMarket", False):
                continue

            mkt_name = mkt.get("marketName", "")

            stat_type = _classify_market_name(mkt_name)
            if not stat_type:
                continue

            # Extract player name from market name (format: "Player Name - Stat Type")
            player = _extract_player_name("", mkt_name)
            if not player:
                continue

            is_alt = "Alt " in mkt_name or "Alt " in mkt_name.lower()

            for runner in mkt.get("runners", []):
                runner_name = runner.get("runnerName", "")

                # For Alt markets: all runners are over-type (e.g. "250+ Yards")
                # For standard: only take "Over" selections
                if not is_alt and "Over" not in runner_name:
                    continue
                if is_alt and "Under" in runner_name:
                    continue

                sel_id = runner.get("selectionId")
                handicap = runner.get("handicap")
                # Alt markets have handicap=0, line is in runner name
                if is_alt or handicap == 0 or handicap is None:
                    line = _extract_line(runner_name)
                else:
                    line = handicap

                entry = {
                    "market_id": mid,
                    "selection_id": sel_id,
                    "market_name": mkt_name,
                    "runner_name": runner_name,
                    "line": line,
                    "stat_type": stat_type,
                }

                players.setdefault(player, {}).setdefault(stat_type, []).append(entry)

    log.info("Event %s: %d players with SGP-eligible props", event_id, len(players))
    return players


def _classify_market_name(name: str) -> str | None:
    """Classify a FanDuel market name into a canonical stat type."""
    n = name.lower()
    if "passing yd" in n and "rushing" not in n:
        return "pass_yds"
    if "passing td" in n:
        return "pass_tds"
    if "receiving yd" in n or "reception yd" in n:
        return "rec_yds"
    if "reception" in n and "yd" not in n:
        return "receptions"
    if "rushing yd" in n and "passing" not in n:
        return "rush_yds"
    # Alt markets
    if "alt passing yd" in n:
        return "pass_yds"
    if "alt passing td" in n:
        return "pass_tds"
    if "alt receiving yd" in n or "alt reception yd" in n:
        return "rec_yds"
    if "alt reception" in n and "yd" not in n:
        return "receptions"
    if "alt rushing yd" in n:
        return "rush_yds"
    return None


def _extract_player_name(runner_name: str, market_name: str) -> str | None:
    """Extract player name from runner/market names."""
    # Try market name first: "Patrick Mahomes - Passing Yds"
    if " - " in market_name:
        parts = market_name.split(" - ")
        if len(parts) >= 2:
            return parts[0].strip()

    # Runner name: "Over 275.5" doesn't have the player
    # Market name might be: "Player Name - Alt Passing Yds"
    for delim in [" - ", " Over ", " Under "]:
        if delim in runner_name:
            return runner_name.split(delim)[0].strip()

    return None


def _extract_line(runner_name: str) -> float | None:
    """Extract the numeric line from a runner name.

    Handles:
      - 'Over 275.5' / 'Under 275.5'
      - 'Patrick Mahomes 250+ Yards'
      - 'Patrick Mahomes 2+ Passing Touchdowns'
    """
    import re
    # Try "NNN+" format first (Alt markets)
    m = re.search(r'(\d+)\+', runner_name)
    if m:
        return float(m.group(1)) - 0.5  # "250+" means over 249.5
    # Standard "Over NNN.N"
    m = re.search(r'[\d]+\.?\d*', runner_name)
    if m:
        try:
            return float(m.group())
        except ValueError:
            pass
    return None


# ─── SGP Price Query ──────────────────────────────────────────────────────────

def get_sgp_price(market_id_a: str, sel_id_a: int,
                  market_id_b: str, sel_id_b: int) -> dict | None:
    """Query FanDuel SGP pricing for a 2-leg combination.

    Returns {price_a, price_b, sgp_price, adj_factor} or None.
    """
    legs = [
        {
            "legType": "SIMPLE_SELECTION",
            "betRunners": [{"runner": {"marketId": market_id_a, "selectionId": sel_id_a}}],
        },
        {
            "legType": "SIMPLE_SELECTION",
            "betRunners": [{"runner": {"marketId": market_id_b, "selectionId": sel_id_b}}],
        },
    ]

    resp = _post_sgp(legs)
    if not resp:
        return None

    # Parse response
    combos = resp.get("betCombinations", [])
    failures = resp.get("betFailures", [])

    if failures:
        fail_codes = [f.get("failureCode", "?") for f in failures]
        # INVALID_COMBINATION is expected for same-game legs when
        # queried as a regular parlay — the SGP combo is separate
        if "RUNNER_SUSPENDED" in fail_codes:
            log.warning("Runner suspended for market %s/%s", market_id_a, market_id_b)
            return None

    # Extract individual leg prices and SGP combo price
    price_a = None
    price_b = None
    sgp_price = None

    for combo in combos:
        is_sgm = combo.get("isSGM", False)
        # Get true decimal odds
        odds_obj = combo.get("winAvgOdds", {}).get("trueOdds", {}).get("decimalOdds", {})
        dec_odds = odds_obj.get("decimalOdds")

        if dec_odds is None:
            dec_odds = combo.get("averageOdds")

        if dec_odds is None:
            continue

        if is_sgm:
            sgp_price = float(dec_odds)
        else:
            # Single leg — identify which one via legCombinations.runners
            leg_combos = combo.get("legCombinations", [])
            if len(leg_combos) == 1:
                runners = leg_combos[0].get("runners", [])
                if runners:
                    r = runners[0]
                    mid = str(r.get("marketId", ""))
                    if mid == str(market_id_a):
                        price_a = float(dec_odds)
                    elif mid == str(market_id_b):
                        price_b = float(dec_odds)

    if price_a is None or price_b is None or sgp_price is None:
        log.warning("Incomplete pricing: a=%s b=%s sgp=%s", price_a, price_b, sgp_price)
        # If we got singles but no SGP, it might be because the combo isn't SGP-eligible
        if sgp_price is None and price_a and price_b:
            log.info("No SGP price returned — legs may not be SGP-eligible together")
        return None

    naive = price_a * price_b
    adj = sgp_price / naive if naive > 0 else None

    return {
        "price_a": round(price_a, 6),
        "price_b": round(price_b, 6),
        "sgp_price": round(sgp_price, 6),
        "naive_product": round(naive, 6),
        "adj_factor": round(adj, 6) if adj else None,
    }


# ─── Combo Builder ────────────────────────────────────────────────────────────

def _categorize_line_depth(selections: list[dict], stat_type: str) -> dict[str, list]:
    """Split selections into chalky/moderate/aggressive by line level.

    For O/U stats, lower lines are chalky (easier), higher are aggressive.
    For pass_tds, 0.5 is chalky, 1.5 is moderate, 2.5 is aggressive.
    """
    if not selections:
        return {}

    sorted_sels = sorted(selections, key=lambda s: s.get("line") or 0)

    if len(sorted_sels) <= 2:
        return {"moderate": sorted_sels}

    n = len(sorted_sels)
    third = max(1, n // 3)

    return {
        "chalky": sorted_sels[:third],
        "moderate": sorted_sels[third:2*third],
        "aggressive": sorted_sels[2*third:],
    }


def build_combos(players: dict, event_name: str) -> list[dict]:
    """Build SGP combos across archetypes × line depths for one event.

    Returns list of combo dicts ready for SGP pricing.
    """
    combos = []

    for arch_name, arch_def in sorted(ARCHETYPES.items(), key=lambda x: x[1]["priority"]):
        stat_a = arch_def["stat_a"]
        stat_b = arch_def["stat_b"]
        same_player = arch_def["same_player"]

        if same_player:
            # Same player: find players who have BOTH stats
            for player, stats in players.items():
                if stat_a not in stats or stat_b not in stats:
                    continue

                sels_a = stats[stat_a]
                sels_b = stats[stat_b]

                depths_a = _categorize_line_depth(sels_a, stat_a)
                depths_b = _categorize_line_depth(sels_b, stat_b)

                for depth in ["chalky", "moderate", "aggressive"]:
                    da = depths_a.get(depth, [])
                    db = depths_b.get(depth, [])
                    if not da or not db:
                        continue

                    # Take first selection from each depth
                    a = da[0]
                    b = db[0]

                    combos.append({
                        "archetype": arch_name,
                        "depth": depth,
                        "player_a": player,
                        "stat_a": stat_a,
                        "line_a": a.get("line"),
                        "market_id_a": a["market_id"],
                        "sel_id_a": a["selection_id"],
                        "runner_a": a.get("runner_name", ""),
                        "player_b": player,
                        "stat_b": stat_b,
                        "line_b": b.get("line"),
                        "market_id_b": b["market_id"],
                        "sel_id_b": b["selection_id"],
                        "runner_b": b.get("runner_name", ""),
                        "game": event_name,
                    })

        else:
            # Cross-player: find RB + QB on same team
            # We need team info — approximate by pairing rush_yds players with pass_yds players
            rush_players = [p for p, s in players.items() if stat_a in s]
            pass_players = [p for p, s in players.items() if stat_b in s]

            for rp in rush_players:
                for pp in pass_players:
                    if rp == pp:
                        continue  # Same player doesn't count for cross-player

                    sels_a = players[rp][stat_a]
                    sels_b = players[pp][stat_b]

                    depths_a = _categorize_line_depth(sels_a, stat_a)
                    depths_b = _categorize_line_depth(sels_b, stat_b)

                    for depth in ["chalky", "moderate", "aggressive"]:
                        da = depths_a.get(depth, [])
                        db = depths_b.get(depth, [])
                        if not da or not db:
                            continue

                        a = da[0]
                        b = db[0]

                        combos.append({
                            "archetype": arch_name,
                            "depth": depth,
                            "player_a": rp,
                            "stat_a": stat_a,
                            "line_a": a.get("line"),
                            "market_id_a": a["market_id"],
                            "sel_id_a": a["selection_id"],
                            "runner_a": a.get("runner_name", ""),
                            "player_b": pp,
                            "stat_b": stat_b,
                            "line_b": b.get("line"),
                            "market_id_b": b["market_id"],
                            "sel_id_b": b["selection_id"],
                            "runner_b": b.get("runner_name", ""),
                            "game": event_name,
                        })

    log.info("Built %d SGP combos for %s", len(combos), event_name)
    return combos


# ─── Collection Pipeline ─────────────────────────────────────────────────────

def collect_quotes(max_events: int = 4, dry_run: bool = False) -> list[dict]:
    """Main collection pipeline.

    1. Get NFL events
    2. For each event, get player prop markets
    3. Build archetype × depth combos
    4. Query SGP pricing for each combo (priority order: sanity check first)
    5. Write to CSV

    Returns list of collected quotes.
    """
    events = get_nfl_events()
    if not events:
        log.error("No NFL events found — check if games are available")
        return []

    # Limit to upcoming events
    events = events[:max_events]
    log.info("Processing %d events", len(events))

    all_combos = []
    for ev in events:
        eid = ev["event_id"]
        ename = ev["name"]
        log.info("Fetching markets for %s (%s)", ename, eid)

        players = get_event_markets(eid)
        if not players:
            log.warning("No player props found for %s", ename)
            continue

        combos = build_combos(players, ename)
        all_combos.extend(combos)

    # Sort by archetype priority (sanity check first)
    all_combos.sort(key=lambda c: ARCHETYPES[c["archetype"]]["priority"])

    log.info("Total combos to price: %d", len(all_combos))

    if dry_run:
        print(f"\n{'Archetype':<25} {'Depth':<10} {'Player A':<20} {'Stat A':<10} "
              f"{'Line A':>7} {'Player B':<20} {'Stat B':<10} {'Line B':>7} {'Game':<20}")
        print("-" * 140)
        for c in all_combos:
            print(f"  {c['archetype']:<23} {c['depth']:<10} {c['player_a']:<20} "
                  f"{c['stat_a']:<10} {c['line_a'] or '?':>7} "
                  f"{c['player_b']:<20} {c['stat_b']:<10} {c['line_b'] or '?':>7} "
                  f"{c['game'][:20]}")
        return all_combos

    # Price each combo
    quotes = []
    for i, combo in enumerate(all_combos):
        log.info("Pricing %d/%d: %s %s %s+%s",
                 i + 1, len(all_combos), combo["archetype"], combo["depth"],
                 combo["player_a"], combo["player_b"])

        pricing = get_sgp_price(
            combo["market_id_a"], combo["sel_id_a"],
            combo["market_id_b"], combo["sel_id_b"],
        )

        if pricing is None:
            log.warning("  -> No pricing returned, skipping")
            continue

        quote = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M"),
            "sport": "NFL",
            "game": combo["game"],
            "archetype": combo["archetype"],
            "depth": combo["depth"],
            "player_a": combo["player_a"],
            "stat_a": combo["stat_a"],
            "line_a": combo.get("line_a", ""),
            "price_a": pricing["price_a"],
            "player_b": combo["player_b"],
            "stat_b": combo["stat_b"],
            "line_b": combo.get("line_b", ""),
            "price_b": pricing["price_b"],
            "sgp_quoted_odds": pricing["sgp_price"],
            "adj_factor": pricing["adj_factor"],
            "book": "FanDuel",
            "notes": f"naive={pricing['naive_product']:.4f}",
        }
        quotes.append(quote)

        log.info("  -> price_a=%.4f price_b=%.4f sgp=%.4f adj=%.4f",
                 pricing["price_a"], pricing["price_b"],
                 pricing["sgp_price"], pricing["adj_factor"])

    if quotes:
        _write_csv(quotes)

    return quotes


def _write_csv(quotes: list[dict]):
    """Append quotes to CSV file, creating header if needed."""
    fieldnames = [
        "timestamp", "sport", "game", "archetype", "depth",
        "player_a", "stat_a", "line_a", "price_a",
        "player_b", "stat_b", "line_b", "price_b",
        "sgp_quoted_odds", "adj_factor", "book", "notes",
    ]

    needs_header = True
    if CSV_PATH.exists():
        with open(CSV_PATH) as f:
            header = f.readline().strip()
            if header and "depth" in header:
                needs_header = False

    mode = "w" if needs_header else "a"
    with open(CSV_PATH, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if needs_header:
            writer.writeheader()
        writer.writerows(quotes)

    log.info("Wrote %d quotes to %s", len(quotes), CSV_PATH)


# ─── Depth-Specific True K ──────────────────────────────────────────────────

def compute_depth_specific_k():
    """Compute true correlation factors at different line depths.

    Uses pooled 2023-2024 NFL game data with median-split methodology,
    but instead of splitting at the season median, splits at different
    percentiles to simulate chalky/moderate/aggressive lines.

    This answers: does the TRUE correlation vary by line depth?
    """
    try:
        import nfl_data_py as nfl
        import pandas as pd
        import numpy as np
    except ImportError:
        log.error("nfl_data_py not available for depth-specific k computation")
        return None

    log.info("Loading weekly data for depth-specific correlation...")
    weekly = nfl.import_weekly_data([2023, 2024])
    weekly = weekly[weekly["season_type"].isin(["REG", "POST"])].copy()

    results = {}

    # WR rec_yds ↔ receptions at different thresholds
    wr = weekly[(weekly["position"] == "WR") & (weekly["targets"].fillna(0) >= 2)].copy()
    wr_med = wr.groupby(["player_id", "season"]).agg(
        n_games=("week", "count"),
        med_rec_yds=("receiving_yards", "median"),
        med_recs=("receptions", "median"),
        p25_rec_yds=("receiving_yards", lambda x: x.quantile(0.25)),
        p75_rec_yds=("receiving_yards", lambda x: x.quantile(0.75)),
        p25_recs=("receptions", lambda x: x.quantile(0.25)),
        p75_recs=("receptions", lambda x: x.quantile(0.75)),
    ).reset_index()
    wr_med = wr_med[wr_med["n_games"] >= 6]
    wr2 = wr.merge(wr_med, on=["player_id", "season"], how="inner")

    for depth, (yds_col, rec_col) in [
        ("chalky", ("p25_rec_yds", "p25_recs")),
        ("moderate", ("med_rec_yds", "med_recs")),
        ("aggressive", ("p75_rec_yds", "p75_recs")),
    ]:
        wr2[f"yds_over_{depth}"] = (wr2["receiving_yards"] > wr2[yds_col]).astype(int)
        wr2[f"recs_over_{depth}"] = (wr2["receptions"] > wr2[rec_col]).astype(int)

        p_yds = wr2[f"yds_over_{depth}"].mean()
        p_recs = wr2[f"recs_over_{depth}"].mean()
        p_both = ((wr2[f"yds_over_{depth}"] == 1) & (wr2[f"recs_over_{depth}"] == 1)).mean()
        p_indep = p_yds * p_recs
        k = p_both / p_indep if p_indep > 0 else 1.0
        n = len(wr2)

        results[("WR_recyds_receptions", depth)] = {
            "k": round(k, 3), "n": n,
            "p_a": round(p_yds, 3), "p_b": round(p_recs, 3),
            "p_both": round(p_both, 3),
        }

    # QB pass_yds ↔ pass_tds at different thresholds
    qb = weekly[(weekly["position"] == "QB") & (weekly["attempts"].fillna(0) >= 10)].copy()
    qb_med = qb.groupby(["player_id", "season"]).agg(
        n_games=("week", "count"),
        med_pass_yds=("passing_yards", "median"),
        med_pass_tds=("passing_tds", "median"),
        p25_pass_yds=("passing_yards", lambda x: x.quantile(0.25)),
        p75_pass_yds=("passing_yards", lambda x: x.quantile(0.75)),
        p25_pass_tds=("passing_tds", lambda x: x.quantile(0.25)),
        p75_pass_tds=("passing_tds", lambda x: x.quantile(0.75)),
    ).reset_index()
    qb_med = qb_med[qb_med["n_games"] >= 6]
    qb2 = qb.merge(qb_med, on=["player_id", "season"], how="inner")

    for depth, (yds_col, tds_col) in [
        ("chalky", ("p25_pass_yds", "p25_pass_tds")),
        ("moderate", ("med_pass_yds", "med_pass_tds")),
        ("aggressive", ("p75_pass_yds", "p75_pass_tds")),
    ]:
        qb2[f"yds_over_{depth}"] = (qb2["passing_yards"] > qb2[yds_col]).astype(int)
        qb2[f"tds_over_{depth}"] = (qb2["passing_tds"] > qb2[tds_col]).astype(int)

        p_yds = qb2[f"yds_over_{depth}"].mean()
        p_tds = qb2[f"tds_over_{depth}"].mean()
        p_both = ((qb2[f"yds_over_{depth}"] == 1) & (qb2[f"tds_over_{depth}"] == 1)).mean()
        p_indep = p_yds * p_tds
        k = p_both / p_indep if p_indep > 0 else 1.0
        n = len(qb2)

        results[("QB_passyds_passtds", depth)] = {
            "k": round(k, 3), "n": n,
            "p_a": round(p_yds, 3), "p_b": round(p_tds, 3),
            "p_both": round(p_both, 3),
        }

    # RB rush ↔ QB pass at different thresholds (cross-player)
    rb = weekly[(weekly["position"] == "RB") & (weekly["carries"].fillna(0) >= 5)].copy()
    rb_med = rb.groupby(["player_id", "season"]).agg(
        n_games=("week", "count"),
        med_rush=("rushing_yards", "median"),
        p25_rush=("rushing_yards", lambda x: x.quantile(0.25)),
        p75_rush=("rushing_yards", lambda x: x.quantile(0.75)),
    ).reset_index()
    rb_med = rb_med[rb_med["n_games"] >= 6]
    rb2 = rb.merge(rb_med, on=["player_id", "season"], how="inner")

    # Get RB1 per team
    rb_carries = rb.groupby(["player_id", "recent_team", "season"]).agg(
        total_carries=("carries", "sum"), n_games=("week", "count")
    ).reset_index()
    rb_carries = rb_carries[rb_carries["n_games"] >= 6]
    rb_carries["rank"] = rb_carries.groupby(["recent_team", "season"])["total_carries"].rank(
        ascending=False, method="first"
    )
    rb1_ids = rb_carries[rb_carries["rank"] == 1][["player_id", "recent_team", "season"]]

    qb_g = qb2[["player_id", "recent_team", "season", "week",
                 "passing_yards", "med_pass_yds", "p25_pass_yds", "p75_pass_yds"]].copy()
    qb_g = qb_g.rename(columns={"player_id": "qb_id"})

    rb_g = rb2[["player_id", "recent_team", "season", "week",
                "rushing_yards", "med_rush", "p25_rush", "p75_rush"]].copy()
    rb_g = rb_g.rename(columns={"player_id": "rb_id"})
    rb_g = rb_g.merge(rb1_ids.rename(columns={"player_id": "rb_id"}),
                       on=["rb_id", "recent_team", "season"], how="inner")

    joined = qb_g.merge(rb_g, on=["recent_team", "season", "week"], how="inner")

    for depth, (qb_col, rb_col) in [
        ("chalky", ("p25_pass_yds", "p25_rush")),
        ("moderate", ("med_pass_yds", "med_rush")),
        ("aggressive", ("p75_pass_yds", "p75_rush")),
    ]:
        joined[f"qb_over_{depth}"] = (joined["passing_yards"] > joined[qb_col]).astype(int)
        joined[f"rb_over_{depth}"] = (joined["rushing_yards"] > joined[rb_col]).astype(int)

        p_qb = joined[f"qb_over_{depth}"].mean()
        p_rb = joined[f"rb_over_{depth}"].mean()
        p_both = ((joined[f"qb_over_{depth}"] == 1) & (joined[f"rb_over_{depth}"] == 1)).mean()
        p_indep = p_qb * p_rb
        k = p_both / p_indep if p_indep > 0 else 1.0
        n = len(joined)

        results[("RB_QB_sameteam", depth)] = {
            "k": round(k, 3), "n": n,
            "p_a": round(p_qb, 3), "p_b": round(p_rb, 3),
            "p_both": round(p_both, 3),
        }

    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Collect FanDuel SGP quotes")
    parser.add_argument("--dry-run", action="store_true", help="Show combos without pricing")
    parser.add_argument("--max-events", type=int, default=4, help="Max NFL events to process")
    parser.add_argument("--depth-k", action="store_true", help="Compute depth-specific true k")
    args = parser.parse_args()

    if args.depth_k:
        results = compute_depth_specific_k()
        if results:
            print(f"\n{'Archetype':<25} {'Depth':<12} {'True k':>8} {'N':>6} "
                  f"{'P(A)':>6} {'P(B)':>6} {'P(both)':>8} {'BE adj':>8}")
            print("-" * 85)
            for (arch, depth), v in sorted(results.items()):
                be = round(1.0 / v["k"], 3) if v["k"] > 0 else "---"
                print(f"  {arch:<23} {depth:<12} {v['k']:>8.3f} {v['n']:>6} "
                      f"{v['p_a']:>6.3f} {v['p_b']:>6.3f} {v['p_both']:>8.3f} {be:>8}")
        return

    quotes = collect_quotes(max_events=args.max_events, dry_run=args.dry_run)

    if not args.dry_run and quotes:
        # Run the analyzer
        print(f"\n{'='*70}")
        print("ANALYSIS OF COLLECTED QUOTES")
        print(f"{'='*70}")

        from sgp_quote_analyzer import print_report
        print_report()


if __name__ == "__main__":
    main()
