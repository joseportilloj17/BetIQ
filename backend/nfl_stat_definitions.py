"""
Prop-settlement stat definitions for NFL player props.

Single source of truth for how stats are measured — used by sim output
conversion, PBP actual computation, and walk-forward validation.

NFL official / prop settlement rules:
  pass_att   — throw attempts only (no sacks)
  pass_yds   — gross completion yards (no sack yardage subtracted)
  completions — completed passes
  receptions — per-player catches (= completions at team level)
  rush_att   — designed runs + scrambles + kneels
  rush_yds   — designed run + scramble + kneel yards
  pass_tds   — passing touchdowns
  rush_tds   — rushing touchdowns (includes scramble TDs)
"""


# ── Prop stat names (canonical) ─────────────────────────────────────────
PROP_STATS = [
    "pass_att", "completions", "pass_yds",
    "rush_att", "rush_yds",
    "pass_tds", "rush_tds",
    "sacks", "interceptions",
    "total_plays", "score",
]


def sim_stats_to_prop(stats: dict) -> dict:
    """Convert ``_compile_play_stats`` output to prop-settlement stats.

    Sim raw definitions:
      pass_att      = all dropbacks (passes + sacks + scrambles)
      rush_att      = designed runs only
      rush_yards    = designed run yards only
      scrambles     = QB scramble count
      scramble_yards = QB scramble yards
      kneels        = kneel count
      kneel_yards   = kneel yards (negative)

    Prop-settlement output:
      pass_att      = throw attempts (dropbacks − sacks − scrambles)
      rush_att      = designed + scrambles + kneels
      rush_yds      = designed + scramble + kneel yards
    """
    return {
        "pass_att":      (stats["pass_att"]
                          - stats["sacks"]
                          - stats["scrambles"]),
        "completions":   stats["completions"],
        "pass_yds":      stats["pass_yards"],       # already gross
        "rush_att":      (stats["rush_att"]
                          + stats["scrambles"]
                          + stats.get("kneels", 0)),
        "rush_yds":      (stats["rush_yards"]
                          + stats["scramble_yards"]
                          + stats.get("kneel_yards", 0)),
        "pass_tds":      stats["pass_tds"],
        "rush_tds":      stats["rush_tds"],
        "sacks":         stats["sacks"],
        "interceptions": stats["interceptions"],
        "total_plays":   stats["total_plays"],
    }


def sim_plays_to_prop(plays: list[dict]) -> dict:
    """Convert a sim play list directly to prop-settlement stats.

    Useful for per-draw conversion inside the allocation layer.
    """
    pass_att = 0
    completions = 0
    pass_yds = 0
    rush_att = 0
    rush_yds = 0
    pass_tds = 0
    rush_tds = 0
    sacks = 0
    interceptions = 0

    for p in plays:
        pr = p.get("play_result", "")

        if p.get("is_pass"):
            if p.get("sack"):
                sacks += 1
            elif p.get("scramble"):
                # Scramble → counts as rush
                rush_att += 1
                rush_yds += p["yards"]
                if p.get("touchdown"):
                    rush_tds += 1
            else:
                # Actual throw attempt
                pass_att += 1
                if p.get("completion"):
                    completions += 1
                    pass_yds += p["yards"]
                if p.get("interception"):
                    interceptions += 1
                if p.get("touchdown"):
                    pass_tds += 1
        elif pr == "kneel":
            rush_att += 1
            rush_yds += p["yards"]
        else:
            # Designed rush
            rush_att += 1
            rush_yds += p["yards"]
            if p.get("touchdown"):
                rush_tds += 1

    return {
        "pass_att":      pass_att,
        "completions":   completions,
        "pass_yds":      pass_yds,
        "rush_att":      rush_att,
        "rush_yds":      rush_yds,
        "pass_tds":      pass_tds,
        "rush_tds":      rush_tds,
        "sacks":         sacks,
        "interceptions": interceptions,
        "total_plays":   len(plays),
    }


def pbp_group_to_prop(gp) -> dict:
    """Convert a PBP DataFrame group (one team-game) to prop-settlement stats.

    ``gp`` should include play_type in ('pass', 'run', 'qb_kneel') with
    two_point_attempt != 1 filtered out.

    PBP field mapping:
      pass_att   = pass_attempt=1 AND sack!=1   (throw attempts)
      completions = complete_pass=1
      pass_yds   = yards_gained where complete_pass=1
      rush_att   = rush_attempt=1 (incl scrambles) + qb_kneel plays
      rush_yds   = yards_gained where rush_attempt=1 or qb_kneel
      sacks      = sack=1
    """
    pass_plays = gp[gp["pass_attempt"] == 1]
    throws = pass_plays[pass_plays["sack"] != 1]
    completions = throws[throws["complete_pass"] == 1]
    sacks = gp[gp["sack"] == 1]

    # Rushes: rush_attempt=1 already includes scrambles in PBP.
    # Kneels: play_type='qb_kneel' with rush_attempt=1.
    rushes = gp[gp["rush_attempt"] == 1]

    # Kneels that somehow have rush_attempt=0 (rare edge)
    kneels_extra = gp[
        (gp["play_type"] == "qb_kneel") & (gp["rush_attempt"] != 1)
    ]

    rush_att = len(rushes) + len(kneels_extra)
    rush_yds = (float(rushes["yards_gained"].sum())
                + float(kneels_extra["yards_gained"].sum())
                if len(kneels_extra) > 0
                else float(rushes["yards_gained"].sum()))

    pass_tds = int(throws[throws["touchdown"] == 1].shape[0])
    rush_tds = int(rushes[rushes["touchdown"] == 1].shape[0])

    return {
        "pass_att":      len(throws),
        "completions":   int(completions["complete_pass"].sum()),
        "pass_yds":      float(completions["yards_gained"].sum()),
        "rush_att":      rush_att,
        "rush_yds":      rush_yds,
        "pass_tds":      pass_tds,
        "rush_tds":      rush_tds,
        "sacks":         len(sacks),
        "interceptions": int(pass_plays["interception"].sum()),
        "total_plays":   len(gp),
    }
