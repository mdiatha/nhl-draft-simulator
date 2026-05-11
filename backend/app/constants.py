"""
Central constants for the NHL Draft Simulator.

Update DRAFT_YEAR and STANDINGS_DATE each offseason — everything else derives from them.
"""
from __future__ import annotations

# ── Season configuration ──────────────────────────────────────────────────────

DRAFT_YEAR: int = 2026
STANDINGS_DATE: str = "2026-04-17"  # Last day of the 2025-26 regular season

# ── Official NHL lottery odds (non-playoff finish order, fixed by NHL rules) ──
# Keys are final standings positions 1–16; values are odds percentages.
LOTTERY_ODDS: dict[int, float] = {
    1: 18.5, 2: 13.5, 3: 11.5, 4: 9.5, 5: 8.5,
    6: 7.5,  7: 6.5,  8: 6.0,  9: 5.0, 10: 3.5,
    11: 3.0, 12: 2.5, 13: 2.0, 14: 1.5, 15: 1.0, 16: 0.5,
}

# ── League tier mapping ───────────────────────────────────────────────────────
# Maps league name → (tier: int, region: str)
# Used by ingestion to populate draft_league_tier on historical/prospect rows.

LEAGUE_TIER_MAP: dict[str, tuple[int, str]] = {
    # Tier 1 CHL (Canadian)
    "OHL":         (1, "CAN"),
    "WHL":         (1, "CAN"),
    "QMJHL":       (1, "CAN"),
    # Tier 1 US
    "NCAA":        (1, "USA"),
    "USHL":        (1, "USA"),
    # Tier 1 European top leagues
    "SHL":         (1, "EUR"),
    "Liiga":       (1, "EUR"),
    "KHL":         (1, "EUR"),
    "NLA":         (1, "EUR"),
    "Extraliga":   (1, "EUR"),
    # Tier 2
    "Allsvenskan": (2, "EUR"),
    "Mestis":      (2, "EUR"),
    "AHL":         (2, "USA"),
    "BCHL":        (2, "CAN"),
    # Tier 3
    "ECHL":        (3, "USA"),
    "NAHL":        (3, "USA"),
    "AJHL":        (3, "CAN"),
}


def get_league_tier(league_name: str | None) -> tuple[int | None, str | None]:
    """Return (tier, region) for a league name, or (None, None) if unknown."""
    if not league_name:
        return None, None
    for key, val in LEAGUE_TIER_MAP.items():
        if key.lower() in league_name.lower():
            return val
    return None, None


_NAT_ALIASES: dict[str, str] = {"CA": "CAN", "US": "USA", "SUI": "CHE"}


def nat_group(nationality: str) -> str:
    """
    Map a nationality code to a broad group used for GM tendency features.

    5 groups capture the meaningful distinctions GMs make in scouting pipelines,
    risk tolerance, and development paths — finer than the old CAN/USA/EUR split:

      CAN       — Canadian prospects (CHL, NCAA, USHL)
      USA       — American prospects
      NORDIC    — Sweden, Finland, Norway, Denmark
                  High-trust pipeline; contracts rarely an issue; strong skating
      SLAVIC    — Russia, Czech, Slovakia, Ukraine, Belarus, Latvia
                  Traditional hockey countries; historically higher variance
                  (KHL contract risk for Russians; Extraliga/Slovak league paths)
      EUR_OTHER — Germany, Switzerland, Austria, and any other country
    """
    nat = _NAT_ALIASES.get(nationality or "", nationality or "")
    if nat == "CAN":
        return "CAN"
    if nat == "USA":
        return "USA"
    if nat in ("SWE", "FIN", "NOR", "DNK"):
        return "NORDIC"
    if nat in ("RUS", "CZE", "SVK", "UKR", "BLR", "LVA"):
        return "SLAVIC"
    return "EUR_OTHER"


def infer_league_key(league: str, tier: int | None = None) -> str:
    """
    Normalize a raw league name string to a canonical key used in ML features
    and GM tendency profiles.

    The NHL Records API returns inconsistent strings for the same league
    (e.g. "SWEDEN", "SWEDEN-JR.", "SHL", "Swedish Hockey League" are all SHL).
    This function canonicalizes before mapping so the model learns per-league
    weights rather than per-string-variant weights.

    Return values match LEAGUE_KEYS in features.py:
      "OHL" | "WHL" | "QMJHL" | "USHL" | "NTDP" | "SHL" | "Liiga" | "KHL" | "other"
    """
    u = (league or "").upper()

    if "OHL" in u:
        return "OHL"
    if "WHL" in u:
        return "WHL"
    if "QMJHL" in u or "LHJMQ" in u:
        return "QMJHL"
    # NTDP before USHL — "NTDP - USHL" should map to NTDP
    if "NTDP" in u:
        return "NTDP"
    if "USHL" in u or "NCAA" in u or "HIGH-" in u or "BIG10" in u or "H-EAST" in u or "HIGH-MN" in u or "HIGH-MA" in u:
        return "USHL"
    # SHL / Swedish leagues — "SWEDEN", "SWEDEN-JR.", "SWEDEN-2", "SHL", "SWEDISH"
    if "SHL" in u or "SWEDEN" in u or "SWEDISH" in u:
        return "SHL"
    # Finnish leagues — "FINLAND", "FINLAND-JR.", "LIIGA", "MESTIS"
    if "LIIGA" in u or "FINLAND" in u or "MESTIS" in u:
        return "Liiga"
    # Russian/KHL — "KHL", "RUSSIA", "RUSSIA-JR.", "RUSSIA-2", "MHL"
    if "KHL" in u or "RUSSIA" in u or "MHL" in u:
        return "KHL"

    return "other"
