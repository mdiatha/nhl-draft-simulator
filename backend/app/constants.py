"""
Central constants for the NHL Draft Simulator.

Update DRAFT_YEAR and STANDINGS_DATE each offseason — everything else derives from them.
"""
from __future__ import annotations

# ── Season configuration ──────────────────────────────────────────────────────

DRAFT_YEAR: int = 2026
STANDINGS_DATE: str = "2026-04-18"  # Last day of the 2025-26 regular season

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
    Map a league name (and optional numeric tier fallback) to the string key
    used in GM tendency profiles and ML features.

    Return values: "tier1_CAN" | "tier1_USA" | "tier1_EUR" | "tier2" | "tier3"
    The numeric tier is used only when the league name doesn't match any known league.
    """
    u = (league or "").upper()
    if any(lg in u for lg in ("OHL", "WHL", "QMJHL")):
        return "tier1_CAN"
    if "NCAA" in u or "USHL" in u:
        return "tier1_USA"
    if any(lg in u for lg in ("SHL", "LIIGA", "KHL", "NLA", "EXTRALIGA")):
        return "tier1_EUR"
    # Fall back to numeric tier when league string is unrecognised
    if tier == 1:
        return "tier1_EUR"
    if tier == 3:
        return "tier3"
    return "tier2"
