"""GTA VI trophy rarity forecaster for make.fun's gta6-trophy-rarity-forecast-01.

Model (written from scratch for this entry -- not the house baseline):

  * Per trophy, fit a weighted least-squares line of logit(earn rate) against
    x = ln(1 + hours since release 2026-11-19T00:00Z).  Log-time is used
    because trophy earning is fast early (launch-night players rush the story)
    and saturates: a line in log-time is a saturating S-curve in real time.
    Reads are recency-weighted (18 h half-life) so the fit tracks the latest
    refresh epoch, and each trophy's slope is shrunk toward its tier's median
    slope, so noisy trophies borrow strength from their peers.
  * Stale-page correction: Exophase refreshes on its own schedule, so the
    rates a case shows are as of its last refresh, not the checkpoint.  When
    the page has sat frozen (identical reads, players-tracked unchanged), the
    true tracked count has kept growing while earners grew more slowly, so
    the true rates at the checkpoint sit below the shown ones.  The correction
    estimates the tracked-growth rate g (from the history's own
    players_tracked when it moves, else a launch-night prior that decays with
    time since release) and dilutes each trophy's current level by
    exp((e - g) * stale_time), where e is that trophy's earner-growth rate
    (fitted, floored by a class prior: prologue/chapter trophies are earned
    almost as fast as new players arrive, grind trophies barely at all) and
    stale_time runs from the last refresh to the next expected refresh after
    the checkpoint.  The fitted trend then extrapolates from the corrected
    level to the target.
  * Domain caps: the median at the target is capped by trophy class, read off
    the trophy's own name/description (chapter trophies climb to tens of
    percent in week one; 250 stunts / 100% completion / platinum stay under a
    few percent).  Caps only bind on runaway extrapolations.
  * Spread: a logit-scale sigma combining a random-walk term (0.5 per
    sqrt-day, as the house baseline), the line fit's prediction variance
    (floored, so a perfectly flat history is never treated as certain) and a
    staleness term that widens while the page is frozen.

Run: python3 main.py reads $MAKEFUN_INPUT_DIR (default /input) and writes
one <id>.out per *.in found to $MAKEFUN_OUTPUT_DIR (default /output).
Deterministic: no randomness, no wall-clock dependence, sorted file order.
Python 3.13, standard library only.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

RELEASE = datetime(2026, 11, 19, 0, 0, tzinfo=timezone.utc)
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# Random-walk spread, logit units per sqrt(day).
SIGMA_RW = 0.5
# Floor on the fit's prediction variance (logit^2): flat history != certainty.
PRED_VAR_FLOOR = 0.09
PRED_VAR_CAP = 2.25
# Staleness spread coefficient: extra logit sigma per log2(1 + stale hours).
STALE_SPREAD = 0.35
# Recency half-life for the trend fit, hours.
HALF_LIFE_H = 18.0
# Launch-night tracked-growth prior, per hour, decaying with time since release.
G_PRIOR_0 = 0.35
G_PRIOR_DECAY_H = 36.0
# Expected gap between Exophase refreshes, hours (early launch night).
REFRESH_GAP_0 = 4.7
# Slope clamp, logit per unit of ln(1+hours).
SLOPE_MIN, SLOPE_MAX = -1.0, 3.0


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def clamp_rate(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_rate(pct)
    return math.log(p / (100.0 - p))


def inv_logit(l: float) -> float:
    l = max(-30.0, min(30.0, l))
    return 100.0 / (1.0 + math.exp(-l))


def round4(x: float):
    """Round to 4 decimals, half up (matches JavaScript Math.round)."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285984469534e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    p_low, p_high = 0.02425, 1.0 - 0.02425
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
        ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)


def wls(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (a, b, resid_var, sxx, xbar, n)."""
    sw = sum(ws)
    xbar = sum(w * x for x, w in zip(xs, ws)) / sw
    ybar = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - xbar) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return ybar, 0.0, 0.0, 0.0, xbar, len(xs)
    sxy = sum(w * (x - xbar) * (y - ybar) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    a = ybar - b * xbar
    n = len(xs)
    if n > 2:
        resid_var = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / max(1, n - 2)
    else:
        resid_var = 0.0
    return a, b, resid_var, sxx, xbar, n


# ---------------------------------------------------------------------------
# Trophy classification from the trophy's own name/description.
# Returns (median_cap_percent_at_target, earner_speed_fraction).
# earner_speed_fraction: how fast earners grow relative to tracked players
# during a frozen page (1.0 = earners keep up fully, 0.0 = earners frozen).
# ---------------------------------------------------------------------------

def classify(trophy):
    desc = (trophy.get("description") or "").lower()
    name = (trophy.get("name") or "").lower()
    text = name + " " + desc
    tier = trophy.get("tier", "bronze")
    if trophy.get("id") == 1 or "platinum" in text:
        return 1.0, 0.0
    if "100% completion" in desc:
        return 1.0, 0.0
    if "gone fishing" in desc:
        return 99.0, 1.0
    if "complete chapter 1" in desc:
        return 75.0, 0.9
    if "complete chapter 2" in desc:
        return 65.0, 0.8
    if "complete chapter 3" in desc:
        return 55.0, 0.7
    if "complete chapter 4" in desc:
        return 50.0, 0.6
    if "complete the final mission" in desc and "for " not in desc:
        return 20.0, 0.5
    if "complete the final mission for" in desc:
        return 20.0, 0.4
    if "250 stunts" in desc:
        return 3.0, 0.0
    if "100 locations" in desc:
        return 3.0, 0.0
    if "collectible card sets" in desc:
        return 4.0, 0.0
    if "gold medal in all" in desc:
        return 5.0, 0.0
    if "meditate at every spot" in desc:
        return 6.0, 0.0
    if "all garage properties" in desc:
        return 5.0, 0.0
    if "8 mission strands" in desc:
        return 10.0, 0.1
    if "50 challenges" in desc:
        return 8.0, 0.1
    if "5 bank vaults" in desc:
        return 8.0, 0.1
    if "win 20 different races" in desc:
        return 12.0, 0.15
    if "drone race at every location" in desc:
        return 8.0, 0.1
    if "laundering marked bills" in desc:
        return 10.0, 0.1
    if "explore all of" in desc:
        return 15.0, 0.2
    if "discover 100 vehicles" in desc:
        return 15.0, 0.2
    if "register 10 vehicles" in desc:
        return 15.0, 0.2
    if desc.startswith("complete "):
        return 40.0, 0.5
    if tier == "gold":
        return 20.0, 0.3
    if tier == "silver":
        return 25.0, 0.25
    return 30.0, 0.35


def forecast_case(case, name):
    trophies = case["trophies"]
    history = case.get("history") or []
    levels = case.get("levels") or LEVELS
    z = [norm_ppf(p) for p in levels]
    horizon_days = case["horizon_minutes"] / 1440.0
    at = parse_time(case["at"])
    t_target_h = (parse_time(case["target"]) - RELEASE).total_seconds() / 3600.0
    x_target = math.log1p(max(0.0, t_target_h))

    if not history:
        # No data at all: flat 1% medians with a wide spread, still valid.
        return {"trophies": [{"id": t["id"], "q": [round4(inv_logit(logit(1.0) + zi * 1.5)) for zi in z]}
                             for t in trophies]}

    # --- Per-read time axis and weights -------------------------------------
    read_t = []          # hours since release
    read_x = []          # ln(1 + hours)
    for r in history:
        h = (parse_time(r["at"]) - RELEASE).total_seconds() / 3600.0
        read_t.append(h)
        read_x.append(math.log1p(max(0.0, h)))
    t_last = read_t[-1]
    x_last = read_x[-1]
    weights = [0.5 ** ((t_last - t) / HALF_LIFE_H) for t in read_t]
    n_reads = len(history)
    n_distinct_x = len(set(round(x, 9) for x in read_x))

    # --- When did the page last change? (staleness of the last read) --------
    last_change_idx = 0
    for i in range(1, n_reads):
        if history[i]["rates"] != history[i - 1]["rates"] or \
                history[i].get("players_tracked") != history[i - 1].get("players_tracked"):
            last_change_idx = i
    freeze_h = max(0.0, (at - parse_time(history[last_change_idx]["at"])).total_seconds() / 3600.0)
    refresh_gap = max(1.0, min(6.0, REFRESH_GAP_0 * math.exp(-t_last / 48.0)))
    # A refresh lands before the target only with some probability: if the
    # last refresh was freeze_h ago and gaps run ~refresh_gap, the chance it
    # lands inside the horizon is horizon / remaining-gap (1 once overdue).
    # The dilution only materialises when that refresh lands, so weight the
    # stale time by that probability; the spread still covers the other branch.
    horizon_h = case["horizon_minutes"] / 60.0
    remaining_gap = refresh_gap - freeze_h
    if remaining_gap <= 0.5:
        p_refresh = 1.0
    else:
        p_refresh = min(1.0, horizon_h / remaining_gap)
    stale_time = p_refresh * (freeze_h + min(horizon_h, refresh_gap))

    # --- Tracked-players growth rate g (per hour) ----------------------------
    n_tracked = [r.get("players_tracked") for r in history]
    g = None
    if len(set(n_tracked)) > 1 and all(isinstance(v, (int, float)) and v > 0 for v in n_tracked):
        log_n = [math.log(v) for v in n_tracked]
        _, k_n, _, _, _, _ = wls(read_x, log_n, weights)
        g = max(0.0, k_n / (1.0 + t_last))
    if g is None:
        # Page never moved: fall back to the launch-night adoption prior.
        g = G_PRIOR_0 * math.exp(-t_last / G_PRIOR_DECAY_H)

    # --- First pass: per-trophy trend fits ----------------------------------
    fits = []
    for i, trophy in enumerate(trophies):
        ys, xs, ws = [], [], []
        for j, r in enumerate(history):
            rate = r["rates"][i] if i < len(r["rates"]) else None
            if rate is None:
                continue
            xs.append(read_x[j])
            ys.append(logit(rate))
            ws.append(weights[j])
        if len(xs) < 3 or len(set(round(x, 9) for x in xs)) < 2 or max(ys) - min(ys) < 1e-9:
            a = ys[-1] if ys else logit(0.0)
            fits.append({"a": a, "b": 0.0, "resid_var": 0.0, "sxx": 0.0,
                         "xbar": xs[-1] if xs else x_last, "n": len(xs), "ys": ys, "xs": xs})
            continue
        a, b, resid_var, sxx, xbar, n = wls(xs, ys, ws)
        b = max(SLOPE_MIN, min(SLOPE_MAX, b))
        fits.append({"a": a, "b": b, "resid_var": resid_var, "sxx": sxx,
                     "xbar": xbar, "n": n, "ys": ys, "xs": xs})

    # Tier-median slopes for shrinkage.
    tier_slopes = {}
    for tier in ("platinum", "gold", "silver", "bronze"):
        bs = sorted(f["b"] for f, t in zip(fits, trophies) if t.get("tier") == tier and f["n"] >= 3)
        if bs:
            tier_slopes[tier] = bs[len(bs) // 2]

    # Earner-growth rate per trophy (log earners vs time), for the freeze fix.
    def earner_rate(i):
        es, xs = [], []
        for j, r in enumerate(history):
            rate = r["rates"][i] if i < len(r["rates"]) else None
            n = r.get("players_tracked")
            if rate is None or not isinstance(n, (int, float)) or n <= 0 or rate <= 0:
                continue
            es.append(math.log(max(1e-9, rate * n / 100.0)))
            xs.append(read_x[j])
        if len(xs) < 3 or len(set(round(x, 9) for x in xs)) < 2:
            return 0.0
        _, k_e, _, _, _, _ = wls(xs, es, [weights[j] for j in range(len(history))
                                          if history[j]["rates"][i] is not None])
        return max(0.0, k_e / (1.0 + t_last))

    # --- Second pass: medians, caps, spreads --------------------------------
    rows = []
    for i, trophy in enumerate(trophies):
        f = fits[i]
        cap, speed = classify(trophy)
        b_tier = tier_slopes.get(trophy.get("tier"), 0.0)
        w_shrink = f["n"] / (f["n"] + 4.0) if f["n"] else 0.0
        b = w_shrink * f["b"] + (1.0 - w_shrink) * b_tier
        b = max(SLOPE_MIN, min(SLOPE_MAX, b))

        # Current level from the fit, corrected for the frozen page.
        level_now = inv_logit(f["a"] + b * x_last)
        if level_now >= 99.0:
            mult = 1.0
        else:
            e = max(earner_rate(i), speed * g)
            mult = math.exp((e - g) * stale_time)
        level_now = clamp_rate(level_now * mult)

        # Extrapolate the (shrunk) trend from the corrected level.
        m_logit = logit(level_now) + b * (x_target - x_last)
        m_rate = min(cap, inv_logit(m_logit))
        m_rate = clamp_rate(m_rate)
        m_logit = logit(m_rate)

        # Spread: random walk + fit prediction variance + staleness.
        if f["n"] > 2 and f["sxx"] > 1e-12:
            pred_var = f["resid_var"] * (1.0 + 1.0 / f["n"] +
                                          (x_target - f["xbar"]) ** 2 / f["sxx"])
        else:
            pred_var = 0.0
        pred_var = max(PRED_VAR_FLOOR, min(PRED_VAR_CAP, pred_var))
        stale_var = (STALE_SPREAD * math.log2(1.0 + stale_time)) ** 2
        spread = math.sqrt(SIGMA_RW ** 2 * horizon_days + pred_var + stale_var)
        spread = min(spread, 2.0)

        qs = [round4(min(100.0, max(0.0, inv_logit(m_logit + zi * spread)))) for zi in z]
        for k in range(1, len(qs)):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
