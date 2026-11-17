"""GTA VI launch-day Twitch CCU forecaster (Python 3.13, standard library only).

Forecasts, from the Twitch viewer count of Grand Theft Auto VI at the end of
the first hour of its release day (2026-11-19 UTC), the viewer count in each
of hours 2..24 as five quantiles (5%, 25%, 50%, 75%, 95%).

Method: kernel-weighted comparable-launch log-ratio regression.
Each case ships with the full hourly Twitch curves of 38-42 past big game
launches ("comparables"), from 14 days before to 3 days after their release
day. For a comparable launch c, let L_c[h] = ln(viewers_h / viewers_1) be its
release-day log-ratio curve. The forecast for the target is built per hour
from the donors' curves:

  * Donor weight: Gaussian kernel on log hour-1 size (bandwidth 1.2 in log
    units), times tier weights (early-access-contaminated 0.35, weak
    0.25), a console-only boost (1.3), and an unlock-shape weight
    (rolling-midnight donors 2.5, simultaneous-global donors 0.3 when the
    target is a rolling launch). A kernel on the pre-release surge was
    tested in leave-one-out and dropped: it did not help once the ramp
    feature below was in, and it slightly hurt the large-launch regime.
  * Centre: weighted ridge regression of L_c[h] on two features --
    log-size difference to the target and the pre-checkpoint ramp
    ln(hour-1 / viewers one hour before day start) -- evaluated at the
    target. The size slope is the dominant signal: bigger hour-1 launches
    have flatter day curves (correlation about -0.76 between log hour-1
    and log peak ratio across the donors).
  * Median re-anchor (v5): the ridge centre carries a small systematic
    bias (the pooled standardised residuals have median about -0.15, i.e.
    the centre sits high). Per hour, b(h) = n_h * median(residuals) /
    (n_h + 10) is added to the centre, where the residuals are the
    UNwinsorised donor residuals after the centre fit. Leave-one-out:
    this alone cuts mean pinball loss 0.395 -> 0.380.
  * Spread: weighted standard deviation of the UNwinsorised regression
    residuals (v4 winsorised them; unwinsorised is worth a further
    ~0.002), scaled by 1.05 (the 1.00-1.10 plateau is flat; 1.05 is its
    centre), floored at 0.30 log units and capped at 1.20. Both centre
    and spread are smoothed across neighbouring hours (0.25/0.5/0.25).
  * Quantiles: p1 * exp(centre_h + z_tau * sigma_h), z the standard normal
    quantiles. Empirical asymmetric quantile constants were tried in
    leave-one-out (pooled + per-hour shrunk, kappa 4/8/16, pure and
    70/30 Gaussian blend, in-sample and honest leave-one-donor-out
    residuals) and did NOT beat the Gaussian table -- the pooled
    empirical constants are nearly Gaussian anyway
    ([-1.54, -0.56, -0.15, 0.56, 1.69] vs z), so the shape was never the
    leak; the centre bias was. A quadratic size feature (dx^2, clamped)
    and a big-launch donor regime (top 10-12 donors, bw 0.65) were also
    tried and both HURT leave-one-out (quad +0.019, big regime +0.001
    overall / +0.021 on big launches, driven by a threshold cliff at
    h1 ~ 10k: one target 0.317 -> 0.904), so neither is in v5.

The target's own pre-release history (history_hourly) supplies its ramp
feature; the Grand Theft Auto VI category's own pre-release
history, shipped as a comparable, is excluded as a donor (it is not a
launch). Any failure (missing data, degenerate donors) falls back to the
house persistence baseline so the case is always valid. The program is
fully deterministic: same case file -> same output bytes. Runtime is far
under the 8 s cap (milliseconds per case, standard library only).

Validation (leave-one-out over the 38 comparable launches in the kit's
practice cases, each launch forecast from the other 37):
  house persistence baseline: mean pinball loss 0.930 (median 0.325)
  v4 model:                    mean 0.395 (median 0.274)
  v5 model:                    mean 0.380 (median 0.269)
  on launches with hour-1 >= 10k viewers (the regime of the practice
  targets and of GTA VI itself): v4 mean 0.255, median 0.183, 90%
  coverage 0.86; v5 mean 0.250, median 0.194, 90% coverage 0.89.
  Paired improvement v5 vs v4: -0.015 mean loss (paired SE 0.005),
  clear of the ~0.01 noise floor. The practice targets are all large
  console rolling-midnight launches (hour-1 23k-270k viewers), the regime
  where the model is strongest; the final case additionally includes
  those four launches as donors.
"""

import json
import math
import os
import statistics
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import. (This entry is a single file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# --- tuned constants (see module docstring; chosen on the leave-one-out
# plateau, not on any single best cell) ---
BW_SIZE = 1.2        # Gaussian kernel bandwidth on log hour-1 size
RIDGE = 2.0          # ridge strength on the regression slopes
WINSOR_SD = 2.0      # winsorise donor log-ratios at median +/- this sd
                     # (centre fit ONLY -- spread/quantiles use raw residuals)
SIG_SCALE = 1.05     # residual-spread calibration multiplier (1.00-1.10
                     # plateau is flat on LOO pinball; 1.05 is its centre)
SIG_FLOOR = 0.30     # minimum log-spread per hour
SIG_CAP = 1.20       # maximum log-spread per hour
ANCHOR_DENOM = 10    # median re-anchor shrinkage: b = n*median/(n+10)
W_CONSOLE = 1.3      # weight boost for console-only donors
W_EARLY_ACCESS = 0.35
W_WEAK = 0.25
W_ROLLING = 2.5      # rolling-midnight donors, for a rolling target
W_SIMULT = 0.3       # simultaneous-global donors, for a rolling target


def parse_ts(s):
    return datetime.fromisoformat(str(s).replace("Z", "+00:00"))


def round2(x):
    """Round to 2 decimals, half up, like JavaScript Math.round."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if float(value).is_integer() else value


def unlock_class(text):
    t = (text or "").lower()
    if "simultaneous" in t or "single global" in t:
        return "sim"
    if "rolling" in t or "midnight local" in t:
        return "roll"
    return "unk"


def tier_of(text):
    t = (text or "").lower()
    if "target-history" in t:
        return "target"
    if "early-access-contaminated" in t:
        return "ea"
    if "tier: weak" in t:
        return "weak"
    return "ok"


def console_only(platforms):
    return "pc" not in (platforms or "").lower()


def wquantile(pairs, tau):
    """Weighted quantile of (value, weight) pairs."""
    s = sorted(pairs)
    tot = sum(w for _, w in s)
    if tot <= 0:
        return s[len(s) // 2][0]
    cut = tau * tot
    acc = 0.0
    for v, w in s:
        acc += w
        if acc >= cut:
            return v
    return s[-1][0]


def wmean(pairs):
    tot = sum(w for _, w in pairs)
    return sum(v * w for v, w in pairs) / tot if tot > 0 else 0.0


def wstd(pairs, center=0.0):
    tot = sum(w for _, w in pairs)
    if tot <= 0:
        return 0.5
    return math.sqrt(sum(w * (v - center) ** 2 for v, w in pairs) / tot)


def solve_ridge(xs, ys, ws, lam):
    """Weighted ridge regression y ~ a + b.x (slopes shrunk, intercept free).

    xs: list of feature vectors, ys: targets, ws: weights. Returns the
    coefficient list [a, b1, b2, ...]. Solves the normal equations by
    Gaussian elimination; the system is tiny (<= 3 unknowns).
    """
    k = len(xs[0]) + 1
    m = [[0.0] * k for _ in range(k)]
    v = [0.0] * k
    for xi, yi, wi in zip(xs, ys, ws):
        row = [1.0] + list(xi)
        for a in range(k):
            v[a] += wi * row[a] * yi
            for b in range(k):
                m[a][b] += wi * row[a] * row[b]
    for a in range(1, k):
        m[a][a] += lam
    aug = [m[i][:] + [v[i]] for i in range(k)]
    for col in range(k):
        piv = max(range(col, k), key=lambda r: abs(aug[r][col]))
        if abs(aug[piv][col]) < 1e-12:
            continue
        aug[col], aug[piv] = aug[piv], aug[col]
        for r in range(k):
            if r != col and aug[r][col] != 0:
                f = aug[r][col] / aug[col][col]
                for cc in range(col, k + 1):
                    aug[r][cc] -= f * aug[col][cc]
    return [aug[i][k] / aug[i][i] if abs(aug[i][i]) > 1e-12 else 0.0
            for i in range(k)]


def release_day_values(comp):
    """The comparable's 24 release-day hourly viewer counts (hours 1..24).

    Missing hours are log-linearly interpolated; returns None when hour 1
    is missing or more than 6 of the 24 hours are missing.
    """
    ds = parse_ts(comp["day_start"])
    by_hour = {}
    for row in comp.get("hourly") or []:
        try:
            t = parse_ts(row["from"])
        except (KeyError, TypeError, ValueError):
            continue
        h = int((t - ds).total_seconds() // 3600) + 1
        # A zero-viewer hour is real data (a dead category hour), not a
        # missing row: keep it, clamped to 1.0 below. Only None is missing.
        if 1 <= h <= 24 and row.get("viewers") is not None:
            by_hour[h] = max(1.0, float(row["viewers"]))
    if 1 not in by_hour or len(by_hour) < 18:
        return None
    vals = []
    for h in range(1, 25):
        if h in by_hour:
            vals.append(by_hour[h])
        else:
            lo = max([k for k in by_hour if k < h], default=None)
            hi = min([k for k in by_hour if k > h], default=None)
            if lo is not None and hi is not None:
                lv, hv = math.log(by_hour[lo]), math.log(by_hour[hi])
                vals.append(math.exp(lv + (hv - lv) * (h - lo) / (hi - lo)))
            elif lo is not None:
                vals.append(by_hour[lo])
            elif hi is not None:
                vals.append(by_hour[hi])
            else:
                return None
    return vals


def pre_features(hourly, day_start):
    """Surge and ramp from an hourly series around a release day start.

    surge = ln(hour-1 viewers / median viewers in the 24 h before the day
    start); ramp = ln(hour-1 / viewers one hour before the day start).
    Missing points fall back to the nearest available value, then to 0.
    """
    ds = parse_ts(day_start)
    pre24, at_minus1 = [], None
    v1 = None
    for row in hourly or []:
        try:
            t = parse_ts(row["from"])
        except (KeyError, TypeError, ValueError):
            continue
        dh = (t - ds).total_seconds() / 3600
        viewers = row.get("viewers")
        if viewers is None:
            continue
        if -24 <= dh < 0:
            pre24.append(float(viewers))
        # Timestamp tolerance (30 s): hourly rows should sit exactly on
        # -1h/0h, but never rely on exact float equality to find them.
        if abs(dh + 1) < 1 / 120:
            at_minus1 = float(viewers)
        if abs(dh) < 1 / 120:
            v1 = float(viewers)
    if v1 is None or v1 <= 0:
        return 0.0, 0.0
    med = statistics.median(pre24) if pre24 else v1
    surge = math.log(v1 / max(1.0, med))
    ramp = math.log(v1 / max(1.0, at_minus1)) if at_minus1 else 0.0
    return surge, ramp


def build_donors(case):
    """Parse the case's comparables into donor records, sorted by name."""
    donors = []
    for comp in case.get("comparables") or []:
        if tier_of(comp.get("unlock", "")) == "target":
            continue  # the target's own pre-release history: not a launch
        vals = release_day_values(comp)
        if vals is None:
            continue
        surge, ramp = pre_features(comp.get("hourly") or [],
                                   comp["day_start"])
        donors.append({
            "name": comp.get("name", ""),
            "vals": vals,
            "tier": tier_of(comp.get("unlock", "")),
            "console": console_only(comp.get("platforms", "")),
            "uclass": unlock_class(comp.get("unlock", "")),
            "lv1": math.log(vals[0]),
            "surge": surge,
            "ramp": ramp,
        })
    donors.sort(key=lambda d: d["name"])
    return donors


def donor_weight(d, lp1, target_uclass):
    w = math.exp(-0.5 * ((d["lv1"] - lp1) / BW_SIZE) ** 2)
    if d["tier"] == "ea":
        w *= W_EARLY_ACCESS
    elif d["tier"] == "weak":
        w *= W_WEAK
    if d["console"]:
        w *= W_CONSOLE
    if target_uclass == "roll":
        if d["uclass"] == "roll":
            w *= W_ROLLING
        elif d["uclass"] == "sim":
            w *= W_SIMULT
    elif target_uclass == "sim":
        if d["uclass"] == "sim":
            w *= W_ROLLING
        elif d["uclass"] == "roll":
            w *= W_SIMULT
    return w


def baseline_quantiles(p1, h):
    """House persistence baseline: flat at hour 1, widening spread."""
    return [round2(p1 * math.exp(z * 0.15 * math.sqrt(h - 1))) for z in Z]


def forecast_case(case):
    known = next((k for k in case.get("known") or [] if k.get("hour") == 1),
                 None)
    if known is None:
        raise ValueError("case has no hour-1 print")
    p1 = max(1.0, float(known["viewers"]))
    # Always emit exactly hours 2..24, regardless of the case's forecast
    # rows (a case shipping any other rows must not change the output shape).
    hours = list(range(2, 25))

    # Target features from its own pre-checkpoint history (the ramp: how
    # hour 1 compares with the hour before the release day started).
    _, ramp_t = pre_features(case.get("history_hourly") or [],
                            case["day_start"])
    target_uclass = unlock_class(case.get("notes", ""))

    donors = build_donors(case)
    if len(donors) < 5:
        return {"hours": [{"hour": h, "q": baseline_quantiles(p1, h)}
                          for h in hours]}

    lp1 = math.log(p1)
    centers, sigmas = {}, {}
    for h in hours:
        xs, ys_raw, ws = [], [], []
        for d in donors:
            ws.append(donor_weight(d, lp1, target_uclass))
            xs.append([d["lv1"] - lp1, d["ramp"] - ramp_t])
            ys_raw.append(math.log(d["vals"][h - 1] / d["vals"][0]))
        # Winsorise the donor log-ratios so one explosive launch day
        # cannot drag the centre fit. The spread and the median
        # re-anchor below use the UNwinsorised residuals: winsorising
        # them would understate how much launch days actually vary.
        med = wquantile(list(zip(ys_raw, ws)), 0.5)
        sd = wstd([(y - med, w) for y, w in zip(ys_raw, ws)], 0.0) or 0.5
        lo, hi = med - WINSOR_SD * sd, med + WINSOR_SD * sd
        ys = [min(max(y, lo), hi) for y in ys_raw]
        co = solve_ridge(xs, ys, ws, RIDGE)
        pred = [co[0] + sum(c * x for c, x in zip(co[1:], xi)) for xi in xs]
        # Median re-anchor: the ridge centre sits systematically high
        # (median donor residual is negative). Shift the centre by the
        # shrunk weighted median residual, then recompute residuals.
        resid = [(y - p, w) for y, p, w in zip(ys_raw, pred, ws)]
        n_h = len(resid)
        bias = n_h * wquantile(resid, 0.5) / (n_h + ANCHOR_DENOM)
        centers[h] = co[0] + bias
        resid = [(y - (p + bias), w) for y, p, w in zip(ys_raw, pred, ws)]
        sigmas[h] = wstd(resid, 0.0)

    def smooth(d):
        out = {}
        for h in hours:
            prev_v = d.get(h - 1, d[h])
            next_v = d.get(h + 1, d[h])
            out[h] = 0.25 * prev_v + 0.5 * d[h] + 0.25 * next_v
        return out

    centers = smooth(centers)
    sigmas = smooth(sigmas)

    out_hours = []
    for h in hours:
        sig = min(max(SIG_FLOOR, sigmas[h] * SIG_SCALE), SIG_CAP)
        qs = [p1 * math.exp(centers[h] + z * sig) for z in Z]
        qs = [round2(min(max(q, 1.0), 1e9)) for q in qs]
        for i in range(1, 5):
            qs[i] = max(qs[i], qs[i - 1])
        out_hours.append({"hour": h, "q": qs})
    return {"hours": out_hours}


def fallback_forecast(case):
    """Last-resort valid forecast: persistence baseline from hour 1."""
    known = next((k for k in case.get("known") or [] if k.get("hour") == 1),
                 None)
    p1 = max(1.0, float(known["viewers"])) if known else 1.0
    # Always emit exactly hours 2..24 (see forecast_case).
    hours = list(range(2, 25))
    return {"hours": [{"hour": h, "q": baseline_quantiles(p1, h)}
                      for h in hours]}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inputs = sorted(p for p in INPUT_DIR.glob("*.in"))
    for path in inputs:
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case)
        except Exception:
            forecast = fallback_forecast(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
