"""Tools for the Hurricane Ian flooding agent (CGN 6905 Homework 2).

Every function here is something the language model can ask the agent loop to run.
The model never sees raw time series and never does arithmetic. It sees the small
summaries these functions return, and every number in the final answer has to come
from one of them.

Tools
  1. fetch_water_levels          NOAA CO-OPS observed water levels (6-minute)
  2. fetch_tide_predictions      NOAA CO-OPS predicted astronomical tide (6-minute)
  3. fetch_flood_thresholds      NOAA minor/moderate/major flood thresholds (metadata API)
  4. analyze_flooding            pandas analysis of the cached data (peak, residual, exceedances)
  5. make_plot                   matplotlib time series of observed and predicted levels
  6. fetch_usgs_high_water_marks USGS STN surveyed high-water marks for Lee County
  7. fetch_document              text excerpts from an allow-listed NOAA/NWS/USGS document

All network access goes through cached_get(), which only talks to allow-listed hosts,
retries once, writes the raw response under data/ and logs the request URL.
"""
import datetime as dt
import hashlib
import html
import json
import math
import os
import re
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"
DOCS = DATA / "documents"
OUT = ROOT / "outputs"
for _d in (RAW, DOCS, OUT):
    _d.mkdir(parents=True, exist_ok=True)

APP_NAME = "cgn6905_hw2_agent"
NOAA_DATA_API = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
NOAA_MD_API = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/{station}/{resource}.json"
STN_API = "https://api.waterdata.usgs.gov/stnservices"

# Only these domains (and their subdomains) may be contacted.
ALLOWED_DOMAINS = ("tidesandcurrents.noaa.gov", "nhc.noaa.gov", "weather.gov", "usgs.gov")

REFRESH = os.environ.get("HW2_REFRESH") == "1"  # set by agent.py --refresh
_refreshed = set()  # URLs already re-downloaded in this process

DATUMS = ("MHHW", "MHW", "MTL", "MSL", "MLW", "MLLW", "NAVD", "STND")
TIME_ZONES = ("gmt", "lst", "lst_ldt")
UNITS = ("english", "metric")
STEP_MIN = 6
STEP_HOURS = STEP_MIN / 60.0

TZ_LABEL = {
    "gmt": "GMT (UTC)",
    "lst": "LST (local standard time, UTC-5 all year)",
    "lst_ldt": "LST/LDT (local clock time; EDT, UTC-4, on these dates)",
}


class ToolError(Exception):
    """A problem the model should be told about (bad input, missing cache, network failure)."""


# ----------------------------------------------------------------------------
# Networking with an allow-list, one retry, a raw-response cache and a request log
# ----------------------------------------------------------------------------
def _host_allowed(url):
    host = (requests.utils.urlparse(url).hostname or "").lower()
    return any(host == d or host.endswith("." + d) for d in ALLOWED_DOMAINS)


def _log_request(url, cache, status, body, path):
    rec = {
        "utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "url": url,
        "cache": cache,
        "http_status": status,
        "bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
        "file": str(Path(path).relative_to(ROOT)),
    }
    with open(DATA / "request_log.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


def cache_path(full_url, label, ext):
    key = hashlib.sha1(full_url.encode()).hexdigest()[:10]
    folder = DOCS if ext in ("pdf", "html", "txt") else RAW
    return folder / f"{label}_{key}.{ext}"


def cached_get(url, params=None, label="resp", ext="json", cache_only=False, timeout=90):
    """GET a URL and return (full_url, body_bytes, from_cache).

    The response is stored in data/raw (or data/documents) the first time. With
    cache_only=True nothing is downloaded, which is how the analysis tools read data.
    """
    full = requests.Request("GET", url, params=params).prepare().url
    if not _host_allowed(full):
        raise ToolError(f"Host not allowed: {requests.utils.urlparse(full).hostname}")
    path = cache_path(full, label, ext)
    use_cache = path.exists() and not (REFRESH and full not in _refreshed and not cache_only)
    if use_cache:
        body = path.read_bytes()
        _log_request(full, "hit", 200, body, path)
        return full, body, True
    if cache_only:
        raise ToolError(
            "Data for this request has not been retrieved yet. Call the matching fetch tool first "
            f"(expected cache file {path.name})."
        )
    last_err = None
    for attempt in (1, 2):  # one retry
        try:
            cur = full
            for _hop in range(4):  # follow redirects only inside the allow-list
                r = requests.get(
                    cur, timeout=timeout, allow_redirects=False,
                    headers={"User-Agent": "cgn6905-hw2-student-agent/1.0"},
                )
                if r.status_code in (301, 302, 303, 307, 308):
                    cur = requests.compat.urljoin(cur, r.headers.get("Location", ""))
                    if not _host_allowed(cur):
                        raise ToolError(f"Redirect to a host that is not allowed: {cur}")
                    continue
                break
            if r.status_code != 200:
                raise ToolError(f"HTTP {r.status_code} from {cur}")
            body = r.content
            if ext == "json":
                try:
                    parsed = json.loads(body)
                except ValueError:
                    raise ToolError(f"Response from {cur} was not JSON")
                if isinstance(parsed, dict) and "error" in parsed:
                    msg = parsed["error"].get("message", parsed["error"]) if isinstance(parsed["error"], dict) else parsed["error"]
                    raise ToolError(f"The service returned an error: {msg}")
            path.write_bytes(body)
            _refreshed.add(full)
            _log_request(full, "miss", r.status_code, body, path)
            return full, body, False
        except ToolError as e:
            if "service returned an error" in str(e):
                raise  # a real answer from the service, retrying will not change it
            last_err = e
        except requests.RequestException as e:
            last_err = ToolError(f"Network problem: {e.__class__.__name__}")
        time.sleep(1.5)
    raise last_err


# ----------------------------------------------------------------------------
# Input checks and small helpers
# ----------------------------------------------------------------------------
def _check_station(station):
    station = str(station).strip()
    if not re.fullmatch(r"\d{7}", station):
        raise ToolError("station must be a 7-digit NOAA station id, for example 8725520")
    return station


def _check_common(datum, time_zone, units):
    datum = str(datum).upper()
    if datum not in DATUMS:
        raise ToolError(f"datum must be one of {DATUMS}")
    time_zone = str(time_zone).lower()
    if time_zone not in TIME_ZONES:
        raise ToolError(f"time_zone must be one of {TIME_ZONES}")
    units = str(units).lower()
    if units not in UNITS:
        raise ToolError(f"units must be one of {UNITS}")
    return datum, time_zone, units


def _parse_window(begin, end):
    """begin/end are YYYYMMDD or 'YYYYMMDD HH:MM'. Date-only end means the end of that day."""
    def one(s, is_end):
        s = str(s).strip()
        for fmt in ("%Y%m%d %H:%M", "%Y%m%d"):
            try:
                t = dt.datetime.strptime(s, fmt)
                if fmt == "%Y%m%d" and is_end:
                    t = t.replace(hour=23, minute=54)
                return pd.Timestamp(t)
            except ValueError:
                pass
        raise ToolError("begin and end must look like 20220927 or '20220927 06:00'")
    start, stop = one(begin, False), one(end, True)
    if stop <= start:
        raise ToolError("end must be after begin")
    if (stop - start) > pd.Timedelta(days=31):
        raise ToolError("NOAA limits 6-minute requests to 31 days per call")
    return start, stop


def _grid(begin, end):
    start, stop = _parse_window(begin, end)
    return pd.date_range(start, stop, freq=f"{STEP_MIN}min")


def _units_label(units):
    return "ft" if units == "english" else "m"


def _fmt(t):
    return None if t is None or pd.isna(t) else pd.Timestamp(t).strftime("%Y-%m-%d %H:%M")


def _r(x, n=3):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), n)


def _runs(mask):
    """Return (start_pos, end_pos) pairs for consecutive True values in a boolean array."""
    pos = np.flatnonzero(np.asarray(mask))
    if len(pos) == 0:
        return []
    breaks = np.flatnonzero(np.diff(pos) > 1)
    starts = np.r_[pos[0], pos[breaks + 1]]
    ends = np.r_[pos[breaks], pos[-1]]
    return list(zip(starts, ends))


def _to_utc(ts, time_zone):
    """UTC equivalent of a station-local timestamp (the station is on US Eastern time)."""
    if ts is None:
        return None
    ts = pd.Timestamp(ts)
    if time_zone == "gmt":
        return _fmt(ts)
    if time_zone == "lst":
        return _fmt(ts + pd.Timedelta(hours=5))
    local = ts.tz_localize(ZoneInfo("America/New_York"), ambiguous=False)
    return _fmt(local.tz_convert("UTC").tz_localize(None))


# ----------------------------------------------------------------------------
# Tool 1 and 2: NOAA CO-OPS data retrieval
# ----------------------------------------------------------------------------
def _water_params(product, station, begin, end, datum, time_zone, units):
    p = {
        "product": product,
        "application": APP_NAME,
        "begin_date": str(begin).replace("-", ""),
        "end_date": str(end).replace("-", ""),
        "datum": datum,
        "station": station,
        "time_zone": time_zone,
        "units": units,
        "format": "json",
    }
    if product == "predictions":
        p["interval"] = str(STEP_MIN)
    return p


def _series_from(raw_bytes, product):
    j = json.loads(raw_bytes)
    rows = j["data"] if product == "water_level" else j["predictions"]
    df = pd.DataFrame(rows)
    if df.empty:
        raise ToolError("The service returned no rows for this request")
    df["t"] = pd.to_datetime(df["t"])
    df["v"] = pd.to_numeric(df["v"], errors="coerce")
    return j, df


def _fetch_series(product, station, begin, end, datum, time_zone, units):
    station = _check_station(station)
    datum, time_zone, units = _check_common(datum, time_zone, units)
    grid = _grid(begin, end)  # also validates the dates
    params = _water_params(product, station, begin, end, datum, time_zone, units)
    url, body, hit = cached_get(NOAA_DATA_API, params, label=f"{product}_{station}")
    j, df = _series_from(body, product)
    meta = j.get("metadata", {})
    ul = _units_label(units)
    on_grid = df["t"].isin(grid)
    valid = df.loc[on_grid & df["v"].notna()].drop_duplicates("t")
    n_missing = int(len(grid) - len(valid))
    imax, imin = df["v"].idxmax(), df["v"].idxmin()
    res = {
        "tool": "fetch_water_levels" if product == "water_level" else "fetch_tide_predictions",
        "request_url": url,
        "from_cache": hit,
        "station": {"id": meta.get("id", station), "name": meta.get("name"), "lat": meta.get("lat"), "lon": meta.get("lon")},
        "parameters": {"begin": str(begin), "end": str(end), "datum": datum, "time_zone": TZ_LABEL[time_zone], "units": units},
        "n_rows_returned": int(len(df)),
        "n_expected_6min_timestamps": int(len(grid)),
        "n_missing_timestamps": n_missing,
        "first_time": _fmt(df["t"].min()),
        "last_time": _fmt(df["t"].max()),
        f"min_{ul}_above_{datum}": {"value": _r(df.loc[imin, "v"]), "time": _fmt(df.loc[imin, "t"])},
        f"max_{ul}_above_{datum}": {"value": _r(df.loc[imax, "v"]), "time": _fmt(df.loc[imax, "t"])},
        "first_3_rows": [{"t": _fmt(r.t), "v": _r(r.v)} for r in df.head(3).itertuples()],
        "last_3_rows": [{"t": _fmt(r.t), "v": _r(r.v)} for r in df.tail(3).itertuples()],
    }
    if product == "water_level":
        flagged = int(df["f"].astype(str).str.contains("1").sum()) if "f" in df else None
        res["n_verified"] = int((df["q"] == "v").sum()) if "q" in df else None
        res["n_preliminary"] = int((df["q"] == "p").sum()) if "q" in df else None
        res["n_rows_with_any_quality_flag"] = flagged
        res["n_null_values"] = int(df["v"].isna().sum())
    res["summary"] = (
        f"{res['n_rows_returned']} rows from {res['first_time']} to {res['last_time']} "
        f"({TZ_LABEL[time_zone]}), {n_missing} of {len(grid)} expected 6-minute timestamps missing, "
        f"datum {datum}, units {ul}. Cache {'hit' if hit else 'miss (downloaded now)'}."
    )
    return res


def fetch_water_levels(station, begin, end, datum="MHHW", time_zone="lst_ldt", units="english"):
    return _fetch_series("water_level", station, begin, end, datum, time_zone, units)


def fetch_tide_predictions(station, begin, end, datum="MHHW", time_zone="lst_ldt", units="english"):
    return _fetch_series("predictions", station, begin, end, datum, time_zone, units)


# ----------------------------------------------------------------------------
# Tool 3: flood thresholds from the CO-OPS metadata API
# ----------------------------------------------------------------------------
FLOOD_KEYS = ("minor", "moderate", "major")


def _md_get(station, resource, units, cache_only=False):
    url = NOAA_MD_API.format(station=station, resource=resource)
    full, body, hit = cached_get(url, {"units": units}, label=f"md_{resource}_{station}", cache_only=cache_only)
    return full, json.loads(body), hit


def _datum_table(station, units, cache_only=False):
    full, j, hit = _md_get(station, "datums", units, cache_only)
    table = {d["name"]: float(d["value"]) for d in j["datums"]}
    return full, j, table


def _thresholds_in_datum(station, datum, units, cache_only=True):
    """NWS and NOS thresholds relative to the requested datum.

    The metadata API gives thresholds as heights above station datum (STND), so the
    height of the requested datum above STND is subtracted from each one.
    """
    _, fl, _ = _md_get(station, "floodlevels", units, cache_only)
    _, _, tbl = _datum_table(station, units, cache_only)
    dname = "NAVD88" if datum == "NAVD" else datum
    off = 0.0 if dname == "STND" else tbl[dname]
    out = {"nws": {}, "nos": {}}
    for k in FLOOD_KEYS:
        for src in ("nws", "nos"):
            v = fl.get(f"{src}_{k}")
            out[src][k] = None if v is None else round(float(v) - off, 3)
    return out, fl, tbl


def fetch_flood_thresholds(station, units="english"):
    station = _check_station(station)
    units_ = _check_common("MHHW", "lst_ldt", units)[2]
    fl_url, fl, hit = _md_get(station, "floodlevels", units_)
    dm_url, dm, tbl = _datum_table(station, units_)
    ul = _units_label(units_)
    conv = {}
    for datum in ("MHHW", "MLLW", "NAVD"):
        conv[datum], _, _ = _thresholds_in_datum(station, datum, units_, cache_only=False)
    raw = {k: fl.get(k) for k in ("nos_minor", "nos_moderate", "nos_major", "nws_minor", "nws_moderate", "nws_major", "action")}
    res = {
        "tool": "fetch_flood_thresholds",
        "request_urls": [fl_url, dm_url],
        "from_cache": bool(hit),
        "station": station,
        "units": units_,
        f"published_values_{ul}_above_station_datum": raw,
        f"datum_heights_{ul}_above_station_datum": {k: tbl.get(k) for k in ("MHHW", "MLLW", "NAVD88")},
        "datum_epoch": dm.get("epoch"),
        "datum_accepted": dm.get("accepted"),
        f"thresholds_{ul}_above_MHHW": conv["MHHW"],
        f"thresholds_{ul}_above_MLLW": conv["MLLW"],
        f"thresholds_{ul}_above_NAVD88": conv["NAVD"],
        "definitions": {
            "nws": "NWS minor/moderate/major coastal flood thresholds: water levels set by the local NWS forecast office "
                   "with local emergency managers, based on years of impact monitoring (road closures, flooding of low areas). "
                   "Minor: low threat of property damage and no direct threat to life. Moderate: elevated threat of property "
                   "damage and some risk to life. Major: significant threat to life and property.",
            "nos": "NOS minor/moderate/major thresholds: derived by NOAA NOS CO-OPS with one consistent method for all stations "
                   "(NOAA Technical Report NOS CO-OPS 086, Patterns and Projections of High Tide Flooding), so that stations can be "
                   "compared. They are not tied to local road or building impacts.",
            "source": "NOAA CO-OPS Metadata API floodlevels and datums resources; descriptions from the NOAA Coastal Inundation "
                      "Dashboard FAQ at https://tidesandcurrents.noaa.gov/inundationdb/",
        },
        "reference_datum_note": (
            "The metadata API does not state the datum of the published values. The Coastal Inundation Dashboard script "
            "(https://tidesandcurrents.noaa.gov/inundationdb/assets/js/app.js) subtracts the plotted datum's height above "
            "station datum from them, so I treat them as heights above station datum and subtract the MHHW height to get "
            "feet above MHHW. Cross-check: the NWS values minus MHHW come out as round numbers (see thresholds above MHHW)."
        ),
        "retrospective_note": (
            "These are the thresholds NOAA publishes now (retrieved "
            + dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d") + "). Applying them to the September 2022 "
            "event is a retrospective use, because the values or datums in force in 2022 may have differed."
        ),
    }
    res["summary"] = (
        f"NWS thresholds above MHHW ({ul}): minor {conv['MHHW']['nws']['minor']}, moderate {conv['MHHW']['nws']['moderate']}, "
        f"major {conv['MHHW']['nws']['major']}. NOS thresholds above MHHW: minor {conv['MHHW']['nos']['minor']}, "
        f"moderate {conv['MHHW']['nos']['moderate']}, major {conv['MHHW']['nos']['major']}. "
        f"Published as heights above station datum, MHHW = {tbl['MHHW']} {ul} above station datum."
    )
    return res


# ----------------------------------------------------------------------------
# Tool 4: analysis
# ----------------------------------------------------------------------------
STATE = {"last_analysis": None}  # set by analyze_flooding, read by the USGS comparison


def _load_cached_series(product, station, begin, end, datum, time_zone, units):
    params = _water_params(product, station, begin, end, datum, time_zone, units)
    url, body, _ = cached_get(NOAA_DATA_API, params, label=f"{product}_{station}", cache_only=True)
    _, df = _series_from(body, product)
    dup = int(df["t"].duplicated().sum())
    s = df.drop_duplicates("t").set_index("t")
    return url, s, dup


def _exceedance(series, thr, peak_value, step_hours=STEP_HOURS):
    above = (series > thr).fillna(False).to_numpy()
    n = int(above.sum())
    spells = _runs(above)
    times = series.index[above]
    return {
        "threshold": _r(thr, 3),
        "exceeded": bool(n > 0),
        "n_samples_above": n,
        "hours_above": _r(n * step_hours, 2),
        "first_time_above": _fmt(times[0]) if n else None,
        "last_time_above": _fmt(times[-1]) if n else None,
        "n_separate_spells": len(spells),
        "longest_spell_hours": _r(max((e - s + 1) for s, e in spells) * step_hours, 2) if spells else 0.0,
        "peak_minus_threshold": _r(peak_value - thr, 3),
    }


def analyze_flooding(station, begin, end, datum="MHHW", time_zone="lst_ldt", units="english"):
    station = _check_station(station)
    datum, time_zone, units = _check_common(datum, time_zone, units)
    grid = _grid(begin, end)
    ul = _units_label(units)
    obs_url, obs_df, obs_dup = _load_cached_series("water_level", station, begin, end, datum, time_zone, units)
    prd_url, prd_df, prd_dup = _load_cached_series("predictions", station, begin, end, datum, time_zone, units)
    thr, fl_raw, tbl = _thresholds_in_datum(station, datum, units, cache_only=True)

    obs = obs_df["v"].reindex(grid)  # missing timestamps become NaN, nothing is interpolated
    prd = prd_df["v"].reindex(grid)
    matched = obs.notna() & prd.notna()
    resid = (obs - prd)[matched]

    obs_missing = obs.isna().to_numpy()
    prd_missing = prd.isna().to_numpy()
    miss_runs = [{"start": _fmt(grid[s]), "end": _fmt(grid[e]), "n": int(e - s + 1)} for s, e in _runs(obs_missing)]

    ipk = obs.idxmax()
    peak = float(obs[ipk])
    ires = resid.idxmax()
    imin = resid.idxmin()
    iobsmin = obs.idxmin()
    ipred = prd.idxmax()

    # offsets to link the gauge to other datums (used to compare with survey marks)
    dname = "NAVD88" if datum == "NAVD" else datum
    stnd_of_datum = 0.0 if dname == "STND" else tbl[dname]
    peak_navd88 = peak + stnd_of_datum - tbl["NAVD88"]
    peak_mhhw = peak + stnd_of_datum - tbl["MHHW"]

    # data-quality details about the peak sample
    sigma = pd.to_numeric(obs_df["s"], errors="coerce") if "s" in obs_df else None
    qc = {
        "n_verified": int((obs_df["q"] == "v").sum()) if "q" in obs_df else None,
        "n_preliminary": int((obs_df["q"] == "p").sum()) if "q" in obs_df else None,
        "n_rows_with_any_quality_flag": int(obs_df["f"].astype(str).str.contains("1").sum()) if "f" in obs_df else None,
        "n_duplicate_timestamps_dropped": obs_dup,
    }
    if sigma is not None and ipk in sigma.index:
        qc["sigma_at_peak_sample"] = _r(sigma[ipk])
        qc["median_sigma_all_samples"] = _r(sigma.median())
        qc["note_sigma"] = "sigma is the spread of the 1-second readings inside each 6-minute sample (wave and turbulence noise)"

    daily = []
    for day in sorted({t.date() for t in grid}):
        m = np.array([t.date() == day for t in grid])
        o, p = obs[m], prd[m]
        r = (obs - prd)[m & matched.to_numpy()]
        row = {
            "date": str(day),
            "obs_max": _r(o.max()), "obs_max_time": _fmt(o.idxmax()) if o.notna().any() else None,
            "pred_max": _r(p.max()), "pred_max_time": _fmt(p.idxmax()) if p.notna().any() else None,
            "max_residual": _r(r.max()) if len(r) else None, "max_residual_time": _fmt(r.idxmax()) if len(r) else None,
            "obs_missing": int(o.isna().sum()),
        }
        daily.append(row)

    sets = {}
    for src in ("nws", "nos"):
        sets[src] = {k: _exceedance(obs, thr[src][k], peak) for k in FLOOD_KEYS if thr[src][k] is not None}
    highest = {}
    for src in sets:
        reached = [k for k in FLOOD_KEYS if k in sets[src] and sets[src][k]["exceeded"]]
        highest[src] = reached[-1] if reached else "none"

    res = {
        "tool": "analyze_flooding",
        "station": station,
        "window": f"{_fmt(grid[0])} to {_fmt(grid[-1])}",
        "datum": datum, "time_zone": TZ_LABEL[time_zone], "units": units, "value_label": f"{ul} above {datum}",
        "source_urls": {"observed": obs_url, "predicted": prd_url},
        "completeness": {
            "n_expected_6min_timestamps": int(len(grid)),
            "n_observed_present": int(obs.notna().sum()),
            "n_observed_missing": int(obs_missing.sum()),
            "missing_observed_runs": miss_runs[:10],
            "n_predicted_present": int(prd.notna().sum()),
            "n_predicted_missing": int(prd_missing.sum()),
            "n_matched_timestamps": int(matched.sum()),
            "rows_off_the_6min_grid_ignored": int((~obs_df.index.isin(grid)).sum()),
            "interpolation": "none; residuals use only timestamps present in both series",
        },
        "data_quality": qc,
        "peak_observed": {"value": _r(peak), "time": _fmt(ipk), "time_utc": _to_utc(ipk, time_zone),
                          "predicted_tide_at_that_time": _r(prd.get(ipk)),
                          "residual_at_that_time": _r(resid.get(ipk))},
        "max_residual": {"value": _r(resid[ires]), "time": _fmt(ires), "time_utc": _to_utc(ires, time_zone),
                         "observed_at_that_time": _r(obs[ires]), "predicted_at_that_time": _r(prd[ires])},
        "most_negative_residual": {"value": _r(resid[imin]), "time": _fmt(imin),
                                   "observed_at_that_time": _r(obs[imin]), "predicted_at_that_time": _r(prd[imin])},
        "lowest_observed": {"value": _r(obs[iobsmin]), "time": _fmt(iobsmin)},
        "highest_predicted": {"value": _r(prd[ipred]), "time": _fmt(ipred)},
        "residual_stats": {"mean": _r(resid.mean()), "std": _r(resid.std()), "n": int(len(resid))},
        "observed_vs_predicted_by_day": daily,
        "peak_in_other_datums": {
            f"{ul}_above_MHHW": _r(peak_mhhw, 2),
            f"{ul}_above_NAVD88": _r(peak_navd88, 2),
            "note": "converted with the datum heights from the CO-OPS datums resource (each rounded to 0.01 ft)",
        },
        "thresholds_used": {"datum": datum, "units": units, "values": thr,
                            "note": "current NOAA thresholds applied retrospectively to a 2022 event"},
        "exceedance": sets,
        "highest_category_reached": highest,
        "duration_method": f"each 6-minute observation above a threshold counts as {STEP_MIN} minutes; no interpolation; "
                           "if samples are missing the hours are a lower bound",
    }
    res["summary"] = (
        f"Peak observed {res['peak_observed']['value']} {res['value_label']} at {res['peak_observed']['time']} "
        f"{TZ_LABEL[time_zone]}; max residual {res['max_residual']['value']} {ul} at {res['max_residual']['time']}; "
        f"{res['completeness']['n_observed_missing']} of {len(grid)} observed timestamps missing; "
        f"highest NWS category reached: {highest['nws']}, NOS: {highest['nos']}."
    )
    # keep the result for verify.py and for the USGS comparison
    out_name = f"analysis_{station}_{str(begin)}_{str(end)}_{datum}.json"
    (OUT / out_name).write_text(json.dumps(res, indent=2, default=str))
    STATE["last_analysis"] = json.loads(json.dumps(res, default=str))
    res["saved_to"] = f"outputs/{out_name}"
    return res


# ----------------------------------------------------------------------------
# Tool 5: plot
# ----------------------------------------------------------------------------
def make_plot(station, begin, end, datum="MHHW", time_zone="lst_ldt", units="english",
              threshold_set="nws", filename=None):
    station = _check_station(station)
    datum, time_zone, units = _check_common(datum, time_zone, units)
    if threshold_set not in ("nws", "nos", "both", "none"):
        raise ToolError("threshold_set must be nws, nos, both or none")
    grid = _grid(begin, end)
    ul = _units_label(units)
    _, obs_df, _ = _load_cached_series("water_level", station, begin, end, datum, time_zone, units)
    _, prd_df, _ = _load_cached_series("predictions", station, begin, end, datum, time_zone, units)
    obs = obs_df["v"].reindex(grid)   # gaps stay as NaN so the line breaks instead of being joined
    prd = prd_df["v"].reindex(grid)
    resid = obs - prd
    thr = None
    if threshold_set != "none":
        thr = _thresholds_in_datum(station, datum, units, cache_only=True)[0]
    name = {"8725520": "Fort Myers, FL"}.get(station, f"Station {station}")

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 7.4), sharex=True, gridspec_kw={"height_ratios": [3.2, 1], "hspace": 0.08})
    ax.plot(grid, obs, color="#1f4e79", lw=1.9, label="Observed water level (6-min)")
    ax.plot(grid, prd, color="#d95f02", lw=1.5, ls="--", label="Predicted astronomical tide")
    ipk = obs.idxmax()
    ax.plot([ipk], [obs[ipk]], "o", color="#1f4e79", ms=7, mec="white", zorder=5)
    ax.annotate(f"Peak {obs[ipk]:.2f} {ul}\n{pd.Timestamp(ipk):%b %d %H:%M}", xy=(ipk, obs[ipk]),
                xytext=(28, 2), textcoords="offset points", ha="left", va="center", fontsize=10,
                arrowprops=dict(arrowstyle="-", color="0.4"))
    xmax = grid[-1]
    if thr is not None:
        styles = {"nws": {"minor": ("#c9a100", ":"), "moderate": ("#e8710a", ":"), "major": ("#b3261e", ":")},
                  "nos": {"minor": ("0.55", "-."), "moderate": ("0.45", "-."), "major": ("0.35", "-.")}}
        for src in ("nws", "nos"):
            if threshold_set in (src, "both"):
                for k in FLOOD_KEYS:
                    v = thr[src][k]
                    if v is None:
                        continue
                    col, ls = styles[src][k]
                    ax.axhline(v, color=col, ls=ls, lw=1.4, zorder=1)
                    ax.text(xmax, v, f" {src.upper()} {k} {v:.2f}", color=col if src == "nws" else "0.35",
                            va="center", ha="left", fontsize=8.5, clip_on=False)
    ax.set_ylabel(f"Water level ({ul} above {datum})", fontsize=11)
    ax.grid(alpha=0.3)
    tz_short = {"gmt": "GMT", "lst": "LST", "lst_ldt": "local time, LST/LDT (EDT)"}[time_zone]
    ax.set_title(f"{name} (NOAA CO-OPS {station}): observed and predicted water level, Hurricane Ian, "
                 f"{grid[0]:%b %d}-{grid[-1]:%d, %Y}", fontsize=12)
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.95)

    ax2.plot(grid, resid, color="#444444", lw=1.3)
    ax2.axhline(0, color="0.5", lw=0.8)
    ires = resid.idxmax()
    ax2.plot([ires], [resid[ires]], "o", color="#444444", ms=6, mec="white")
    ax2.annotate(f"Max residual {resid[ires]:.2f} {ul}", xy=(ires, resid[ires]), xytext=(-12, -4),
                 textcoords="offset points", ha="right", va="top", fontsize=9)
    ax2.set_ylabel(f"Observed minus\npredicted ({ul})", fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.set_xlabel(f"Time ({tz_short})", fontsize=11)
    ax2.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    ax2.set_xlim(grid[0], xmax)

    # shade missing observations so gaps are visible
    miss = obs.isna().to_numpy()
    for s, e in _runs(miss):
        for a in (ax, ax2):
            a.axvspan(grid[s], grid[e], color="red", alpha=0.15, lw=0)
    n_miss = int(miss.sum())
    note = (f"Data: NOAA CO-OPS API, station {station}, 6-minute observations ({int(obs.notna().sum())} of {len(grid)} present, "
            f"{n_miss} missing) and predictions, datum {datum}.")
    if thr is not None:
        note += "\nFlood thresholds: current NOAA values from the CO-OPS metadata API, applied retrospectively to this 2022 event."
    fig.text(0.01, 0.006, note, fontsize=8, color="0.3", ha="left", va="bottom")
    fig.subplots_adjust(left=0.08, right=0.86 if thr is not None else 0.97, top=0.94, bottom=0.145)
    fname = filename or ("fort_myers_ian_water_levels.png" if station == "8725520" else f"{station}_water_levels.png")
    fname = Path(fname).name  # always inside outputs/
    path = OUT / fname
    fig.savefig(path, dpi=150)
    plt.close(fig)
    res = {
        "tool": "make_plot",
        "saved_to": f"outputs/{fname}",
        "size_bytes": path.stat().st_size,
        "panels": ["observed and predicted water level with flood-threshold lines", "residual (observed minus predicted)"],
        "y_axis": f"{ul} above {datum}",
        "x_axis": TZ_LABEL[time_zone],
        "threshold_lines_drawn": None if thr is None else {s: thr[s] for s in ("nws", "nos") if threshold_set in (s, "both")},
        "missing_observed_timestamps_shaded": n_miss,
    }
    res["summary"] = f"Saved {res['saved_to']} ({res['size_bytes']} bytes) with y axis {res['y_axis']} and x axis {res['x_axis']}."
    return res


# ----------------------------------------------------------------------------
# Tool 6: USGS Short-Term Network high-water marks
# ----------------------------------------------------------------------------
GAUGE_LATLON = {"8725520": (26.6478, -81.8711)}
AREAS = {
    "lee_county": "all Lee County marks",
    "gauge_vicinity_5km": "marks within 5 km of the Fort Myers tide gauge (26.6478, -81.8711)",
    "fort_myers_beach": "approximate box lat 26.38-26.47, lon -81.98 to -81.88 (Estero Island / Fort Myers Beach)",
    "sanibel_captiva": "approximate boxes: lat 26.38-26.475, lon -82.20 to -82.00 (Sanibel) plus lat 26.475-26.56, "
                       "lon -82.26 to -82.15 (western Sanibel and Captiva)",
}
QUALITY_RANK = {"excellent": 0, "good": 1, "fair": 2, "poor": 3, "vp": 4}


def _haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def _area_mask(df, area):
    lat, lon = df["latitude_dd"], df["longitude_dd"]
    if area == "lee_county":
        return pd.Series(True, index=df.index)
    if area == "gauge_vicinity_5km":
        g = GAUGE_LATLON["8725520"]
        return _haversine_km(g[0], g[1], lat, lon) <= 5.0
    if area == "fort_myers_beach":
        return lat.between(26.38, 26.47) & lon.between(-81.98, -81.88)
    if area == "sanibel_captiva":
        return (lat.between(26.38, 26.475) & lon.between(-82.20, -82.00)) | (lat.between(26.475, 26.56) & lon.between(-82.26, -82.15))
    raise ToolError(f"area must be one of {list(AREAS)}")


def _stats(s):
    s = pd.Series(s).dropna()
    if s.empty:
        return {"n": 0}
    return {"n": int(len(s)), "min": _r(s.min(), 2), "p25": _r(np.percentile(s, 25), 2), "median": _r(s.median(), 2),
            "p75": _r(np.percentile(s, 75), 2), "max": _r(s.max(), 2)}


def fetch_usgs_high_water_marks(area="lee_county", min_quality="fair", event_name="2022 Ian", state="FL", county="Lee County"):
    if area not in AREAS:
        raise ToolError(f"area must be one of {list(AREAS)}")
    mq = str(min_quality).lower()
    if mq != "any" and mq not in QUALITY_RANK:
        raise ToolError("min_quality must be excellent, good, fair, poor or any")
    ev_url, ev_body, ev_hit = cached_get(f"{STN_API}/events/", None, label="stn_events")
    events = json.loads(ev_body)
    match = [e for e in events if (e.get("name") or "").strip().lower() == str(event_name).strip().lower()]
    if not match:
        raise ToolError(f"No STN event named {event_name!r}. Example: '2022 Ian'")
    event_id = match[0]["id"]
    hwm_url, hwm_body, hwm_hit = cached_get(f"{STN_API}/hwms/filteredhwms/", {"event": event_id, "states": state}, label="stn_hwms")
    df = pd.DataFrame(json.loads(hwm_body))
    n_state = int(len(df))
    df = df[df["county_name"].str.lower() == str(county).lower()].copy()
    for c in ("elev_ft", "height_above_ground", "latitude_dd", "longitude_dd"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    n_county = int(len(df))
    df = df[_area_mask(df, area)]
    n_area = int(len(df))
    rank = df["hwm_quality_name"].str.split(":").str[0].str.lower().map(QUALITY_RANK)
    if mq != "any":
        df = df[rank <= QUALITY_RANK[mq]]
    df = df.dropna(subset=["elev_ft"])
    top = df.sort_values("elev_ft", ascending=False).head(5)
    res = {
        "tool": "fetch_usgs_high_water_marks",
        "request_urls": [ev_url, hwm_url],
        "from_cache": bool(ev_hit and hwm_hit),
        "source": "USGS Short-Term Network (STN) flood event data, high-water marks",
        "event": {"id": event_id, "name": match[0]["name"]},
        "area": {"name": area, "definition": AREAS[area]},
        "min_quality_kept": mq,
        "counts": {f"marks_in_{state}_for_event": n_state, f"marks_in_{county}": n_county,
                   "marks_in_area": n_area, "marks_used_after_quality_filter": int(len(df))},
        "elevation_ft_NAVD88": _stats(df["elev_ft"]),
        "height_above_ground_ft": _stats(df["height_above_ground"]),
        "mark_types": df["hwm_type_name"].value_counts().to_dict(),
        "environment": df["environment"].value_counts().to_dict(),
        "highest_5_marks": [
            {"site_number": r.site_number, "lat": _r(r.latitude_dd, 3), "lon": _r(r.longitude_dd, 3),
             "elev_ft_NAVD88": _r(r.elev_ft, 2), "height_above_ground_ft": _r(r.height_above_ground, 2),
             "quality": str(r.hwm_quality_name).split(":")[0], "type": r.hwm_type_name}
            for r in top.itertuples()
        ],
        "field_meanings": {
            "elev_ft_NAVD88": "surveyed elevation of the mark, the peak local water-surface elevation, feet above NAVD88",
            "height_above_ground_ft": "height of the mark above the ground at that structure or site, "
                                      "so the local depth of water above ground at the mark",
            "caution": "a mark is one surveyed peak at one place. Seed lines and mud lines can include wave effects. "
                       "Marks are not a time series and are not comparable to gauge data without a datum conversion.",
        },
    }
    # compare with the Fort Myers gauge peak, converted to NAVD88
    gauge = STATE["last_analysis"]
    if gauge and gauge.get("station") == "8725520" and len(df):
        gp = gauge["peak_in_other_datums"].get("ft_above_NAVD88")
        if gp is not None:
            e = df["elev_ft"]
            res["comparison_with_fort_myers_gauge"] = {
                "gauge_peak_ft_above_NAVD88": gp,
                "gauge_peak_conversion": gauge["peak_in_other_datums"]["note"],
                "n_marks_higher_than_gauge_peak": int((e > gp).sum()),
                "share_of_marks_higher_than_gauge_peak": _r((e > gp).mean(), 2),
                "median_mark_minus_gauge_peak_ft": _r(e.median() - gp, 2),
                "max_mark_minus_gauge_peak_ft": _r(e.max() - gp, 2),
            }
    else:
        res["comparison_with_fort_myers_gauge"] = "not available: analyze_flooding has not been run in this session for the Fort Myers gauge"
    out_csv = OUT / f"usgs_hwm_{area}.csv"
    df[["id", "site_number", "latitude_dd", "longitude_dd", "elev_ft", "height_above_ground", "vertical_datum_name",
        "hwm_quality_name", "hwm_type_name", "environment", "flag_date"]].to_csv(out_csv, index=False)
    res["saved_marks_table"] = f"outputs/{out_csv.name}"
    s = res["elevation_ft_NAVD88"]
    res["summary"] = (
        f"{res['counts']['marks_used_after_quality_filter']} marks used in area '{area}': elevation median {s.get('median')} ft NAVD88 "
        f"(range {s.get('min')} to {s.get('max')}); height above ground median "
        f"{res['height_above_ground_ft'].get('median')} ft (n={res['height_above_ground_ft']['n']})."
    )
    return res


# ----------------------------------------------------------------------------
# Tool 7: allow-listed document excerpts
# ----------------------------------------------------------------------------
def _pdf_text(path):
    import fitz  # PyMuPDF
    doc = fitz.open(path)
    parts = [f"\f=== PAGE {i + 1} ===\n" + p.get_text() for i, p in enumerate(doc)]
    return "".join(parts), doc.page_count


def fetch_document(url, keywords=None, max_snippets=8):
    if not _host_allowed(url):
        raise ToolError(f"Only NOAA/NWS/USGS hosts are allowed: {', '.join(ALLOWED_DOMAINS)}")
    path_part = requests.utils.urlparse(url).path
    ext = "pdf" if path_part.lower().endswith(".pdf") else "html"
    base = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(path_part).name or "page")
    full, body, hit = cached_get(url, None, label=base.rsplit(".", 1)[0], ext=ext, timeout=180)
    cache_file = cache_path(full, base.rsplit(".", 1)[0], ext)
    txt_file = cache_file.with_suffix(".txt")
    if ext == "pdf":
        if txt_file.exists():
            text = txt_file.read_text()
            m = re.findall(r"=== PAGE (\d+) ===", text)
            pages = len(m)
        else:
            text, pages = _pdf_text(cache_file)
            txt_file.write_text(text)
    else:
        raw = body.decode("utf-8", errors="ignore")
        raw = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw)
        text = html.unescape(re.sub(r"<[^>]+>", " ", raw))
        pages = None
        txt_file.write_text(text)
    keywords = [k for k in (keywords or []) if str(k).strip()]
    snippets = []
    page_starts = [(m.start(), int(m.group(1))) for m in re.finditer(r"=== PAGE (\d+) ===", text)]

    def page_of(pos):
        pg = None
        for st, n in page_starts:
            if st <= pos:
                pg = n
        return pg

    flat = text
    for kw in keywords:
        count = 0
        for m in re.finditer(re.escape(kw), flat, flags=re.I):
            a, b = max(0, m.start() - 260), min(len(flat), m.end() + 520)
            snippet = re.sub(r"\s+", " ", flat[a:b]).strip()
            if any(snippet[:80] in s["text"] for s in snippets):
                continue  # skip near-duplicates
            snippets.append({"keyword": kw, "page": page_of(m.start()), "text": snippet})
            count += 1
            if count >= 3 or len(snippets) >= int(max_snippets):
                break
        if len(snippets) >= int(max_snippets):
            break
    if not keywords:
        snippets = [{"keyword": None, "page": 1 if pages else None, "text": re.sub(r"\s+", " ", flat[:1500]).strip()}]
    res = {
        "tool": "fetch_document",
        "url": full,
        "from_cache": hit,
        "retrieved_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d"),
        "pages": pages,
        "size_bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest()[:16],
        "keywords": keywords,
        "n_snippets": len(snippets),
        "snippets": snippets,
    }
    res["summary"] = f"{full}: {len(snippets)} excerpt(s) for {keywords or 'start of document'}."
    return res


# ----------------------------------------------------------------------------
# Tool registry for the agent loop
# ----------------------------------------------------------------------------
_COMMON = {
    "station": {"type": "string", "description": "7-digit NOAA CO-OPS station id. Fort Myers, FL is 8725520."},
    "begin": {"type": "string", "description": "Start date YYYYMMDD (or 'YYYYMMDD HH:MM'), in the requested time zone."},
    "end": {"type": "string", "description": "End date YYYYMMDD (a date-only end means the end of that day), in the requested time zone."},
    "datum": {"type": "string", "enum": list(DATUMS), "description": "Vertical datum. MHHW means feet above Mean Higher High Water."},
    "time_zone": {"type": "string", "enum": list(TIME_ZONES), "description": "gmt, lst (local standard) or lst_ldt (local clock time)."},
    "units": {"type": "string", "enum": list(UNITS), "description": "english (feet) or metric (meters)."},
}


def _schema(name, desc, props, required):
    return {"name": name, "description": desc,
            "input_schema": {"type": "object", "properties": props, "required": required}}


TOOL_SCHEMAS = [
    _schema("fetch_water_levels",
            "Retrieve OBSERVED 6-minute water levels from the NOAA CO-OPS Data API (product=water_level) and save the raw response. "
            "Returns row counts, missing-timestamp count, quality counts, min/max and a few sample rows, not the full series. "
            "Must be called before analyze_flooding or make_plot for the same station, dates, datum, time zone and units.",
            _COMMON, ["station", "begin", "end", "datum", "time_zone", "units"]),
    _schema("fetch_tide_predictions",
            "Retrieve PREDICTED astronomical tide (6-minute) from the NOAA CO-OPS Data API (product=predictions) and save the raw response. "
            "Use the same station, dates, datum, time zone and units as the observed data. Must be called before analyze_flooding or make_plot.",
            _COMMON, ["station", "begin", "end", "datum", "time_zone", "units"]),
    _schema("fetch_flood_thresholds",
            "Retrieve NOAA's published minor/moderate/major coastal flood thresholds for a station from the CO-OPS metadata API, "
            "with the station datum heights, thresholds converted to feet above MHHW, MLLW and NAVD88, their source, definition and a "
            "retrospective-use note. Must be called before analyze_flooding or make_plot.",
            {"station": _COMMON["station"], "units": _COMMON["units"]}, ["station"]),
    _schema("analyze_flooding",
            "Analyze the data already retrieved by the fetch tools (reads the saved responses, does not download). Computes: peak observed water "
            "level and its time, observed vs predicted comparison by day, maximum storm residual (observed minus predicted at matching timestamps) "
            "and its time, counts and runs of missing values, data-quality counts, number of observations and hours above each NWS and NOS "
            "flood threshold, the highest category reached, and the peak expressed in NAVD88. Use this for any number about the gauge data.",
            _COMMON, ["station", "begin", "end", "datum", "time_zone", "units"]),
    _schema("make_plot",
            "Draw and save the time-series plot (observed vs predicted water level, flood-threshold lines, and a residual panel) from the saved data. "
            "Saves a PNG under outputs/. Needs the same three fetch tools called first.",
            {**_COMMON,
             "threshold_set": {"type": "string", "enum": ["nws", "nos", "both", "none"], "description": "Which threshold lines to draw. Default nws."}},
            ["station", "begin", "end", "datum", "time_zone", "units"]),
    _schema("fetch_usgs_high_water_marks",
            "Retrieve surveyed USGS Short-Term Network high-water marks for Hurricane Ian in Lee County, Florida (regional context). Returns counts, "
            "elevation statistics (feet above NAVD88), height-above-ground statistics, the five highest marks, and a comparison with the Fort Myers "
            "gauge peak if analyze_flooding was run. Choose an area: lee_county, gauge_vicinity_5km, fort_myers_beach or sanibel_captiva. "
            "Marks are single surveyed peaks, not time series, and use NAVD88, not MHHW.",
            {"area": {"type": "string", "enum": list(AREAS), "description": "Sub-area of Lee County to summarize."},
             "min_quality": {"type": "string", "enum": ["excellent", "good", "fair", "poor", "any"],
                             "description": "Worst survey quality to keep. Default fair (about +/- 0.2 ft or better)."}},
            ["area"]),
    _schema("fetch_document",
            "Download a document from an allow-listed NOAA, NWS or USGS website (nhc.noaa.gov, weather.gov, tidesandcurrents.noaa.gov, usgs.gov) "
            "and return excerpts around the given keywords with page numbers. Known document: the National Hurricane Center Tropical Cyclone "
            "Report for Hurricane Ian, https://www.nhc.noaa.gov/data/tcr/AL092022_Ian.pdf (storm surge section on pages 8-10, Table 3 on page 24). "
            "Excerpts are the only content returned, so choose specific keywords.",
            {"url": {"type": "string", "description": "Full https URL on an allowed domain."},
             "keywords": {"type": "array", "items": {"type": "string"}, "description": "Words or phrases to find, for example ['Sanibel', 'Fort Myers Beach', 'bridges']."}},
            ["url", "keywords"]),
]

TOOL_FUNCS = {
    "fetch_water_levels": fetch_water_levels,
    "fetch_tide_predictions": fetch_tide_predictions,
    "fetch_flood_thresholds": fetch_flood_thresholds,
    "analyze_flooding": analyze_flooding,
    "make_plot": make_plot,
    "fetch_usgs_high_water_marks": fetch_usgs_high_water_marks,
    "fetch_document": fetch_document,
}


def run_tool(name, args):
    """Run one tool. Returns (result_dict, is_error)."""
    func = TOOL_FUNCS.get(name)
    if func is None:
        return {"error": f"unknown tool {name}"}, True
    try:
        return func(**args), False
    except ToolError as e:
        return {"error": str(e)}, True
    except TypeError as e:
        return {"error": f"bad arguments for {name}: {e}"}, True
