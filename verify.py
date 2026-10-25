#!/usr/bin/env python
"""Independent check of the agent's numbers.

This script does not import tools.py. It reads the raw NOAA and USGS responses saved
under data/ and recomputes the key numbers with plain Python (json, statistics, math),
then compares them with what the agent's analysis tool reported in outputs/ and with the
National Hurricane Center Tropical Cyclone Report for Ian (Table 3, cached text).

Run:  python verify.py          (prints PASS or FAIL for each check, exit code 1 if any fail)
"""
import glob
import json
import math
import re
import statistics
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
results = []


def check(name, ours, theirs, tol=0.0005):
    if isinstance(ours, str) or isinstance(theirs, str) or ours is None or theirs is None:
        ok = ours == theirs
    else:
        ok = abs(ours - theirs) <= tol
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}: recomputed={ours}  reported={theirs}")


def find_raw(substr):
    """Raw cache file for the first logged request whose URL contains substr."""
    for line in (ROOT / "data" / "request_log.jsonl").read_text().splitlines():
        rec = json.loads(line)
        if substr in rec["url"]:
            return ROOT / rec["file"], rec["url"]
    raise SystemExit(f"No logged request containing {substr!r}. Run the agent first.")


def tstamp(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M")


# ---------------------------------------------------------------- load raw files
obs_file, obs_url = find_raw("product=water_level")
prd_file, prd_url = find_raw("product=predictions")
fl_file, _ = find_raw("/floodlevels.json")
dm_file, _ = find_raw("/datums.json")
obs_rows = json.loads(obs_file.read_text())["data"]
prd_rows = json.loads(prd_file.read_text())["predictions"]
floods = json.loads(fl_file.read_text())
datums = {d["name"]: d["value"] for d in json.loads(dm_file.read_text())["datums"]}

obs = {r["t"]: float(r["v"]) for r in obs_rows if r["v"] not in ("", None)}
prd = {r["t"]: float(r["v"]) for r in prd_rows if r["v"] not in ("", None)}

# ---------------------------------------------------------------- the agent's analysis output
analysis = json.loads(next(iter(sorted((ROOT / "outputs").glob("analysis_8725520_*_MHHW.json")))).read_text())

print(f"Observed file : {obs_file.relative_to(ROOT)}")
print(f"Request URL   : {obs_url}")
print(f"Window        : {analysis['window']}   datum {analysis['datum']}   {analysis['time_zone']}")
print()

# ---------------------------------------------------------------- completeness
start, end = datetime(2022, 9, 27, 0, 0), datetime(2022, 9, 30, 23, 54)
expected = []
t = start
while t <= end:
    expected.append(t.strftime("%Y-%m-%d %H:%M"))
    t += timedelta(minutes=6)
c = analysis["completeness"]
check("expected 6-min timestamps", len(expected), c["n_expected_6min_timestamps"])
check("observed missing", sum(1 for x in expected if x not in obs), c["n_observed_missing"])
check("predicted missing", sum(1 for x in expected if x not in prd), c["n_predicted_missing"])

# ---------------------------------------------------------------- peak and residual
peak_t = max(obs, key=obs.get)
check("peak observed (ft above MHHW)", obs[peak_t], analysis["peak_observed"]["value"])
check("peak observed time", peak_t, analysis["peak_observed"]["time"])
resid = {k: obs[k] - prd[k] for k in obs if k in prd}
rmax_t = max(resid, key=resid.get)
check("max residual (ft)", round(resid[rmax_t], 3), analysis["max_residual"]["value"])
check("max residual time", rmax_t, analysis["max_residual"]["time"])
rmin_t = min(resid, key=resid.get)
check("most negative residual (ft)", round(resid[rmin_t], 3), analysis["most_negative_residual"]["value"])
check("predicted tide at peak", prd[peak_t], analysis["peak_observed"]["predicted_tide_at_that_time"])

# ---------------------------------------------------------------- thresholds and durations
# published values are heights above station datum, so subtract the MHHW height
mhhw = datums["MHHW"]
for src in ("nws", "nos"):
    for level in ("minor", "moderate", "major"):
        thr = floods[f"{src}_{level}"] - mhhw
        n_above = sum(1 for v in obs.values() if v > thr)
        rep = analysis["exceedance"][src][level]
        check(f"{src} {level} threshold above MHHW", round(thr, 3), rep["threshold"], tol=0.0006)
        check(f"{src} {level} samples above", n_above, rep["n_samples_above"])
        check(f"{src} {level} hours above", round(n_above * 0.1, 2), rep["hours_above"], tol=0.005)
        times = sorted(k for k, v in obs.items() if v > thr)
        check(f"{src} {level} first time above", times[0] if times else None, rep["first_time_above"])
        check(f"{src} {level} last time above", times[-1] if times else None, rep["last_time_above"])

# ---------------------------------------------------------------- datum conversion
peak_navd88 = obs[peak_t] + mhhw - datums["NAVD88"]
check("peak in ft above NAVD88", round(peak_navd88, 2), analysis["peak_in_other_datums"]["ft_above_NAVD88"], tol=0.0051)

# ---------------------------------------------------------------- National Hurricane Center report, Table 3
txt_files = glob.glob(str(ROOT / "data" / "documents" / "AL092022_Ian_*.txt"))
if txt_files:
    txt = Path(txt_files[0]).read_text()
    m = re.search(r"Fort Myers \(FMRF1\)\s*\(.*?\)\s*([\d.]+)\s*([\d.]+)\s*([\d.]+)", txt, flags=re.S)
    if m:
        surge, tide, inund = map(float, m.groups())
        print()
        print(f"NHC Tropical Cyclone Report Table 3, Fort Myers (FMRF1): storm surge {surge}, storm tide {tide} (NAVD88), "
              f"inundation proxy {inund} (above MHHW)")
        check("max residual vs NHC storm surge", round(resid[rmax_t], 2), surge, tol=0.0101)
        check("peak vs NHC inundation proxy (ft above MHHW)", round(obs[peak_t], 2), inund, tol=0.0101)
        check("peak vs NHC storm tide (ft NAVD88, datums rounded to 0.01)", round(peak_navd88, 2), tide, tol=0.0101)
else:
    print("\n(NHC report text not cached, skipping the Table 3 comparison)")

# ---------------------------------------------------------------- USGS high-water marks
hwm_file, hwm_url = find_raw("/hwms/filteredhwms/")
marks = [m for m in json.loads(hwm_file.read_text()) if (m.get("county_name") or "").lower() == "lee county"]
rank = {"excellent": 0, "good": 1, "fair": 2}


def keep(m):
    q = (m.get("hwm_quality_name") or "").split(":")[0].lower()
    return q in rank and m.get("elev_ft") not in ("", None)


def km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def inside(area, m):
    la, lo = float(m["latitude_dd"]), float(m["longitude_dd"])
    if area == "lee_county":
        return True
    if area == "gauge_vicinity_5km":
        return km(26.6478, -81.8711, la, lo) <= 5.0
    if area == "fort_myers_beach":
        return 26.38 <= la <= 26.47 and -81.98 <= lo <= -81.88
    if area == "sanibel_captiva":
        return (26.38 <= la <= 26.475 and -82.20 <= lo <= -82.00) or (26.475 <= la <= 26.56 and -82.26 <= lo <= -82.15)


trace = json.loads(next(iter(sorted((ROOT / "outputs").glob("main_tool_trace.json")))).read_text())
reported = {t["input"]["area"]: t["result"] for t in trace if t["tool"] == "fetch_usgs_high_water_marks" and not t["is_error"]}
print()
print(f"USGS STN request: {hwm_url}")
gauge_peak_navd88 = round(peak_navd88, 2)
for area, rep in reported.items():
    sel = [m for m in marks if inside(area, m) and keep(m)]
    elev = [float(m["elev_ft"]) for m in sel]
    agl = [float(m["height_above_ground"]) for m in sel if m.get("height_above_ground") not in ("", None)]
    check(f"HWM {area}: marks used", len(sel), rep["counts"]["marks_used_after_quality_filter"])
    check(f"HWM {area}: median elevation (ft NAVD88)", round(statistics.median(elev), 2), rep["elevation_ft_NAVD88"]["median"], tol=0.0051)
    check(f"HWM {area}: max elevation (ft NAVD88)", max(elev), rep["elevation_ft_NAVD88"]["max"], tol=0.0051)
    check(f"HWM {area}: median height above ground (ft)", round(statistics.median(agl), 2), rep["height_above_ground_ft"]["median"], tol=0.0051)
    comp = rep.get("comparison_with_fort_myers_gauge")
    if isinstance(comp, dict):
        check(f"HWM {area}: marks higher than gauge peak", sum(1 for e in elev if e > gauge_peak_navd88), comp["n_marks_higher_than_gauge_peak"])

# the one mark the NHC report lists near the gauge (Table 3, FLLEE32905)
ref = [m for m in marks if m.get("site_number") == "FLLEE32905"]
if ref:
    print()
    print("NHC Table 3 lists USGS mark FLLEE32905 at 7.7 ft (NAVD88) and 4.43 ft above ground")
    check("FLLEE32905 elevation (ft NAVD88) vs NHC Table 3", float(ref[0]["elev_ft"]), 7.7, tol=0.0051)
    check("FLLEE32905 height above ground (ft) vs NHC Table 3", float(ref[0]["height_above_ground"]), 4.43, tol=0.0051)

print()
print(f"{sum(results)} of {len(results)} checks passed")
sys.exit(0 if all(results) else 1)
