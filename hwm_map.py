#!/usr/bin/env python
"""Optional extension: map of USGS high-water marks in Lee County next to the Fort Myers gauge peak.

Reads outputs/usgs_hwm_lee_county.csv (written by the agent's USGS tool) and the datum heights
cached under data/. Marks are in feet above NAVD88. The gauge peak is in feet above MHHW, so it
is converted to NAVD88 with the CO-OPS datum heights before it is compared with the marks:
    NAVD88 value = MHHW value + (MHHW height above station datum) - (NAVD88 height above station datum)

Run:  python hwm_map.py     ->  outputs/lee_county_hwm_map.png
There is no basemap, because the homework limits network access to NOAA and USGS hosts.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
marks = pd.read_csv(ROOT / "outputs" / "usgs_hwm_lee_county.csv")
marks = marks.dropna(subset=["elev_ft"])

# gauge peak and the datum conversion, read from the raw cached files (not typed in)
analysis = json.loads(next((ROOT / "outputs").glob("analysis_8725520_*_MHHW.json")).read_text())
peak_mhhw = analysis["peak_observed"]["value"]
datum_file = next(Path(json.loads(l)["file"]) for l in (ROOT / "data" / "request_log.jsonl").read_text().splitlines()
                  if "/datums.json" in json.loads(l)["url"])
datums = {d["name"]: d["value"] for d in json.loads((ROOT / datum_file).read_text())["datums"]}
offset = datums["MHHW"] - datums["NAVD88"]
peak_navd = peak_mhhw + offset
gauge_lat, gauge_lon = 26.6478, -81.8711

fig, (ax, axh) = plt.subplots(1, 2, figsize=(12.5, 7), gridspec_kw={"width_ratios": [1.25, 1]})
sc = ax.scatter(marks["longitude_dd"], marks["latitude_dd"], c=marks["elev_ft"], s=28, cmap="viridis_r",
                edgecolor="k", linewidth=0.3, vmin=2, vmax=14, zorder=3)
ax.plot(gauge_lon, gauge_lat, marker="*", ms=17, color="#d62728", mec="k", zorder=5, ls="none")
ax.annotate(f"Fort Myers gauge (NOAA 8725520)\npeak {peak_mhhw:.2f} ft above MHHW\n= {peak_navd:.2f} ft above NAVD88",
            xy=(gauge_lon, gauge_lat), xytext=(-175, 38), textcoords="offset points", ha="left", fontsize=9, color="#a01010",
            arrowprops=dict(arrowstyle="->", color="#a01010"))
for name, lon, lat, ha in [("Sanibel / Captiva", -82.13, 26.385, "center"), ("Fort Myers Beach", -81.96, 26.385, "left"),
                           ("Pine Island / Matlacha", -82.11, 26.615, "center"), ("Estero / Bonita", -81.76, 26.378, "center")]:
    ax.text(lon, lat, name, fontsize=9, color="0.25", ha=ha, style="italic")
ax.set_aspect(1 / np.cos(np.radians(26.5)))
ax.set_xlabel("Longitude (degrees)")
ax.set_ylabel("Latitude (degrees)")
ax.set_title("USGS high-water marks, Hurricane Ian, Lee County, FL", fontsize=11.5)
ax.grid(alpha=0.3)
cb = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
cb.set_label("Mark elevation (ft above NAVD88)")

bins = np.arange(2, 14.5, 0.5)
axh.hist(marks["elev_ft"], bins=bins, color="#4c78a8", edgecolor="white")
axh.axvline(peak_navd, color="#d62728", lw=2)
axh.text(peak_navd - 0.15, axh.get_ylim()[1] * 0.93,
         f"Fort Myers gauge peak\n{peak_navd:.2f} ft NAVD88\n({peak_mhhw:.2f} ft MHHW + {offset:.2f} ft)",
         color="#a01010", ha="right", va="top", fontsize=9.5)
axh.set_xlabel("Mark elevation (ft above NAVD88)")
axh.set_ylabel("Number of marks")
axh.set_title(f"{len(marks)} marks (quality fair or better)", fontsize=11.5)
axh.grid(alpha=0.3)

fig.text(0.01, 0.01,
         "Marks: USGS STN, event 2022 Ian, Lee County (api.waterdata.usgs.gov/stnservices). Gauge: NOAA CO-OPS 8725520, 6-min data.\n"
         f"Gauge peak converted from MHHW to NAVD88 with CO-OPS datum heights (MHHW - NAVD88 = {offset:.2f} ft).\n"
         "Each mark is one surveyed peak at one place, not a time series. Seed lines and mud lines can include wave effects. No basemap.",
         fontsize=8, color="0.3", va="bottom")
fig.subplots_adjust(left=0.07, right=0.97, top=0.94, bottom=0.14, wspace=0.18)
out = ROOT / "outputs" / "lee_county_hwm_map.png"
fig.savefig(out, dpi=150)
print(f"saved {out.relative_to(ROOT)}  (gauge peak {peak_mhhw} ft MHHW = {peak_navd:.3f} ft NAVD88, offset {offset:.2f} ft)")
