# Assessment: coastal flooding in Lee County during Hurricane Ian

Question: How severe was coastal flooding in Lee County, Florida, during Hurricane Ian, and what were the potential implications for local transportation infrastructure?

Hurricane Ian caused major coastal flooding at the Fort Myers gauge (NOAA 8725520). The peak water level was 7.26 feet above MHHW at 6:30 PM EDT on September 28, 2022. The level stayed above the NWS major flood threshold of 5.0 feet for 7.5 hours, the moderate threshold of 3.5 feet for 13.7 hours, and the minor threshold of 2.0 feet for 20.3 hours. These are current thresholds applied retrospectively. The maximum storm residual, observed minus predicted tide, was 7.25 feet while the predicted tide was near zero. No observations were missing.

One gauge does not represent all of Lee County. USGS surveyed high-water marks within 5 km of the gauge had a median elevation of 7.5 feet above NAVD88, close to the gauge peak of 7.54 feet above NAVD88. Marks at Fort Myers Beach had a median of 12.3 feet and marks at Sanibel and Captiva a median of 11.1 feet. The National Hurricane Center report states that roads and bridges to Sanibel and Pine Islands were washed away. Such extreme levels could potentially have flooded low-lying roadways and left causeways impassable. The gauge measures water-surface elevation (the level above a datum), and the storm residual follows from it. Neither is inundation depth above ground, so the gauge cannot give depth on streets or at buildings.

Figure: `outputs/fort_myers_ian_water_levels.png` (observed and predicted water level, NWS flood thresholds and residual, feet above MHHW, local time EDT).

## Data notes

All gauge values are in feet above MHHW and local clock time (LST/LDT, which is EDT on these dates). The USGS marks are in feet above NAVD88, and the gauge peak was converted to NAVD88 with the CO-OPS datum heights. All 960 six-minute observations and all 960 predictions were present, and nothing was interpolated. The flood thresholds are what NOAA publishes now, and applying them to a 2022 event is retrospective. Water-surface elevation, storm residual and inundation depth above ground are different quantities. The gauge gives the first two only. The high-water mark medians above are also water-surface elevations, not depths. Some marks also record their height above ground at that spot, which is closer to inundation depth, but only for that spot. The NWS minor, moderate and major coastal flood thresholds are water levels set by the local NWS forecast office with emergency managers from years of impact monitoring, such as road closures and flooding of low areas. They come from the NOAA CO-OPS metadata API (reference 3). One hurricane describes the flooding of this event only and does not by itself give long-term flood probabilities.

## References

1. NOAA CO-OPS Data API, observed water level, station 8725520 (Fort Myers), 2022-09-27 to 2022-09-30, MHHW, lst_ldt, feet. Retrieved 2026-10-25.
   `https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?product=water_level&application=cgn6905_hw2_agent&begin_date=20220927&end_date=20220930&datum=MHHW&station=8725520&time_zone=lst_ldt&units=english&format=json`
2. NOAA CO-OPS Data API, predicted tide, same station, datum and time zone, 6-minute interval.
   `https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?product=predictions&application=cgn6905_hw2_agent&begin_date=20220927&end_date=20220930&datum=MHHW&station=8725520&time_zone=lst_ldt&units=english&format=json&interval=6`
3. NOAA CO-OPS Metadata API, flood levels and datums for station 8725520.
   `https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/8725520/floodlevels.json?units=english`
   `https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/8725520/datums.json?units=english`
4. USGS Short-Term Network (STN) high-water marks, event "2022 Ian" (event 325), Florida, filtered to Lee County in the agent. Retrieved 2026-10-25.
   `https://api.waterdata.usgs.gov/stnservices/hwms/filteredhwms/?event=325&states=FL`
5. National Hurricane Center, Tropical Cyclone Report, Hurricane Ian (AL092022), version dated 12 March 2026. The statement about roads and bridges to Sanibel and Pine Islands is on page 13. Table 3 on page 24 lists the Fort Myers gauge and agrees with the gauge values above.
   `https://www.nhc.noaa.gov/data/tcr/AL092022_Ian.pdf`

## How this text relates to the agent's answer

The text above is the final answer from the main run (`outputs/main_answer.md`) with light edits for clarity. I replaced the opening phrase "catastrophic and highly variable" with a plain statement of what the gauge showed, moved the sentence about the thresholds so it comes after the thresholds, added the gauge ID and EDT, and changed "would have potentially inundated low-lying roadways and rendered causeways impassable" to "could potentially have flooded low-lying roadways and left causeways impassable". I also shortened a few sentences and rewrote the last two sentences in plain words. No number was changed. The numbers were checked against the saved data by `verify.py`.
