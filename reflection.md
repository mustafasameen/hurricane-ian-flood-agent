# Reflection

I built the agent around the Anthropic Messages API with claude-sonnet-4-5. The loop in agent.py sends the question, a system prompt and seven tool definitions to the model. When the model asks for a tool, the loop runs it, returns the result and calls the model again until it writes an answer. Five tools handle the gauge data: fetch_water_levels, fetch_tide_predictions, fetch_flood_thresholds, analyze_flooding and make_plot. Two give regional context: fetch_usgs_high_water_marks and fetch_document, which only opens NOAA, NWS and USGS pages.

The model chooses the tools from their descriptions and the question. For the broad question it used all seven. For the Fort Myers Beach and Sanibel question it skipped the plot and the report. The fetch tools save raw responses and the analysis tools only read those files, so the model has to retrieve before it analyzes. Numbers come only from tool results, and each log ends with a check of every number in the answer against the tool output.

The main strength is traceability. Every number maps to a saved request, and verify.py reproduced the peak, the residual and the exceedance hours without importing my tools. The limits are practical. The model ignored my word limit until I added a length check, and it only made the plot after I asked for it in the prompt. Answers change in wording from run to run. The flood thresholds do not state their datum, so I inferred it from NOAA's dashboard code, and my Fort Myers Beach and Sanibel areas are rough boxes.

One improvement would be a ground-elevation tool, for example the USGS 3DEP service, plus a road and bridge layer. The agent could then name which road segments lie below a given water level instead of describing potential effects in general terms.
