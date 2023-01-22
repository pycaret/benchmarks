@REM To Run: scripts\mseries\run_extract_properties.bat

@echo OFF
for %%d in (M3) do (
    for %%g in (Other Yearly Quarterly Monthly) do (
        python scripts/mseries/extract_properties.py --dataset=%%d --group=%%g
    )
)

for %%d in (M4) do (
    for %%g in (Weekly Hourly) do (
        python scripts/mseries/extract_properties.py --dataset=%%d --group=%%g
    )
)
