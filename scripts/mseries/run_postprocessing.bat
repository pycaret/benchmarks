@REM To Run: scripts\mseries\run_postprocessing.bat

@echo OFF

@REM @REM M3 dataset ----
@REM for %%d in (M3) do (
@REM     python scripts/mseries/evaluation.py --dataset=%%d
@REM     python scripts/mseries/update.py --dataset=%%d
@REM     for %%g in (Other Yearly Quarterly Monthly) do (
@REM         python scripts/mseries/plot.py --dataset=%%d --group=%%g
@REM     )
@REM )

@REM M4 dataset ----
for %%d in (M4) do (
    python scripts/mseries/evaluation.py --dataset=%%d
    python scripts/mseries/update.py --dataset=%%d
    for %%g in (Weekly Hourly) do (
        python scripts/mseries/plot.py --dataset=%%d --group=%%g
    )
)

