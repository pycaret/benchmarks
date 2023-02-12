@REM To Run: scripts\mseries\run_postprocessing.bat

@echo OFF

@REM M3 dataset ----
for %%d in (M3) do (
    python scripts/mseries/evaluation.py --dataset=%%d
    python scripts/mseries/update.py --dataset=%%d
    for %%g in (Other Yearly Quarterly Monthly) do (
        for %%p in (smape mape primary_model_per) do (
            python scripts/mseries/plot.py --dataset=%%d --group=%%g --metric=%%p
        )
    )
)

@REM M4 dataset ----
for %%d in (M4) do (
    python scripts/mseries/evaluation.py --dataset=%%d
    python scripts/mseries/update.py --dataset=%%d
    for %%g in (Weekly Hourly) do (
        for %%p in (smape mape mase owa primary_model_per) do (
            python scripts/mseries/plot.py --dataset=%%d --group=%%g --metric=%%p
        )
    )
)

