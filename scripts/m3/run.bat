
@REM To Run: scripts\m3\run.bat
@REM Defaults: --execution_mode=native --engine=ray 
@REM python scripts/m3/experiment.py --model=ets --ts_category=Other 

@echo OFF
for %%c in (Other Monthly Quarterly Yearly) do (
    for %%m in (grand_means naive snaive polytrend croston arima auto_arima ets exp_smooth theta) do (
        python scripts/m3/experiment.py --model=%%m --ts_category=%%c
    )
)

