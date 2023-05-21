@REM To Run: scripts\mseries\run_experiment_ens_freq.bat
@REM Defaults: --execution_mode=native --engine=ray
@REM python scripts/mseries/run_experiment_ens_freq.py --model=ets --group=Other

@echo OFF

for %%d in (M3) do (
    for %%g in (Other Yearly Quarterly Monthly) do (
        for %%w in (True False) do (
            for %%f in (1) do (
                for %%c in (3 5) do (
                    for %%h in (False) do (
                        for %%o in (harmonic_max) do (
                        @REM for %%o in (harmonic_max harmonic_strength raw_strength) do (
                            @REM Statistical Models (single engine) ----
                            for %%m in (arima) do (
                                python scripts\mseries\experiment_ens_freq.py ^
                                --dataset=%%d ^
                                --model=%%m ^
                                --group=%%g ^
                                --max_models=%%c ^
                                --weighted=%%w ^
                                --fold=%%f ^
                                --remove_harmonics=%%h ^
                                --harmonic_order_method=%%o
                            )
                        )
                    )
                )
            )
        )
    )
)

for %%d in (M4) do (
    for %%g in (Weekly Hourly) do (
        for %%w in (True False) do (
            for %%f in (1) do (
                for %%c in (3 5) do (
                    for %%h in (False) do (
                        for %%o in (harmonic_max) do (
                        @REM for %%o in (harmonic_max harmonic_strength raw_strength) do (
                            @REM Statistical Models (single engine) ----
                            for %%m in (arima) do (
                                python scripts\mseries\experiment_ens_freq.py ^
                                --dataset=%%d ^
                                --model=%%m ^
                                --group=%%g ^
                                --max_models=%%c ^
                                --weighted=%%w ^
                                --fold=%%f ^
                                --remove_harmonics=%%h ^
                                --harmonic_order_method=%%o
                            )
                        )
                    )
                )
            )
        )
    )
)