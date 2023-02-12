@REM To Run: scripts\mseries\run_experiments.bat
@REM Defaults: --execution_mode=native --engine=ray
@REM python scripts/mseries/experiment.py --model=ets --group=Other

@echo OFF

for %%d in (M3) do (
    for %%g in (Other Yearly Quarterly Monthly) do (
        @REM Statistical Models (single engine) ----
        for %%m in (grand_means naive snaive polytrend croston arima ets exp_smooth theta) do (
            python scripts/mseries/experiment.py --dataset=%%d --model=%%m --group=%%g
        )

        @REM Reduced Regression Models (Linear) ----
        for %%m in (lr_cds_dt ridge_cds_dt lasso_cds_dt lar_cds_dt llar_cds_dt en_cds_dt br_cds_dt huber_cds_dt omp_cds_dt par_cds_dt) do (
            python scripts/mseries/experiment.py --dataset=%%d --model=%%m --group=%%g
        )

        @REM Reduced Regression Models (Tree Based + Others) ----
        for %%m in (et_cds_dt dt_cds_dt rf_cds_dt ada_cds_dt gbr_cds_dt lightgbm_cds_dt xgboost_cds_dt catboost_cds_dt knn_cds_dt) do (
            python scripts/mseries/experiment.py --dataset=%%d --model=%%m --group=%%g
        )

        @REM Slow Models ----
        for %%m in (prophet) do (
            python scripts/mseries/experiment.py --dataset=%%d --model=%%m --group=%%g
        )

        @REM @REM Slow Models ----
        @REM Statistical Models (multiple engines) ----
        for %%m in (auto_arima) do (
            for %%e in (pmdarima statsforecast) do (
                python scripts/mseries/experiment.py --dataset=%%d --model=%%m --model_engine=%%e --group=%%g
            )
        )

        @REM Really Slow Models ----
        for %%m in (bats tbats) do (
            python scripts/mseries/experiment.py --dataset=%%d --model=%%m --group=%%g
        )
    )
)

for %%d in (M4) do (
    for %%g in (Weekly Hourly) do (
        @REM Statistical Models (single engine) ----
        for %%m in (grand_means naive snaive polytrend croston arima ets exp_smooth theta) do (
            python scripts/mseries/experiment.py --dataset=%%d --model=%%m --group=%%g
        )

        @REM Reduced Regression Models (Linear) ----
        for %%m in (lr_cds_dt ridge_cds_dt lasso_cds_dt lar_cds_dt llar_cds_dt en_cds_dt br_cds_dt huber_cds_dt omp_cds_dt par_cds_dt) do (
            python scripts/mseries/experiment.py --dataset=%%d --model=%%m --group=%%g
        )

        @REM Reduced Regression Models (Tree Based + Others) ----
        for %%m in (et_cds_dt dt_cds_dt rf_cds_dt ada_cds_dt gbr_cds_dt lightgbm_cds_dt xgboost_cds_dt catboost_cds_dt knn_cds_dt) do (
            python scripts/mseries/experiment.py --dataset=%%d --model=%%m --group=%%g
        )

        @REM Slow Models ----
        for %%m in (prophet) do (
            python scripts/mseries/experiment.py --dataset=%%d --model=%%m --group=%%g
        )

        @REM @REM Slow Models ----
        @REM Statistical Models (multiple engines) ----
        @REM NOTE: pmdarima is running out of memory with rc9 and M4
        for %%m in (auto_arima) do (
            for %%e in (pmdarima statsforecast) do (
                python scripts/mseries/experiment.py --dataset=%%d --model=%%m --model_engine=%%e --group=%%g
            )
        )

        @REM Really Slow Models ----
        @REM NOTE: TBATS is too slow for M4
        for %%m in (bats tbats) do (
            python scripts/mseries/experiment.py --dataset=%%d --model=%%m --group=%%g
        )
    )
)