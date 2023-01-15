
@REM To Run: scripts\m3\run.bat
@REM Defaults: --execution_mode=native --engine=ray
@REM python scripts/m3/experiment.py --model=ets --ts_category=Other

@echo OFF
for %%c in (Other Yearly Quarterly Monthly) do (
    @REM @REM Statistical Models (single engine) ----
    @REM for %%m in (grand_means naive snaive polytrend croston arima ets exp_smooth theta) do (
    @REM     python scripts/m3/experiment.py --model=%%m --ts_category=%%c
    @REM )

    @REM Statistical Models (multiple engines) ----
    for %%m in (auto_arima) do (
        for %%e in (pmdarima statsforecast) do (
            python scripts/m3/experiment.py --model=%%m --model_engine=%%e --ts_category=%%c
        )
    )

    @REM @REM Reduced Regression Models (Linear) ----
    @REM for %%m in (lr_cds_dt ridge_cds_dt lasso_cds_dt lar_cds_dt llar_cds_dt en_cds_dt br_cds_dt huber_cds_dt omp_cds_dt par_cds_dt) do (
    @REM     python scripts/m3/experiment.py --model=%%m --ts_category=%%c
    @REM )

    @REM @REM Reduced Regression Models (Tree Based + Others) ----
    @REM for %%m in (et_cds_dt dt_cds_dt rf_cds_dt ada_cds_dt gbr_cds_dt lightgbm_cds_dt knn_cds_dt) do (
    @REM     python scripts/m3/experiment.py --model=%%m --ts_category=%%c
    @REM )

    @REM @REM Slow Models ----
    @REM for %%m in (prophet) do (
    @REM     python scripts/m3/experiment.py --model=%%m --ts_category=%%c
    @REM )

    @REM Slow Models + Uses too much memory ----
    @REM for %%m in (auto_arima) do (
    @REM     python scripts/m3/experiment.py --model=%%m --ts_category=%%c
    @REM )

    @REM Really Slow Models ----
    @REM for %%m in (bats tbats) do (
    @REM     python scripts/m3/experiment.py --model=%%m --ts_category=%%c
    @REM )
)