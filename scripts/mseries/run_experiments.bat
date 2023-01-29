@REM To Run: scripts\mseries\run_experiments.bat
@REM Defaults: --execution_mode=native --engine=ray
@REM python scripts/mseries/experiment.py --model=ets --group=Other

@echo OFF

for %%d in (M4) do (
    for %%g in (Weekly Hourly) do (
        @REM @REM Statistical Models (single engine) ----
        @REM for %%m in (exp_smooth) do (
        @REM     python scripts/mseries/experiment.py --dataset=%%d --model=%%m --group=%%g
        @REM )

        @REM @REM Statistical Models (single engine) ----
        @REM for %%m in (grand_means naive snaive polytrend croston arima ets exp_smooth theta) do (
        @REM     python scripts/mseries/experiment.py --dataset=%%d --model=%%m --group=%%g
        @REM )

        @REM @REM Reduced Regression Models (Linear) ----
        @REM for %%m in (lr_cds_dt ridge_cds_dt lasso_cds_dt lar_cds_dt llar_cds_dt en_cds_dt br_cds_dt huber_cds_dt omp_cds_dt par_cds_dt) do (
        @REM     python scripts/mseries/experiment.py --dataset=%%d --model=%%m --group=%%g
        @REM )

        @REM @REM Reduced Regression Models (Tree Based + Others) ----
        @REM for %%m in (et_cds_dt dt_cds_dt rf_cds_dt ada_cds_dt gbr_cds_dt lightgbm_cds_dt xgboost_cds_dt catboost_cds_dt knn_cds_dt) do (
        @REM     python scripts/mseries/experiment.py --dataset=%%d --model=%%m --group=%%g
        @REM )

        @REM @REM Slow Models ----
        @REM for %%m in (prophet) do (
        @REM     python scripts/mseries/experiment.py --dataset=%%d --model=%%m --group=%%g
        @REM )

        @REM Slow Models + Uses too much memory ----
        @REM for %%m in (auto_arima) do (
        @REM     python scripts/mseries/experiment.py --dataset=%%d --model=%%m --group=%%g
        @REM )

        @REM Really Slow Models ----
        @REM for %%m in (bats tbats) do (
        @REM     python scripts/mseries/experiment.py --dataset=%%d --model=%%m --group=%%g
        @REM )
    )
)


for %%d in (M3) do (
    for %%g in (Other Yearly Quarterly Monthly) do (
        @REM Statistical Models (single engine) ----
        for %%m in (exp_smooth) do (
            python scripts/mseries/experiment.py --dataset=%%d --model=%%m --group=%%g
        )
        
        @REM @REM Statistical Models (single engine) ----
        @REM for %%m in (grand_means naive snaive polytrend croston arima ets exp_smooth theta) do (
        @REM     python scripts/mseries/experiment.py --dataset=%%d --model=%%m --group=%%g
        @REM )

@REM         @REM Reduced Regression Models (Linear) ----
@REM         for %%m in (lr_cds_dt ridge_cds_dt lasso_cds_dt lar_cds_dt llar_cds_dt en_cds_dt br_cds_dt huber_cds_dt omp_cds_dt par_cds_dt) do (
@REM             python scripts/mseries/experiment.py --dataset=%%d --model=%%m --group=%%g
@REM         )

@REM         @REM Reduced Regression Models (Tree Based + Others) ----
@REM         for %%m in (et_cds_dt dt_cds_dt rf_cds_dt ada_cds_dt gbr_cds_dt lightgbm_cds_dt xgboost_cds_dt catboost_cds_dt knn_cds_dt) do (
@REM             python scripts/mseries/experiment.py --dataset=%%d --model=%%m --group=%%g
@REM         )

@REM         @REM Slow Models ----
@REM         for %%m in (prophet) do (
@REM             python scripts/mseries/experiment.py --dataset=%%d --model=%%m --group=%%g
@REM         )

@REM         @REM Slow Models + Uses too much memory ----
@REM         @REM for %%m in (auto_arima) do (
@REM         @REM     python scripts/mseries/experiment.py --dataset=%%d --model=%%m --group=%%g
@REM         @REM )

@REM         @REM Really Slow Models ----
@REM         @REM for %%m in (bats tbats) do (
@REM         @REM     python scripts/mseries/experiment.py --dataset=%%d --model=%%m --group=%%g
    )
)

@REM     @REM @REM Disabling Monthly for now since too slow
@REM     @REM for %%g in (Other Yearly Quarterly Monthly) do (
@REM     @REM     @REM Statistical Models (multiple engines) ----
@REM     @REM     for %%m in (auto_arima) do (
@REM     @REM         for %%e in (pmdarima statsforecast) do (
@REM     @REM             python scripts/mseries/experiment.py --dataset=%%d --model=%%m --model_engine=%%e --group=%%g
@REM     @REM         )
@REM     @REM     )
@REM     @REM )
@REM )