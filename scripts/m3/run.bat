
@REM To Run: scripts\m3\run.bat
@REM Defaults: --execution_mode=native --engine=ray 
@REM python scripts/m3/experiment.py --model=ets --ts_category=Other 

@echo OFF
for %%c in (Other Monthly Quarterly Yearly) do (
    @REM Statistical Models ----
    for %%m in (grand_means naive snaive polytrend croston arima auto_arima ets exp_smooth theta) do (
        python scripts/m3/experiment.py --model=%%m --ts_category=%%c
    )

    @REM Reduced Regression Models (Linear) ----
    for %%m in (lr_cds_dt ridge_cds_dt lasso_cds_dt lar_cds_dt llar_cds_dt en_cds_dt br_cds_dt huber_cds_dt omp_cds_dt par_cds_dt) do (
        python scripts/m3/experiment.py --model=%%m --ts_category=%%c
    )

    @REM Reduced Regression Models (Tree Based + Others) ----
    for %%m in (et_cds_dt dt_cds_dt rf_cds_dt ada_cds_dt gbr_cds_dt lightgbm_cds_dt knn_cds_dt) do (
        python scripts/m3/experiment.py --model=%%m --ts_category=%%c
    )
)