# PyCaret Benchmarks

The purpose of this repository is to assist with the benchmarking of the pycaret module. Currently, the benchmarks are only for time series forecasting and feature extraction. The benchmarking is done using the M3 and M4 competition dataset, but can be extended to other datasets as well.

## Result Summary

[M3 Benchmark Summary](http://htmlpreview.github.io/?data/m3/M3_all_results_summary.html)
[M4 Benchmark Summary](http://htmlpreview.github.io/?data/m4/M4_all_results_summary.html)

# Time Series Benchmarking (Windows)

The benchmarking is done in 4 steps as described below.

## Step 1: Create forecasts

You can edit the batch file include all the models and categories you want to benchmark. Then run using the following command. This will execute `experiment.py` in a loop for all combinations.

```
scripts\m3\run_experiments.bat
```

This will create 2 files per model-category combination - one with the predictions and the other with the run statistics.

## Steps 2, 3, & 4

Steps 2, 3, 4 have also been complied into a single script which can be run as follows (you can edit this to select only a subset of datasets or plots).

```
scripts\mseries\run_postprocessing.bat
```

Or if you want, you can run them individually as follows.

### Step 2: Evaluate results (metrics, time, etc.)

Once you have run Step 1 for all the combinations of interest, you can run the evaluation script to compile the benchmark results (metrics, time, etc.)

```
python scripts/mseries/evaluation.py --dataset=M3
```

This will produce a file called `data\m3\current_evaluation_full.csv` and `data\m3\current_evaluation_full.csv` with the summary of the benchmark.

### Step 3: Update running metrics (i.e. combine previous results with current results)

Next, you can combine these results with the already run benchmarks in the past.

```
python scripts/mseries/update.py --dataset=M3
```

### Step 4: Plot results

Finally, you can plot the results using plotly.

```
python scripts/mseries/plot.py --dataset=M3 --group=Monthly
```


# Time Series Feature Extraction (Windows)

This library can also be used to extract features from time series data. The following command will extract features from the M3 dataset and save them to a csv file. This can be useful to evaluate the characteristics of the data before modeling and deciding on the appropriate settings to use for modeling (especially at scale).

```
scripts\mseries\run_extract_properties.bat
```

* This will save the results into a folder as such `data\m4\results\properties`
* The captured properties include
    - Total length of series (train + test)
    - Total length of training data
    - Total length of test data
    - Whether the data is strictly positive or not
    - Whether the data is white noise or not
    - The recommended value of 'd' and 'D' to use for models like ARIMA
    - Whether the data has seasonality or not
    - The candidate seasonal periods to be tested
    - The significant seasonal periods present
    - All the seasonal periods to use for modeling (when models accept multiple seasonal periods)
    - The primary seasonal period to use for modeling (when models only accept a single seasonality)
    - The significant seasonal periods present when harmonics are removed.
    - All the seasonal periods to use for modeling with harmonics removed (when models accept multiple seasonal periods)
    - The primary seasonal period to use for modeling when harmonics are removed (when models only accept a single seasonality)