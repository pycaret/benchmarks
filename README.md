# PyCaret Benchmarks

The purpose of this repository is to assist with the benchmarking of the pycaret module


# Example Steps

## Time Series

### Windows machine

1. You can edit the batch file include all the models and categories you want to benchmark. Then run using the following command. This will execute `experiment.py` in a loop for all combinations.

```
scripts\m3\run_experiments.bat
```

This will create 2 files per model-category combination - one with the predictions and the other with the run statistics.

2. Once you have run Step 1 for all the combinations of interest, you can run the evaluation script to compile the benchmark results (metrics, time, etc.)

```
python scripts/m3/evaluation.py --dataset="M3"
```

This will produce a file called `data\m3\current_evaluation_full.csv` and `data\m3\current_evaluation_full.csv` with the summary of the benchmark.

3. Finally, you can combine these results with the already run benchmarks in the past.

```
python scripts/m3/update.py --dataset="M3"
```