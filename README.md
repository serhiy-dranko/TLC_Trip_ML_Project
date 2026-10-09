# NYC Taxi Demand Forecasting

Predicts the number of taxi pickups per **pickup zone per hour** in New York City, using past demand (lags), calendar features and hourly weather. The whole project runs from raw data to a final test score with one command.

**Result:** on the locked June test month the model has an MAE of **3.55 trips per zone-hour**, versus **4.68** for the baseline "same zone, same hour last week" (24% lower).

---

## Data sources

| Data | Source | Used for |
|---|---|---|
| Trip records (yellow, green, FHV), Jan-Jun 2026 | [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) | Target: pickups per zone-hour |
| Taxi zone lookup | Same TLC page ("Taxi Zone Lookup Table") | Zone -> Borough |
| Hourly weather | [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api), one point for the whole city (latitude / longitude in `config.yaml`) | `temperature_2m`, `precipitation`, `snowfall`, `wind_speed_10m` |
| Holiday calendar | Python [`holidays`](https://pypi.org/project/holidays/) package, US holidays with `state="NY"` | `is_holiday` flag (see Limitations) |

Raw trip files and the zone lookup are downloaded by the `ingest` stage and skipped if they already exist. Weather is fetched from the API on every `ingest` run, so an internet connection is always required.

---

## How to run

**Setup** (tested with Python 3.13):

```powershell
python -m venv venv
venv\Scripts\activate          # macOS / Linux: source venv/bin/activate
pip install -r requirements.txt
```

**Run everything from scratch** (about 3 minutes on a laptop, longer on the first run because of downloads):

```powershell
python run_pipeline.py --stage all      # ingest -> clean -> features -> train
```

**Final test evaluation, once:**

```powershell
python run_pipeline.py --stage test
```

The test stage is deliberately not part of `all`. It refuses to run again if `results/test_results.csv` exists, so the test month is evaluated once and nothing is tuned afterwards.

**Single stages:** `ingest`, `validate`, `audit`, `joins`, `clean`, `features`, `train`, `test`, `predict`.
A failed stage stops the run, logs the stage name and full traceback to `results/pipeline.log`, and exits with a non-zero code.

Repeated runs give the same validation MAE (3.6562); randomness is fixed with `random_state=42`.

---

## Project structure

```
week-5/
├── run_pipeline.py        # one entry point: --stage all | <stage>
├── config.yaml            # paths, time windows, model parameters
├── requirements.txt       # pinned dependencies
├── decision_log.md        # detailed record of decisions and the test result
├── README.md
├── src/
│   ├── utils.py           # config loading, logging setup
│   ├── ingest.py          # download trips, zones, weather, holidays -> interim tables
│   ├── validate.py        # source validation checks
│   ├── audit.py           # data-quality audit
│   ├── joins_check.py     # join-readiness report
│   ├── clean.py           # cleaning rules + cleaning report
│   ├── features.py        # zone-hour aggregation, lags, rolling mean, splits
│   ├── train.py           # model comparison, selection, error breakdowns, saving
│   ├── evaluate.py        # final test evaluation (run once)
│   └── predict.py         # predictions for the test file
├── data/
│   ├── raw/               # downloaded files
│   ├── interim/           # parquet tables per source and month
│   └── processed/         # cleaned monthly trips, train / val / test parquet
├── models/
│   ├── demand_model.joblib
│   └── model_metadata.json
└── results/
    ├── pipeline.log
    ├── data_audit.csv, join_readiness.md, cleaning_report.csv
    ├── model_comparison.csv, test_results.csv
    ├── mae_by_hour.csv, mae_by_borough.csv, mae_by_weather.csv
    └── mae_by_hour_dual_axis.png, mae_by_borough.png, mae_by_weather.png
```

---

## Method

- **Grain:** one row per (pickup zone, hour). Zone-hours with no trips are kept as demand 0 (265 zones x 4,344 hours).
- **Split by time:** train Jan-Apr, validation May, test June (locked). Train 718,680 rows, validation 197,160, test 190,800. The first week is dropped because `lag_168` needs a week of history.
- **Features (14):** `hour`, `day_of_week`, `month`, `is_weekend`, `temperature_2m`, `precipitation`, `snowfall`, `wind_speed_10m`, `lag_1`, `lag_24`, `lag_168`, `rolling_24`, `Borough`, `PULocationID`.
- **Preprocessing:** median imputation and scaling for numeric columns, one-hot encoding for zone and borough.
- **Models compared:** Ridge and HistGradientBoosting, with expanding-window `TimeSeriesSplit` (3 folds). The winner is chosen by validation MAE; predictions are clipped at 0.
- **Baseline:** demand in the same zone at the same hour one week earlier (`lag_168`).
- **Metric:** MAE, in trips per zone-hour.

---

## Results

| | Validation (May) | Test (June) |
|---|---|---|
| **HistGradientBoosting** | 3.6562 | **3.5475** |
| Baseline (same zone, same hour last week) | 5.0987 | 4.6806 |
| Model vs baseline | 28.3% lower | 24.2% lower |

The test MAE is slightly lower than validation (gap -0.11). The baseline also scored lower in June, so June seems to be an easier month to forecast; this is not evidence that the model generalises better. A single test month is not enough to claim performance in other seasons.

**Where the model errs (validation):**
- **By hour:** MAE is lowest at night (about 1.5 at 03:00) and highest in the evening (about 5.5 at 22:00).
- **By borough:** Manhattan has by far the largest MAE (about 10) while the other boroughs are below 2. MAE is in absolute trips, so high-volume zones dominate it.
- **By weather:** MAE is higher in rain than in dry hours (about 4.2 vs 3.6).

Full numbers: `results/test_results.csv`, `results/model_comparison.csv`, `results/mae_by_*.csv`.

---

## Limitations and known issues

1. **Six months of data.** Jan-Jun 2026 does not cover a full year of seasons, and the test set is a single month.
2. **Snow is only in training.** Snowfall occurs in the training months but not in validation (May), so the error under snow was never measured.
3. **Weather is treated as known.** The model uses observed weather; a real deployment would use forecasts, which are less accurate.
4. **One weather point for the whole city.** Weather comes from a single latitude / longitude, so local differences between zones are not captured.
5. **FHV demand is under-counted.** About 85-92% of FHV rows have no pickup zone and are dropped, so FHV trips are only partly represented.
6. **Holidays are not used by the model.** `is_holiday` is created in `features.py`, but it is not in the model's feature list, so holidays have no direct effect on the predictions.
7. **Absolute error is dominated by busy zones.** MAE is not normalised by zone volume, so Manhattan drives the overall number.

---

## Reusing this template

**What stays the same**
- the stage runner (`run_pipeline.py`) with ordered stages, timing and fail-fast error handling
- the validation, audit and join-check modules
- logging to `results/pipeline.log`
- the train / evaluate / save structure, including the run-once test stage
- this README format and the decision log

**What changes for a new project**
- `config.yaml`: paths, time windows, model parameters
- `src/ingest.py`: data sources and download logic
- `src/clean.py`: cleaning rules for the new data
- `src/features.py`: features and the target definition. The train / val / test months are hard-coded in `time_split` (and the dates in the model metadata in `train.py`); move them into `config.yaml`
- `features.py` appends the baseline to `decision_log.md` on every run, which creates duplicate entries; write it to `results/` instead
- the baseline, the limitations and the results sections
