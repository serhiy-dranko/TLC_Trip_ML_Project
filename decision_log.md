 

## Join plan
- trips -> zone lookup: trip grain (1 row per trip) vs zone grain (1 row per LocationID); key PULocationID = LocationID; many_to_one, row count unchanged.
- (zone, hour) demand -> weather: (zone, hour) grain vs hour grain; key pickup_hour = weather datetime; many_to_one, row count unchanged (= number of zone-hour rows).
- (zone, hour) demand -> holidays: (zone, hour) grain vs date grain; key date(pickup_hour) = holiday date; many_to_one, row count unchanged; non-holidays get null -> is_holiday flag.
- Weather is joined AFTER aggregating trips to (zone, hour): joining at trip level would repeat the same hourly values millions of times.


## Day 2 Baseline
Baseline MAE (lag_168): 5.0987


## Final test evaluation (2026-10-08)

Run once on the locked June test set. No changes to the model, features
or parameters were made afterwards.

| Model | Validation MAE (May) | Test MAE (June) |
|---|---|---|
| hist_gradient_boosting | 3.6562 | 3.5475 |
| Baseline (same zone, same hour last week) | 5.0987 | 4.6806 |

- Gap (test - val): -0.1087. The model is slightly better on test than on validation.
- On test the model has 24.2% lower MAE than the baseline (28.3% on validation).

**Interpretation:** Both the model and the baseline scored lower MAE in June,
so June appears to be an easier month to forecast, rather than the model
improving. A likely (untested) explanation is more stable summer demand and
weather. The advantage over the baseline shrank slightly, from 28.3% to 24.2%.
One month of test data is not enough to claim the model generalises to other seasons.
