# Join readiness

## Pickup zone found in lookup
- yellow: 100.00% of all rows, 100.00% of non-null rows (22,836,530 rows)
- green: 100.00% of all rows, 100.00% of non-null rows (255,175 rows)
- fhv: 13.05% of all rows, 100.00% of non-null rows (12,690,686 rows)

## Weather coverage
- expected 4344 hourly rows (24/day), found 4344
- missing hours: 0 []
- unexpected hours: 0, duplicated timestamps: 0

## Clock changes
- 2026-03-08: weather has 24 rows; hours present: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

## Join plan
- trips -> zone lookup: trip grain (1 row per trip) vs zone grain (1 row per LocationID); key PULocationID = LocationID; many_to_one, row count unchanged.
- (zone, hour) demand -> weather: (zone, hour) grain vs hour grain; key pickup_hour = weather datetime; many_to_one, row count unchanged (= number of zone-hour rows).
- (zone, hour) demand -> holidays: (zone, hour) grain vs date grain; key date(pickup_hour) = holiday date; many_to_one, row count unchanged; non-holidays get null -> is_holiday flag.
- Weather is joined AFTER aggregating trips to (zone, hour): joining at trip level would repeat the same hourly values millions of times.
