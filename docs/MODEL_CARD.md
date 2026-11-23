# Model card: outage-risk forecaster v1.0.0

## Model details

- **What it predicts.** The share of customers without power in a US county, x = customers out / customers served.
  Task A gives 48 hourly values from a 00 UTC issue; Task B gives 72 five-minute values from 00/06/12/18 UTC issues.
  Each value is mapped to the grid-supply risk of four AI data-center power architectures,
  P = 1 / (1 + exp(−4000 (x − x₀))), with x₀ from 0.100 % to 0.160 %.
- **Model.** Fixed-weight ensembles of a GRU history network, XGBoost Tweedie feature trees and (Task A) a Q95 tail
  classifier, with separate models for missing radar, weather or outage history. Diagrams and member details are in
  [MODEL.md](MODEL.md).
- **Version.** v1.0.0 (frozen 2026-10-29). Model files are verified by SHA-256 at load time.
- **Author.** [@lishuangyi](https://github.com/lishuangyi). Code licence MIT. Built for Huawei TechArena 2026
  (Topic 2, Challenge 2).

## Intended use

- Research and demonstration of point-in-time forecasting with public data.
- A decision-support signal for "how likely is grid supply to this county to be disturbed in the next hours", for
  example to plan backup generation at a data center.

**Out of scope.** Operational grid control, emergency response, and any county outside the five it was built for,
unless you retrain it. The forecasts are statistical; they do not replace utility or weather-service warnings.

## Data

| Source | Use | Licence |
|---|---|---|
| EAGLE-I county outages (ORNL), 15 min | outage history (input) and labels | CC BY 4.0 |
| NOAA HRRR forecasts | county-polygon statistics of 14 variables, runs 00Z and 18Z | public domain |
| NOAA MRMS composite reflectivity | county radar exposure, every 10 min | public domain |

- **Training.** Labels from 2022–2024 for 40 counties: the 5 forecast counties and 35 neighbours.
- **Selection.** June–August 2025.
- **Retrospective check.** The other 9 months of 2025; January–May also informed one fallback decision.
- **Time rule.** Each input row carries an availability time, and the models only read rows available at the issue
  time ([TEMPORAL_CORRECTNESS.md](TEMPORAL_CORRECTNESS.md)).

## Metrics

The challenge score per task: 100 × (0.75 × normalised AUPRC + 0.25 × (1 − MAE)) on the risk scale, pooled over
counties, architectures, issues and leads. The event label is a true risk of at least 0.5. Persistence is the
baseline.

## Performance (details in [RESULTS.md](RESULTS.md))

<!-- run-report:model-card -->
| | Task A | Task B |
|---|---:|---:|
| June–August 2025 (selection window) | 54.45 (persistence 38.69) | 75.13 (persistence 65.88) |
| other 9 months of 2025 (retrospective) | 55.00 (persistence 46.92) | 72.75 (persistence 66.59) |
<!-- /run-report:model-card -->

On 2025 the gain over persistence is clear: score difference A +11.0 [5.6, 16.1] and B +7.1 [5.2, 9.0] (paired
block-bootstrap 95 % intervals). Performance varies more across periods and counties than between models. A county and period with many events
scores 70–90, while a quiet one scores 25–40. The lead also matters: Task A falls from 78 (1–6 h) to 45 (25–48 h).

## Limitations and risks

- **Few places.** Five counties and one year of evaluation. Behaviour elsewhere is unknown.
- **Replay assumptions.** Publication times are assumed. Real delays cost about 5 points on Task A.
- **New outages are hard.** Outages that start after the issue time are hard to predict. Much of the skill comes from
  tracking outages already under way. In a budgeted alert replay (4 alerts per county per 30 days), the model's alerts
  were right more often than those of persistence or a gust score. It did not find clearly more new outages.
- **Over-confident at the top.** Predicted risk above 0.5 is over-confident, so do not read it as a probability.
  A separate event probability fitted on summer improves Task A on September–December and leaves Task B unchanged.
  It stays over-confident when the season changes ([RESULTS.md](RESULTS.md#11-event-probability-and-alert-budget)).
- **Long events.** Multi-day events, such as the January 2025 Los Angeles wildfires, can be handled worse than by
  persistence.
- **No 5-minute dynamics.** Task B models the 15-minute series and does not model 5-minute dynamics.
- **Training code.** The frozen weights come from experiment code that is not published. The reproducible path is
  the simpler reference trainer ([TRAINING.md](TRAINING.md)).

## Ethical considerations

The data are public and aggregated to counties; there is no personal information. Outage risk is unevenly
distributed. A model that tracks restoration well in dense areas may be less reliable for small or rural
utilities, and forecasts should not be used to rank which communities receive help.
