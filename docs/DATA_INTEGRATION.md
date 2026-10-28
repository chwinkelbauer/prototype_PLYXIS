# Data integration: keeping the twin equal to reality

The twin is only useful if it mirrors the plant. PLYXIS treats this as a closed loop that runs
every data step (10 minutes in the showcase):

```
 connectors ──> TagMap ──> Estimator ──────────────> LiveTwin (parameters, states, data window)
 (SCADA, market,   │        ├ direct assimilation           │
  HTTP, historian) │        ├ well tests                    ├─> calibration.plx  (generated `set` lines)
                   │        └ moving-horizon estimation     │
                   ├──────> AnomalyDetector ─> alerts, urgent re-plan
                   └──────> Discovery ─> PLX code proposal ─> live.plx (generated) ─> recompile, hot swap
```

Everything the loop needs is declared in the PLX model itself, so the same code calibrates an
oil field, a mine or any other twin:

```plx
telemetry:
  liquid   <- "FLOW.LIQ" [bbl/d];                       ## measured signal -> SCADA tag suffix
  vib      <- "ESP.VIB" [mm/s] { warn = 5.5; alarm = 8.0; };
  up       <- "ESP.RUN";                                ## status (run / trip)
  gor      <- "TEST.GOR" [Mscf/bbl] { test = true; };   ## well test (test separator)
properties:
  q_max = 900 [bbl/d] calibrate in [30, 5000];         ## estimated online
discover:
  rule new_well: on "KST.P{pad}.W{well}.FLOW.LIQ" {     ## unknown tag -> code proposal
    init P{pad}_W{well} as esp_well { tag = "KST.P{pad}.W{well}"; };
    connect(P{pad}_W{well}.prod, PAD{pad}.wells);
  };
```

## Connectors (`plyxis/data/connectors.py`)

A connector turns a source into `Telemetry(tag, value, unit, t, quality)` and `Birth` messages
(Sparkplug-style device birth certificates with type and coordinates). The runtime depends only
on this interface.

| Connector | Use |
|---|---|
| `EmulatorConnector` | synthetic SCADA gateway in front of the plant emulator (demo, tests) |
| `MarketConnector` | crude, gas hub and power prices |
| `HttpPushConnector` | external systems POST to `/api/ingest` |
| `CsvReplayConnector` | replay of a historian export (`t, tag, value[, unit]`) |

For a pilot, an OPC UA client, an MQTT Sparkplug B subscriber or a PI Web API poller is a new
class with the same `poll()` method (see [PILOT_PLAN_OIL_GAS.md](PILOT_PLAN_OIL_GAS.md)).

## Tag mapping (`TagMap`)

Derived from the `telemetry:` sections: tag = instance tag prefix + signal suffix
(`KST.P05.W01` + `FLOW.LIQ`). Units are converted to canonical units on the way in. Status
signals (`up`, `avail`) are assimilated directly; `test = true` signals are well tests;
everything else is a measurement for estimation and anomaly detection.

## Estimation (`plyxis/data/estimation.py`)

**Direct assimilation.** Measured states (tank levels, run status, available compressor units, ore-pass
and sump levels, broken stock from cavity surveys) overwrite the twin state each step, clamped to the
state's declared physical limits: a noisy level reading of an empty bin is not a negative stock (negative
stocks would turn into negative flows and blow up flow-weighted attributes such as ore grade).

**Well tests.** A direct measurement of a property (GOR, water cut on the test separator, one
well per hour in the showcase) re-anchors that property for that well. The same mechanism serves other
industries: in the mine, muck-pile grab-sample assays (`ASSAY.CU`, rotating through the stopes) re-anchor
stope grades, while the mill's on-stream analyser (`OSA.HEAD`) constrains the blend.

**Moving-horizon estimation (MHE)** every hour over the last 3 hours of data:

```
min_theta  sum_k huber((y_k(theta) - m_k) / sigma_k)          measurement fit
         + sum |theta / sigma_anchor|^2 / 2                     engineering / well-test anchor
         + sum |(theta - theta_prev) / sigma_step|^2 / 2        smoothness between calibrations
```

* `y(theta)` is a replay of the twin's **physical** program (saturating tanks, `sat()` limits)
  over the window, with the controls the plant actually applied (DCS read-back) and the observed
  status trajectories. Replay step *j* starts from the state measured at the end of step *j-1*,
  so stateful and memoryless signals are aligned with the step that produced them.
* `theta` = log-multipliers of every `calibrate`d property (well deliverability, GOR,
  injectivity of disposal wells, ...) plus unmeasured states with an observation path (ESP pump
  health, seen only through vibration).
* `sigma_k` = 5 % relative + 2 % of the signal's typical level; the Huber loss bounds the
  influence of single bad readings.
* Properties with a recent well test are held tight (4 %); untested ones may move within the
  engineering uncertainty (15 %) when aggregate measurements (station suction, battery gas,
  injection pressure) demand it.
* **Step changes are evidence, not outliers.** While the anomaly detector reports a persistent
  mismatch on an asset (e.g. disposal injection pressure far above the twin), that asset's anchor
  and smoothness priors are released and its residuals are no longer down-weighted. Once the step
  is confirmed, the new value becomes the asset's reference. In the showcase, the unannounced
  55 % injectivity loss of SWD-2 is estimated to within 1 % three hours after it happens.
* **Known discontinuities come from work orders.** When a work order closes (pump replaced,
  compressor unit restarted), the twin is told directly - as it would be from SAP PM / Maximo -
  instead of waiting for telemetry: the asset is back in service, a new pump has full health, and
  the asset's pre-event data are dropped from the estimation window.
* Solved by L-BFGS with strong-Wolfe line search on the differentiable twin. An estimate that
  does not improve the objective is rejected; the previous estimate is kept.
* **Off the plant clock.** Calibration is split into *compute* and *apply*. The engine snapshots
  the window and the parameters and solves the MHE on a background worker while the plant keeps
  running; the result is applied at the next step. A well test that arrived in the meantime wins
  over the estimate, and hidden-state estimates (pump health) are shifted by what the twin's own
  prediction moved since the snapshot (`current + (estimate_end - snapshot)`), with a guard that
  drops a stale estimate if the state jumped (e.g. a pump was replaced). The console therefore
  never pauses for estimation; headless benchmark runs keep it inline for determinism.
* **Prediction between calibrations.** Between solves the twin propagates its unmeasured
  states with its own model (pump health declines with speed⁴), so trends and maintenance
  decisions do not wait for the next calibration.

Estimates that move by more than 1 % are written to `calibration.plx` as reviewable code
(`set kestrel_live.P05_W01.q_max = 982.4 [bbl/d];`) and committed to the twin repository.

**Accuracy in the showcase** (96 wells, engineering estimates deliberately off by realistic
margins, deliverability drifting ±4 %, 1.7 % sensor noise):

| | engineering model | live twin after 30 h |
|---|---|---|
| well deliverability `q_max` (mean abs. error) | 11.0 % | 2.4 % |
| GOR (mean abs. error) | 12.7 % | 10.1 % (falls further as well tests rotate; see the 72 h run in [BENCHMARKS.md](BENCHMARKS.md)) |
| compressor station suction gas (twin vs. plant) | up to 37 % | within ±3 % |
| SWD-2 injectivity after an unannounced 55 % loss | - | within 1 % after 3 h |

An earlier version of the estimator (fixed-step Adam, re-anchored every hour, replay misaligned
by one step) passed the same averages but diverged whenever compression was at its limit: station
gas was then under-predicted by up to 60 %, which made the planner flare unknowingly. The
accuracy harness that exposed this compares the twin with the plant's hidden truth every hour
and is part of the evaluation tooling.

A structural check backs this up: with the plant's true parameters, the twin's replay reproduces
every one of ~1,000 plant signals exactly, so all remaining error is estimation error, not model
error.

## Anomaly detection (`plyxis/data/anomaly.py`)

* **Model mismatch**: EWMA of normalised innovations (measurement minus one-step twin prediction).
  A persistent mismatch triggers recalibration and is shown to the operator.
* **Thresholds** from the model (`warn`, `alarm`).
* **Trends**: least-squares slope over the recent window gives a predicted time to alarm
  ("ESP failure expected in ~9 h"), used by the maintenance planner.
* **Status**: trips, failures and restarts trigger an immediate re-plan.

## Discovery: the twin rewrites itself (`plyxis/data/discovery.py`)

When telemetry arrives from a tag that no twin asset owns, the `discover:` rules of the model are
matched. A match produces a **PLX code proposal** (rule body instantiated with the bindings and
the device's birth-certificate coordinates). The proposal is compiled before it is accepted; on
success it is committed to `live.plx`, the twin is recompiled and hot-swapped (live parameters
and states are transferred by asset path), and an urgent re-plan follows. In the showcase,
proposals are applied automatically and shown in the Data view; in a pilot, an engineer approves
them (management of change).

## Twin repository (`plyxis/data/repo.py`, `runtime/twin_manager.py`)

```
field.plx          engineered model (edited in Twin Studio)
live.plx           generated: discovered assets   (define kestrel_live as kestrel { ... })
calibration.plx    generated: estimator results   (set kestrel_live.<asset>.<property> = ...)
```

Every change - engineer edit, discovery, calibration - is a commit with author, time and diff.
Twin Studio shows the history; any version can be inspected.

## Forecasts (`plyxis/data/forecast.py`)

Deliberately transparent: persistence for crude and gas, a learned daily profile for power,
plus announced events (e.g. a pipeline maintenance notice that pushes the hub price negative for
24 hours) layered on top.
