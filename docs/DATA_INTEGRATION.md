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

### Plant systems (`plyxis/data/industrial.py`)

All read only, all producing the same messages as the emulated gateway, with the source's quality flag. A site lists them
in its `scenario.json` under `connectors`; nothing downstream changes.

| Connector | Protocol | Tested against | Not yet tested against |
|---|---|---|---|
| `PiWebApiConnector` | PI Web API `streamsets/value` over HTTPS, basic authentication, digital states, `Good` / `Questionable` | a local HTTP server returning the documented JSON | a real PI server; Kerberos authentication is not implemented |
| `SparkplugConnector` | MQTT 3.1.1 (QoS 0) with a Sparkplug B decoder written from the specification: births, aliases, deaths, nulls, sequence numbers; never publishes | a local socket that speaks enough MQTT, payloads encoded by the test | a real broker or edge gateway; TLS is not implemented |
| `OpcUaConnector` | OPC UA through the optional `asyncua` package (`pip install -r requirements-connectors.txt`), polling `read_data_value` | `asyncua`'s own server on the same machine, no security policy (the test is skipped without the package) | a real server; certificates; subscriptions |

## Data quality (`plyxis/data/quality.py`)

A bad reading must not become twin state. In front of the estimator, per tag:

| Flag | Rule | Then |
|---|---|---|
| stale | a tag that reported before has been silent for more than 1 h | the twin predicts the signal |
| frozen | an analog value repeats exactly 12 times (2 h at the 10-minute data step); real instruments have noise | reading held back, the twin predicts the signal |
| out of range | below a state's physical lower limit by more than the tolerance | reading held back, the twin predicts the signal |
| bad at the source | the source says so (PI `Good`, OPC UA status, Sparkplug null) | reading held back, the twin predicts the signal |

Flagged readings stay visible in the tag list, raise an alert and enter the audit log, and do not reach estimation or
anomaly detection. For a measured state (a tank level) the estimator's prediction step takes over until the signal recovers.
On clean data there are no flags (tested), so the A/B runs are unaffected. `POST /api/event` with `kind: "sensor"`
(`freeze`, `dropout`, `value`) injects an instrument fault in a live demo.

## History: backfill and backtest (`plyxis/data/historian.py`, `runtime/backfill.py`, `bench/backtest.py`)

A historian export is one compressed CSV of measurements, direct samples, set points, line-ups and prices.
`python -m plyxis backfill history.csv.gz --scenario kestrel` replays it through the live estimator and writes a
`calibration.plx`. `bench/backtest.py` is pilot gate G1: the plant writes the file, a fresh twin sees only the file, is
calibrated on the first half and frozen, and predicts the second half (results in [BENCHMARKS.md](BENCHMARKS.md)).

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

**Offset-free correction.** Where a measured output summarises many uncertain parameters - the mill's head
grade is the blend of dozens of stope grades - the twin also carries a calibrated ratio between the measured
and the computed output (`concentrator.head_factor`, declared `calibrate in [0.8, 1.25]`). The estimator
fits it like any other property; the planner then plans against the corrected output. This is the
disturbance model of offset-free MPC: a plan that sits on a limit as the twin sees it (the head-grade
window) stays inside it in reality even while the individual stope grades are still off.

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

## Direct samples and uncertainty (`Estimator.ingest`, `Estimator.uncertainty`)

A direct sample of a property updates it at once. Well tests are accurate (1 % error) and replace most of the old value. Mine
assays are composite muck samples at 5 % error (`scenario.json` -> `assays.sigma`): they use a Kalman update, so repeated
samples average out and an old estimate is not thrown away for one noisy assay.

Grade control samples where the mine works (`assays.policy: "working"`, three samples per hour, the working places sampled
longest ago): with about 35 working places, every place about once a shift. Both worlds sample with the same rule; the
baseline's grade control reconciles its block model with its own assays through the same update.

`Estimator.uncertainty` gives the relative standard error per instance: the posterior of its samples if sampled, otherwise how
far the samples so far put the engineering values off (their spread in log; the prior until eight samples exist). The Data view
reports it, and it sizes the planning margin of the head grade ([OPTIMIZER.md](OPTIMIZER.md)).

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
