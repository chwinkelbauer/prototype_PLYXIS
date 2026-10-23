# PLYXIS - building and implementation plan for the first oil & gas pilots

*Version 1.0 - prepared with prototype 2 (this repository) as the technical baseline.*

## 1. Summary

The first pilots put PLYXIS on a real onshore production field and prove three things, in this
order:

1. **The twin mirrors the field.** Built from the customer's own data in weeks, not months, and
   kept accurate live (well rates, station gas, tank levels, disposal pressures).
2. **The plans are better than today's decisions** - first in shadow mode (no action), then as
   recommendations that operators execute.
3. **Value is measurable and attributable**: production, flaring, power, trucking and
   failure-related costs, under an agreed measurement-and-verification method.

Closed-loop operation (PLYXIS writing set points) is the last step, limited to low-risk set
points, behind the customer's own control system and management of change (MOC).

| | |
|---|---|
| Target asset | onshore field or gathering system: 100-1,000 artificially lifted wells (ESP, rod pump, gas lift), tank batteries, compression, flares, water disposal, trucking, field crews and rigs |
| Duration | ~10 months from contract to closed-loop decision; commercial conversion decision after ~7 months |
| PLYXIS team | 4-5 FTE per pilot (first pilot), shared core team afterwards |
| Customer effort | sponsor + production engineer champion (~30 %) + OT/IT engineer (~20 %) + field foreman input |
| Pilot fee | paid pilot, USD 150-250k fixed, credited to the first-year licence on conversion |
| Conversion model | site licence (~USD 300k ARR for a medium field) or gain-share 15-20 % of verified value |

## 2. What the prototype already proves - and what a pilot needs in addition

| Capability | Prototype 2 status | Needed for pilot |
|---|---|---|
| Model language (PLX) + compiler | complete: units, constraints, GIS, tables, packages, inheritance, discovery rules; compiled differentiable simulator; reference interpreter; test-suite | hardening: language versioning, model linting, larger standard library (gas lift, plunger lift, separators, LACT, pipelines with hydraulics) |
| Twin build from data | from CSV tables + GeoJSON + PLX | importers for the customer's well master data, GIS feature services (Esri), facility lists; templates per asset class |
| Live data | connector interface, emulated SCADA, HTTP push, CSV historian replay | **OPC UA, MQTT Sparkplug B, PI Web API / AVEVA connectors**; data-quality flags, gap handling, time-zone/clock sync |
| Estimation | moving-horizon estimation, well tests, hidden states, anomaly detection, discovery | historian backfill calibration (6-12 months), well-test import from production accounting, uncertainty reporting |
| Optimisation | cascade solves 40k decisions in ~55 s on a laptop CPU; neural sub-solvers add 2-4 % on re-plans | solve-time SLA monitoring, job scheduling, plan approval workflow, constraint editor for engineers |
| Plans to action | set points, line-ups, truck trips, work orders, explanations | **recommendation workflow** (accept / modify / reject with reason), shift handover report, export to SAP PM / Maximo, mobile view for lease operators |
| Write-back | emulator only | Phase 4 only: bounded set-point write-back through the SCADA's own interface, with operator acknowledgement |
| Platform | single process, local | containerised deployment (customer cloud tenant or on-prem), SSO/RBAC, audit log, backups, monitoring |
| Security | n/a | IEC 62443-aligned zone/conduit design, penetration test, SOC 2 Type I readiness |

Because PLYXIS uses no external AI service, the whole stack can run inside the customer's tenant
or fully on-prem, air-gapped if required - a strong argument with OT security teams.

## 3. Building plan (product readiness, months -4 to 0, parallel to sales)

| Workstream | Deliverables | Effort |
|---|---|---|
| Connectors | OPC UA client (read), Sparkplug B subscriber, PI Web API / AVEVA Data Hub reader, CSV/Parquet bulk import; quality & staleness handling | 1 data engineer, 3 months |
| Time-series & backfill | storage for raw tags (TimescaleDB or the customer historian), backfill runner that replays 6-12 months through the estimator | 1 data engineer, 1.5 months |
| Library & templates | `oilgas` extensions: gas lift, plunger lift, separators, LACT units, VRUs, pipeline hydraulics (pressure-driven flow), chemical injection; field template wizard | 1 twin engineer, 3 months |
| Application | recommendation workflow, approvals, shift report, roles, comments; operator mobile view | 1 full-stack engineer, 4 months |
| Optimizer productisation | solve service with queues and SLAs, plan versioning, constraint editor, scenario ("what-if") runs | 1 optimisation engineer, 3 months |
| Platform & security | containers, Helm chart, SSO (Entra ID / Okta), RBAC, audit log, observability, CI/CD, penetration test | 1 platform engineer, 3 months |
| Validation harness | backtesting tool: replay history, compare twin vs. measurements, compute counterfactual value | shared, 1 month |

Exit criterion for "pilot-ready v1": a 300-well synthetic field with 12 months of synthetic
history goes from raw tag export to calibrated live twin and first approved plan in under
5 working days, run by someone who did not build the tooling.

## 4. Pilot phases and gates

```
week   0    4         12        18                30          40
       |-P0-|----P1----|---P2----|-------P3-------|-----P4-----|
        qualify  build &   shadow     advisory        supervised
        & sign   backtest  mode       (recommend)     closed loop
             G0        G1        G2               G3           G4
```

### Phase 0 - Qualification and contracting (weeks 0-4)

* Site selection with the customer: one field or gathering system with a real bottleneck
  (compression, water disposal, takeaway, trucking) and reasonable instrumentation (well-level
  rates or run status, facility meters, tank levels).
* Data inventory: tag list and historian access, well master data, well tests, production
  accounting, GIS layers, maintenance history, price assumptions.
* Agree success criteria, the M&V method (section 7) and the baseline period.
* Security questionnaire, architecture review with OT security; NDA, pilot agreement, data
  processing terms.

**Gate G0** - signed pilot agreement; read-only data path approved; named customer champion.

### Phase 1 - Data connection, twin build and backtest (weeks 4-12)

* Read-only connection (historian replication or OPC UA through the DMZ, see section 5).
* Tag-mapping workshop: map the customer's tag conventions to PLX `telemetry:` bindings (a
  2-day workshop with the SCADA engineer; the mapping is code and is reviewed like code).
* Build the field model in PLX from master data and GIS: wells, pads, flowlines, batteries,
  compression, flares, disposal, sales points, trucking lanes, crews and rigs. Engineering
  constraints (permits, capacities, pressure limits) are entered with the production engineer.
* Calibrate on 6-12 months of history (backfill), then **backtest**: replay unseen months,
  compare twin predictions with measurements.
* Value hypothesis: run the optimizer on historical situations and estimate the counterfactual.

**Gate G1 - twin fidelity** (on the backtest months):

| Signal | Criterion |
|---|---|
| Well liquid rates (instrumented wells) | MAPE ≤ 8 % |
| Facility gas (compression suction, sales) | MAPE ≤ 5 % |
| Tank levels, 3 h ahead | MAE ≤ 10 % of tank height |
| Disposal injection pressure | MAPE ≤ 10 % |
| Discovered/unmodelled assets | all assets with tags represented, or explicitly excluded |

### Phase 2 - Shadow mode (weeks 12-18)

* PLYXIS runs live, re-plans on its rolling horizon and after every event, and publishes plans
  and explanations - **no actions are taken**.
* Daily comparison: what operators did vs. what PLYXIS recommended, valued on the calibrated twin.
* Weekly review with the champion; model fixes are commits in the twin repository.

**Gate G2** - estimated value ≥ 1.5 × pilot fee annualised; ≥ 80 % of reviewed recommendations
judged "technically executable" by the champion; no unresolved safety concerns.

### Phase 3 - Advisory mode (weeks 18-30)

* Shift plans go to production engineers and foremen as recommendations: ESP speed set points,
  well on/off and restart sequences, battery line-ups and pump-outs, truck schedules, compressor
  unit commitment, crew and rig work orders (exported to SAP PM / Maximo).
* Operators accept, modify or reject each recommendation with a reason; the reason is fed back
  into the model (missing constraints become PLX constraints).
* M&V runs continuously (section 7).

**Gate G3 - commercial**: verified value against the agreed targets (section 6); adoption
≥ 60 % of recommendations; customer decision on licence or gain-share contract.

### Phase 4 - Supervised closed loop (weeks 30-40, limited scope)

* Scope: low-risk set points only, e.g. ESP VSD speed within an engineer-approved band, tank
  pump-out rates, compressor unit start/stop *proposals*.
* Write-back through the SCADA/DCS's own set-point interface, never directly to controllers;
  bounds, rate-of-change limits and operator acknowledgement enforced on the control-system
  side. Interlocks, ESD and SIS remain authoritative and untouched.
* MOC and a HAZOP/LOPA review of the automation function before go-live; staged roll-out
  (one pad, then a battery area, then the field).

**Gate G4** - 4 weeks without a PLYXIS-caused alarm or trip; operators prefer the closed loop for
the scoped set points; MOC closed.

## 5. OT/IT integration architecture

```
 Level 4/5  Enterprise / cloud tenant   PLYXIS (twin, estimation, optimizer, console)
                                          │  ▲ SAP PM / Maximo (work orders), ProCount/Enertia (allocation),
                                          │  │ Esri ArcGIS (feature services), SSO (Entra ID / Okta)
 Level 3.5  IT/OT DMZ                   historian replica (PI-to-PI / AVEVA Data Hub) · MQTT broker
                                          ▲  read-only (P1-P3)          │ set points (P4 only, bounded)
 Level 3    Site operations             SCADA servers / historian ───────┘ via SCADA set-point interface
 Level 0-2  Control                     RTUs, PLCs, VSDs, SIS (unchanged, authoritative)
```

* **Read path (phases 1-3):** historian replication into the DMZ, or OPC UA from the SCADA
  server through the DMZ, or MQTT Sparkplug B where edge gateways exist. No inbound connection
  into Level 3 is required.
* **Write path (phase 4 only):** one conduit from PLYXIS to the SCADA's set-point interface,
  limited to an allow-list of tags with bounds; every write is logged and acknowledged.
* **Standards:** IEC 62443-3-2 zones and conduits (target SL-2 for the PLYXIS conduit),
  IEC 62443-4-1 secure development practices for the platform; customer cloud policies; data
  stays in the customer's tenant or on-prem.
* **Identity & audit:** SSO, role-based access (viewer, operator, engineer, admin), complete audit
  trail of plans, recommendations, decisions and twin commits.

## 6. Success criteria and KPI targets

Targets are set per field in phase 0 from the bottleneck analysis; typical ranges:

| KPI | Typical pilot target | Measured how |
|---|---|---|
| Oil production at equal constraints | +1 to +3 % | M&V (section 7) |
| Flaring during facility upsets | -30 to -60 % | flare meters, event-based comparison |
| Unplanned ESP failures / deferred production | -10 to -20 % | failure records, deferment accounting |
| Power cost (load shifting, VSD speed) | -5 to -10 % | metered power, tariff |
| Water handling & trucking cost | -5 to -15 % | trucking tickets, SWD volumes |
| Planning effort (engineer hours per day) | -30 to -50 % | time study before/after |
| Twin fidelity | Gate G1 criteria held live | continuous |
| Safety | zero PLYXIS-related incidents | HSE reporting |

For scale: on a 20,000 bbl/d field, +1 % oil at USD 70/bbl is ~USD 5 M per year before
royalties - an order of magnitude above the licence fee.

## 7. Measurement and verification (M&V)

Agreed with the customer's production and finance teams before phase 2:

1. **Baseline**: 3-6 months of historical operation, normalised for decline, downtime and prices
   (decline curves per well from the twin's calibration).
2. **Control group**: comparable pads or batteries that stay under manual planning during
   phase 3, or **switch-back** (alternating days/shifts) where the field cannot be split.
3. **Counterfactual on the twin**: the calibrated twin replays what the operators would have
   done; the difference to the executed PLYXIS plan is the modelled value. Used as supporting
   evidence, never as the only evidence.
4. **Attribution rules** for events outside PLYXIS's control (weather, third-party outages).

The showcase in this repository uses exactly this logic in simulation: two worlds with the same
truth, events and noise, one run by PLYXIS and one by a rule-based model of manual operations.

## 8. Safety and governance principles

* PLYXIS plans within the operating envelope the customer defines; it never overrides ESD,
  SIS, interlocks or operator commands.
* Human-in-the-loop until Gate G4; afterwards only for the allow-listed set points.
* Every recommendation carries its reason (the objective terms and bottleneck prices that drive
  it) and can be rejected.
* Every model change is a reviewed commit; structural changes from discovery require approval.
* MOC for each step that changes how the field is operated (advisory, closed loop, scope
  extensions).
* Fallback: if data or the planner is unavailable, the field runs as today; PLYXIS degrades to
  "no recommendation", never to a stale one (plans carry a validity time).

## 9. Team

| Role | PLYXIS | Customer |
|---|---|---|
| Pilot lead / solutions engineer | 1.0 FTE (customer-facing, M&V, gates) | executive sponsor (VP operations / asset manager) |
| Twin engineer (production engineering background) | 1.0 | production engineer champion (~30 %) |
| Data / OT integration engineer | 1.0 (P0-P2), 0.5 after | SCADA / OT engineer (~20 %) |
| Optimisation engineer | 0.5 | - |
| Full-stack / UX | 0.5 | foremen, lease operators (feedback sessions) |
| Security & platform | 0.5 | IT / OT security (reviews) |

## 10. Budget (first pilot, PLYXIS cost)

| Item | Estimate (EUR) |
|---|---|
| Team, 4.5 FTE × 10 months (fully loaded, ~EUR 9.5k/month) | 430k |
| Cloud / hardware (customer tenant usually paid by customer) | 15k |
| Travel and site visits | 35k |
| Penetration test, security review | 25k |
| Contingency (10 %) | 50k |
| **Total** | **~EUR 555k** |

Pilot fee income: USD 150-250k. Subsequent pilots reuse the platform and templates; expected
cost falls to ~EUR 250-300k each, and a second pilot with the same customer to well below that.

## 11. Commercial model

* **Paid pilot** with fixed fee, credited on conversion.
* **Site licence**: ~USD 300k ARR for a medium field (matches the business plan), including
  model maintenance and solver service.
* **Gain-share** alternative: 15-20 % of verified value under the M&V method, with a floor fee.
* **Enterprise**: multi-field licence with company-level optimisation (shared plants, takeaway,
  permits, crews), as demonstrated by the generated enterprise twin (8 fields, 40k decisions).

## 12. Candidate customer profiles

| Profile | Why PLYXIS fits |
|---|---|
| Mid-size US unconventional operators (Permian/Delaware, Eagle Ford, Bakken), 200-2,000 wells | gas-constrained gathering, disposal limits, trucking, high well counts with ESP/gas lift; fast decision cycles; often PE-backed and value-driven |
| Mature onshore portfolios in Central/Eastern Europe (Vienna Basin, Romania), thousands of rod-pumped and ESP wells | large, ageing well stocks, power-cost pressure, emissions targets; proximity to the PLYXIS team |
| Gathering & processing midstream companies | compression, takeaway and nomination optimisation across producers |
| National / integrated companies' onshore business units (MENA, Latin America) | scale, standardisation needs, on-prem/air-gapped requirement favours an AI-service-free stack |

Ideal first pilot: an operator with a known, recurring bottleneck (compression or disposal),
historian access, a motivated production engineer and a sponsor who owns the P&L of the field.

## 13. Risks and mitigations

| Risk | Mitigation |
|---|---|
| OT/security approval delays the data connection | read-only historian replication first; security package (architecture, IEC 62443 mapping, pen test) ready before phase 0; on-prem option |
| Poor or missing instrumentation | twin uses facility-level measurements and well tests; MHE estimates unmeasured quantities; instrument gaps become a recommendation |
| Model does not match reality | backtest gate G1; discovery and calibration are continuous; model fixes are commits reviewed with the champion |
| Operators do not trust recommendations | explanations for every recommendation, shadow phase, accept/reject workflow, foremen involved from phase 1 |
| Value is hard to attribute | M&V agreed upfront; control group or switch-back design |
| Solve time grows with field size | measured: 40k decisions in ~55 s on one laptop CPU; decomposition and neural warm starts; horizon/resolution configurable |
| Key-person risk in a small team | everything is code (PLX models, tag maps, configs) under version control; runbooks |
| Safety incident attributed to automation | closed loop only after G4 criteria, MOC and HAZOP; allow-listed set points with bounds enforced in the control system |

## 14. Deliverables checklist

* P0: site selection memo, data inventory, success criteria, M&V plan, security architecture.
* P1: connected data feed, tag map (code), field model (PLX) with review record, backtest report.
* P2: shadow-mode dashboard, weekly value reports, model change log.
* P3: recommendation workflow live, adoption and value reports, gate G3 business case.
* P4: MOC package, HAZOP/LOPA record, closed-loop runbook, final pilot report.
