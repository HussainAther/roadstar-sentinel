# RoadStar Sentinel

**Pressure -> Chaos -> Control**

RoadStar Sentinel is a **CPU-only, MacBook-first** hackathon prototype for entropy-aware AI decision support in trucking and logistics. It treats freight operations as a dynamic system: operational **pressure** rises, future trajectories become less predictable (**chaos / predictive entropy**), available intervention capacity (**control authority**) changes, and Sentinel searches for actions that move the network toward a more stable regime.

## v0.3: CPU batch experiments + ablations

This version keeps the entire project functional without a GPU and adds an experimental harness around the v0.2 dynamic early-warning demo.

- Runs 120 seeded synthetic disturbance scenarios on CPU in a few seconds on a typical laptop.
- Includes stable, mechanical, HOS, congestion, and compound disturbance families.
- Compares four detectors: the original PCC+entropy warning rule, a scalar PCC composite operating point, pressure-only, and a conventional hard alarm.
- Reports precision, recall, false-positive rate, and warning lead time.
- Adds `/experiment` and shows the benchmark directly in the dashboard.
- Keeps the original time-dependent `P(t)`, entropy `H(t)`, control `C(t)`, instability, `dH/dt`, and counterfactual control search.
- Uses a short rolling entropy slope in the batch harness to reduce Monte Carlo jitter without requiring GPU-scale rollout counts.

**Important:** all benchmark values are synthetic prototype results. They are not validated trucking-industry safety, reliability, or performance claims. The purpose is to make the hypothesis falsifiable and expose tradeoffs such as sensitivity versus false alarms.

## Run locally on a MacBook

```bash
cd roadstar-sentinel
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
python -m roadstar_sentinel.cli
```

Dashboard:

```bash
uvicorn roadstar_sentinel.api:app --reload
```

Open `http://127.0.0.1:8000/dashboard`.

Useful API endpoints:

- `GET /trajectory` — the creeping-failure early-warning demonstration
- `GET /experiment` — the 120-scenario CPU benchmark and detector ablations
- `GET /scenario/baseline` and `GET /scenario/disrupted` — single-state PCC/control examples

## Architecture

- `models.py` — fleet, truck, load, and control data structures
- `simulator.py` — lightweight stochastic Monte Carlo operational rollouts
- `pcc.py` — pressure, normalized predictive entropy, control authority, instability, cascade risk
- `dynamics.py` — time evolution, entropy/pressure derivatives, early-warning logic, lead-time calculation
- `experiments.py` — CPU disturbance generator, batch benchmark, ablations, lead-time statistics
- `controls.py` — candidate interventions and counterfactual ranking
- `scenarios.py` — baseline, hard breakdown, and creeping-failure demo scenarios
- `api.py` — FastAPI endpoints and zero-dependency browser dashboard
- `cli.py` — terminal demo including batch benchmark

## Core prototype hypothesis

> Cascading logistics failures may be anticipated by monitoring operational pressure, predictive entropy, their rates of change, and remaining control authority before conventional hard constraints are tripped.

The v0.3 experiment deliberately exposes where a detector succeeds **and** where it false-alarms or misses failures. A later version can add train/test threshold calibration, richer fleet state progression, real public freight/traffic datasets, and model-predictive control over sequences of interventions—all while retaining the CPU-first path.

## v0.4: closed-loop decision support

The v0.4 prototype adds the missing feedback-control step. Sentinel now branches the synthetic fleet simulation at the first PCC/entropy warning, evaluates feasible reassignment actions over the remaining horizon, selects the lowest projected trajectory cost, and compares the resulting controlled future against the same no-control disturbance.

The controller is intentionally CPU-only. Candidate actions are evaluated with paired Monte Carlo seeds to reduce noise between counterfactual comparisons. The primary demonstration metrics are post-warning instability burden (area under the instability curve), cascade-risk burden, peak instability, and whether the conventional failure is avoided or delayed.

New API endpoint:

```text
GET /closed-loop
```

This is a synthetic systems/control demonstration, not a validated trucking dispatch policy or safety system.

## v0.5: dispatcher-facing decision support

v0.5 turns the control engine into a dispatcher-facing copilot while keeping the entire demo CPU-only and fully functional offline except for serving the local browser UI.

The copilot is deliberately **grounded in Sentinel's computed state** rather than asking a language model to invent dispatch decisions. It explains the warning in plain language, exposes a causal chain (pressure, predictive entropy, control authority, cascade exposure), states the no-action counterfactual, reports expected intervention effects and tradeoffs, assigns a transparent confidence level from the control-ranking margin, and compares feasible interventions.

A lightweight local question interface supports prompts such as:

- "Why are you recommending this now?"
- "What happens if I do nothing?"
- "How confident are you?"
- "Compare the alternatives."
- "What are the tradeoffs?"

New endpoints:

```text
GET  /dispatcher-brief
GET  /dispatcher-compare?first=0&second=1
POST /dispatcher-query
```

Example POST body:

```json
{"question":"Why are you recommending this now?"}
```

This v0.5 copilot uses transparent deterministic intent routing over the same PCC/counterfactual evidence, so the hackathon demo has no cloud-LLM or GPU dependency. A later optional LLM adapter can improve conversational breadth without becoming part of the control or safety logic.

## v0.6: interactive disturbance injection

v0.6 turns the dashboard into a judge-friendly **Scenario Lab**. A dispatcher can inject one of five synthetic disturbance families and adjust severity from the browser:

- truck breakdown
- traffic surge
- hours-of-service shortage
- depot outage (represented by a regional-capacity proxy in the compact model)
- compound shock

For each run, Sentinel evolves the disturbance over time, computes PCC/entropy dynamics, detects the first warning, evaluates feasible CPU-only control actions, and plots the counterfactual **no control vs. Sentinel control** instability trajectories. The dashboard reports warning time, hard-failure time when reached, lead time, the selected intervention, and ranked alternatives.

New endpoints:

```text
GET /incidents
GET /interactive-scenario?incident=compound_shock&severity=0.8
```

The scenario lab remains a synthetic demonstration rather than a calibrated trucking simulator. Its purpose is to make the control-system idea tangible: judges can perturb the system themselves and observe how pressure, uncertainty, and control options interact.

## v0.7: human-in-the-loop dispatcher cockpit

v0.7 makes the control recommendation explicitly **advisory rather than autonomous**. In the Scenario Lab, Sentinel still ranks counterfactual interventions, but the dispatcher can click any feasible alternative and simulate that override before acting. The dashboard then redraws the no-control versus chosen-control trajectory and reports the chosen action's objective improvement, failure timing, and **regret versus Sentinel's best-ranked action**.

This gives the hackathon demo a concrete human-AI workflow: **detect -> recommend -> inspect alternatives -> operator chooses -> counterfactual replay -> compare**. The underlying scoring and simulation remain CPU-only and deterministic under fixed seeds; no GPU or cloud LLM is required.

New endpoint:

```text
GET /operator-counterfactual?incident=compound_shock&severity=0.8&action_id=...
```

As in prior versions, all outcomes are synthetic prototype results rather than validated trucking dispatch recommendations.

## v0.8: trucking operations-center visualization

v0.8 adds a judge-facing operations-center surface without changing the CPU-only architecture. The dashboard now includes a synthetic fleet network map, truck/load status, affected freight lanes, a replay-time slider, and an incident timeline that places the PCC early warning and recommended intervention relative to the later conventional failure.

The map is intentionally schematic: its coordinates are generated from the compact synthetic fleet and are **not** GPS positions or road-network claims. It exists to make disturbance propagation and control timing visually legible during a short demo.

New endpoint:

```text
GET /operations-view?incident=compound_shock&severity=0.8&time_hours=1.0
```

This endpoint returns hubs, truck states, load lanes, incident propagation edges, the event timeline, warning/failure timing, and Sentinel's selected control action.
