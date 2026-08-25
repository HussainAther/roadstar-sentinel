from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse

from .controls import rank_controls
from .closed_loop import run_closed_loop
from .experiments import percentile, run_experiment
from .dynamics import degraded_state, simulate_trajectory, warning_lead_time
from .dispatcher import answer_dispatcher_question, build_dispatcher_brief, compare_outcomes
from .pcc import evaluate_pcc
from .scenarios import baseline_scenario, creeping_failure_scenario, disrupted_scenario
from .interactive import analyze_incident, analyze_selected_action, incident_catalog

app = FastAPI(title="RoadStar Sentinel", version="0.7.0")


def serialize_scenario(name: str) -> dict:
    state = disrupted_scenario() if name == "disrupted" else baseline_scenario()
    metrics = evaluate_pcc(state)
    ranked = rank_controls(state)[:5]
    return {
        "scenario": name,
        "metrics": metrics.__dict__,
        "recommendations": [
            {
                "action": r.action.__dict__,
                "projected_metrics": r.metrics.__dict__,
                "instability_improvement": r.improvement,
            }
            for r in ranked
        ],
    }


def serialize_trajectory() -> dict:
    base = creeping_failure_scenario()
    points = simulate_trajectory(base)
    warning_point = next((p for p in points if p.early_warning), None)
    failure_point = next((p for p in points if p.conventional_failure), None)
    recommendations = []
    if warning_point is not None:
        warning_state = degraded_state(base, warning_point.time_hours)
        recommendations = [
            {
                "action": r.action.__dict__,
                "projected_metrics": r.metrics.__dict__,
                "instability_improvement": r.improvement,
            }
            for r in rank_controls(warning_state, rollouts=140, seed=31)[:5]
        ]
    return {
        "scenario": "creeping_t03_failure",
        "lead_time_hours": warning_lead_time(points),
        "warning_time_hours": warning_point.time_hours if warning_point else None,
        "failure_time_hours": failure_point.time_hours if failure_point else None,
        "points": [
            {
                "time_hours": p.time_hours,
                "entropy_rate": p.entropy_rate,
                "pressure_rate": p.pressure_rate,
                "early_warning": p.early_warning,
                "conventional_failure": p.conventional_failure,
                **p.metrics.__dict__,
            }
            for p in points
        ],
        "warning_recommendations": recommendations,
    }


@app.get("/")
def root() -> dict:
    return {
        "name": "RoadStar Sentinel",
        "tagline": "Detect instability before logistics systems fail.",
        "framework": "Pressure -> Chaos -> Control",
        "version": "0.7.0",
        "dashboard": "/dashboard",
    }


@app.get("/scenario/{name}")
def scenario(name: str) -> dict:
    return serialize_scenario(name)


@app.get("/trajectory")
def trajectory() -> dict:
    return serialize_trajectory()


@lru_cache(maxsize=1)
def serialize_experiment() -> dict:
    summary, _ = run_experiment(n=120, seed=2026, rollouts=80)
    return {
        "scenarios": summary.scenarios,
        "failures": summary.failures,
        "non_failures": summary.non_failures,
        "family_counts": summary.family_counts,
        "detectors": [d.__dict__ for d in summary.detectors],
        "sentinel_lead_time": {
            "p25": percentile(summary.sentinel_lead_times_hours, 0.25),
            "median": percentile(summary.sentinel_lead_times_hours, 0.50),
            "p75": percentile(summary.sentinel_lead_times_hours, 0.75),
        },
        "disclaimer": "Synthetic CPU-only benchmark; not validated trucking performance.",
    }


@app.get("/experiment")
def experiment() -> dict:
    return serialize_experiment()


def _serialize_point(p) -> dict:
    return {
        "time_hours": p.time_hours,
        "entropy_rate": p.entropy_rate,
        "pressure_rate": p.pressure_rate,
        "early_warning": p.early_warning,
        "conventional_failure": p.conventional_failure,
        **p.metrics.__dict__,
    }


@lru_cache(maxsize=1)
def serialize_closed_loop() -> dict:
    result = run_closed_loop(creeping_failure_scenario(), rollouts=120, seed=23)
    return {
        "control_time_hours": result.control_time_hours,
        "selected_action": result.selected.action.__dict__,
        "selected_outcome": {
            "objective": result.selected.objective,
            "mean_instability": result.selected.mean_instability,
            "mean_cascade_risk": result.selected.mean_cascade_risk,
            "peak_instability": result.selected.peak_instability,
            "failure_time_hours": result.selected.failure_time_hours,
            "failure_avoided": result.selected.failure_avoided,
            "failure_delay_hours": result.selected.failure_delay_hours,
        },
        "instability_auc": {
            "no_control": result.instability_auc_no_control,
            "sentinel_control": result.instability_auc_control,
            "reduction": result.instability_auc_reduction,
        },
        "cascade_auc": {
            "no_control": result.cascade_auc_no_control,
            "sentinel_control": result.cascade_auc_control,
            "reduction": result.cascade_auc_reduction,
        },
        "no_control_failure_time_hours": next((p.time_hours for p in result.no_control if p.conventional_failure), None),
        "alternatives": [
            {
                "action": a.action.__dict__,
                "objective": a.objective,
                "mean_instability": a.mean_instability,
                "mean_cascade_risk": a.mean_cascade_risk,
                "peak_instability": a.peak_instability,
                "failure_time_hours": a.failure_time_hours,
                "failure_avoided": a.failure_avoided,
                "failure_delay_hours": a.failure_delay_hours,
            }
            for a in result.alternatives
        ],
        "no_control": [_serialize_point(p) for p in result.no_control],
        "sentinel_control": [_serialize_point(p) for p in result.sentinel_control],
        "disclaimer": "Synthetic CPU-only counterfactual control demonstration; not validated trucking performance.",
    }


@app.get("/closed-loop")
def closed_loop() -> dict:
    return serialize_closed_loop()


class DispatcherQuestion(BaseModel):
    question: str


@app.get("/dispatcher-brief")
def dispatcher_brief() -> dict:
    result = run_closed_loop(creeping_failure_scenario(), rollouts=120, seed=23)
    return build_dispatcher_brief(result).as_dict()


@app.get("/dispatcher-compare")
def dispatcher_compare(first: int = 0, second: int = 1) -> dict:
    result = run_closed_loop(creeping_failure_scenario(), rollouts=120, seed=23)
    options = [x for x in result.alternatives if x.action.kind != "none"]
    if not options:
        return {"error": "No feasible interventions."}
    first = max(0, min(first, len(options) - 1))
    second = max(0, min(second, len(options) - 1))
    return compare_outcomes(options[first], options[second])


@app.post("/dispatcher-query")
def dispatcher_query(payload: DispatcherQuestion) -> dict:
    result = run_closed_loop(creeping_failure_scenario(), rollouts=120, seed=23)
    return answer_dispatcher_question(payload.question, result)


@app.get("/incidents")
def incidents() -> dict:
    return {"incidents": incident_catalog()}


@app.get("/interactive-scenario")
def interactive_scenario(incident: str = "truck_breakdown", severity: float = 0.7) -> dict:
    valid = {item["kind"] for item in incident_catalog()}
    if incident not in valid:
        return {"error": f"Unknown incident '{incident}'.", "valid_incidents": sorted(valid)}
    result = analyze_incident(incident, severity=max(0.0, min(1.0, severity)), rollouts=70, seed=41)
    return {
        **{k: v for k, v in result.items() if k not in {"no_control", "sentinel_control"}},
        "no_control": [_serialize_point(p) for p in result["no_control"]],
        "sentinel_control": [_serialize_point(p) for p in result["sentinel_control"]],
        "disclaimer": "Synthetic scenario lab. Depot outage uses a regional-capacity proxy because the compact demo has no explicit depot object.",
    }


@app.get("/operator-counterfactual")
def operator_counterfactual(incident: str, action_id: str, severity: float = 0.7) -> dict:
    valid = {item["kind"] for item in incident_catalog()}
    if incident not in valid:
        return {"error": f"Unknown incident '{incident}'.", "valid_incidents": sorted(valid)}
    result = analyze_selected_action(
        incident, action_id, severity=max(0.0, min(1.0, severity)), rollouts=70, seed=41
    )
    if "error" in result:
        return result
    return {
        **{k: v for k, v in result.items() if k not in {"no_control", "chosen_control", "sentinel_control"}},
        "no_control": [_serialize_point(p) for p in result["no_control"]],
        "chosen_control": [_serialize_point(p) for p in result["chosen_control"]],
        "sentinel_control": [_serialize_point(p) for p in result["sentinel_control"]],
        "disclaimer": "Synthetic human-in-the-loop counterfactual; not a validated dispatch policy.",
    }


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> str:
    return r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>RoadStar Sentinel</title>
<style>
:root{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#e9eef5;background:#09111b}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at top left,#14263a 0,#09111b 38%,#060b12 100%);min-height:100vh}.wrap{max-width:1220px;margin:auto;padding:28px}.header{display:flex;justify-content:space-between;gap:20px;align-items:end;margin-bottom:22px}.kicker{color:#75d8ff;font-size:13px;letter-spacing:.16em;text-transform:uppercase}h1{margin:5px 0 4px;font-size:38px}.sub{color:#9eb0c2}.badge{border:1px solid #31536d;background:#0d2030;padding:8px 11px;border-radius:999px;color:#a9dcf6;font-size:12px}.metric-grid{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin-bottom:18px}.metric{background:#0b1825;border:1px solid #1d3245;border-radius:12px;padding:14px}.metric .name{color:#8fa7bc;font-size:11px;text-transform:uppercase;letter-spacing:.07em}.metric .value{font-size:25px;font-weight:700;margin-top:5px}.grid{display:grid;grid-template-columns:1.18fr .82fr;gap:18px}.panel{background:rgba(10,21,33,.88);border:1px solid #1f3447;border-radius:16px;padding:18px;box-shadow:0 16px 50px rgba(0,0,0,.22)}.alert{background:#241a12;border:1px solid #5d4223;border-radius:12px;padding:13px 15px;margin-bottom:14px;color:#ffd9a6}.canvas-wrap{background:#07121c;border:1px solid #263f53;border-radius:12px;padding:10px}.legend{display:flex;gap:16px;flex-wrap:wrap;color:#8fa7bc;font-size:12px;margin:10px 0 2px}.sw{display:inline-block;width:18px;height:3px;vertical-align:middle;margin-right:5px;border-radius:4px}.rec{padding:13px 0;border-bottom:1px solid #172b3d}.rec:last-child{border-bottom:0}.rank{color:#6fd4ff;font-weight:700;margin-right:6px}.small{color:#93a9bc;font-size:13px;margin-top:5px}.story{display:grid;grid-template-columns:repeat(3,1fr);gap:9px;margin-top:14px}.story>div{background:#0b1825;border:1px solid #1d3245;border-radius:10px;padding:11px}.story b{display:block;font-size:19px;margin-top:4px}.footer{color:#657b8e;font-size:12px;margin-top:18px}.ask{display:flex;gap:8px;margin-top:8px}.ask input{flex:1;background:#07121c;color:#e9eef5;border:1px solid #31536d;border-radius:9px;padding:10px}.ask button{background:#17364d;color:#dff4ff;border:1px solid #3d6f8f;border-radius:9px;padding:10px 14px;cursor:pointer}.factor{padding:10px 0;border-bottom:1px solid #172b3d}.pill{font-size:10px;text-transform:uppercase;letter-spacing:.08em;border:1px solid #31536d;border-radius:999px;padding:3px 7px;margin-left:6px;color:#a9dcf6}.scenario-lab{margin-bottom:18px}.scenario-controls{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:12px 0}.scenario-btn{background:#0b1825;color:#cfe8f8;border:1px solid #31536d;border-radius:9px;padding:9px 11px;cursor:pointer}.scenario-btn.active{background:#17364d;border-color:#69d7ff}.slider{display:flex;gap:10px;align-items:center;min-width:260px}.slider input{flex:1}.scenario-result{display:grid;grid-template-columns:1.2fr .8fr;gap:14px;margin-top:12px}.statusline{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.status-chip{font-size:11px;font-weight:700;letter-spacing:.08em;border:1px solid #5d4223;background:#241a12;color:#ffd9a6;border-radius:999px;padding:5px 8px}.choice{width:100%;text-align:left;background:#0b1825;color:#dbeaf5;border:1px solid #31536d;border-radius:10px;padding:10px 11px;margin-top:7px;cursor:pointer}.choice:hover{border-color:#69d7ff}.choice.sentinel{border-color:#4a7b65}.decision-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:10px}.decision-grid>div{background:#07121c;border:1px solid #263f53;border-radius:9px;padding:9px}.decision-grid b{display:block;font-size:18px;margin-top:3px}@media(max-width:900px){.grid{grid-template-columns:1fr}.metric-grid{grid-template-columns:repeat(2,1fr)}.story{grid-template-columns:1fr}.scenario-result{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="wrap">
 <div class="header"><div><div class="kicker">Pressure -> Chaos -> Control</div><h1>RoadStar Sentinel</h1><div class="sub">Dynamic entropy early warning for resilient freight networks.</div></div><div class="badge">v0.7 · CPU-only human-in-the-loop cockpit</div></div>
 <div class="panel scenario-lab">
  <div class="kicker">Interactive Scenario Lab</div>
  <div class="statusline"><h2 style="margin:7px 0 4px">Break the fleet. Watch Sentinel respond.</h2><span id="scenarioStatus" class="status-chip">READY</span></div>
  <div class="sub">Inject a synthetic disturbance, adjust its severity, and compare no-control versus Sentinel-control trajectories. Everything runs locally on CPU.</div>
  <div class="scenario-controls" id="scenarioButtons"></div>
  <div class="slider"><span class="small">Severity</span><input id="severity" type="range" min="0.2" max="1" step="0.1" value="0.7" oninput="document.getElementById('severityValue').textContent=this.value"/><b id="severityValue">0.7</b><button class="scenario-btn" onclick="runScenario()">Run scenario</button></div>
  <div class="scenario-result">
   <div><div class="canvas-wrap"><canvas id="scenarioChart" width="760" height="300" style="width:100%;height:auto"></canvas></div><div class="legend"><span><i class="sw" style="background:#ef7c7c"></i>No control</span><span><i class="sw" style="background:#93e7b9"></i>Sentinel control</span></div></div>
   <div><div id="scenarioSummary" class="alert">Choose a disturbance and run it.</div><div class="kicker">Human-in-the-loop decision</div><div class="small">Sentinel recommends an action, but the dispatcher can override it and simulate that choice before committing.</div><div id="scenarioRecs"></div><div id="operatorOutcome"></div></div>
  </div>
 </div>
 <div class="metric-grid" id="metrics"></div>
 <div class="grid">
  <div class="panel">
   <div id="alert" class="alert"></div>
   <div class="kicker">PCC dynamics over time</div><h2 style="margin:7px 0 10px">Instability emerges before failure</h2>
   <div class="canvas-wrap"><canvas id="chart" width="760" height="380" style="width:100%;height:auto"></canvas></div>
   <div class="legend"><span><i class="sw" style="background:#69d7ff"></i>Pressure</span><span><i class="sw" style="background:#f5b95f"></i>Entropy</span><span><i class="sw" style="background:#93e7b9"></i>Control</span><span><i class="sw" style="background:#ef7c7c"></i>Instability</span></div>
   <div class="story"><div class="small">First Sentinel warning<b id="warningTime">—</b></div><div class="small">Conventional failure<b id="failureTime">—</b></div><div class="small">Early-warning lead<b id="leadTime">—</b></div></div>
   <div class="footer">Prototype thresholds are for the synthetic demonstration only; they are not validated trucking-industry safety limits.</div><div style="margin-top:18px" class="kicker">Closed-loop counterfactual</div><h2 style="margin:7px 0 10px">No control vs Sentinel control</h2><div class="canvas-wrap"><canvas id="controlChart" width="760" height="300" style="width:100%;height:auto"></canvas></div><div class="story" id="controlStory"></div>
   <div style="margin-top:18px" class="kicker">CPU batch experiment</div><h2 style="margin:7px 0 8px">120 synthetic disturbance runs</h2><div id="experiment" class="story"></div><div id="detectors" style="margin-top:8px"></div>
  </div>
  <div class="panel"><div class="kicker">Dispatcher copilot</div><h2 style="margin:7px 0 4px">Sentinel intervention</h2><div class="sub" style="margin-bottom:8px">Grounded explanation over PCC state and counterfactual controls. Core functionality stays local and CPU-only.</div><div id="selectedControl" class="alert"></div><div id="dispatcherBrief"></div><div style="margin-top:14px" class="kicker">Causal chain</div><div id="causalChain"></div><div style="margin-top:14px" class="kicker">Compare interventions</div><div id="recommendations"></div><div style="margin-top:18px" class="kicker">Ask Sentinel</div><div class="ask"><input id="question" value="Why are you recommending this now?"/><button onclick="askSentinel()">Ask</button></div><div id="answer" class="rec"></div></div>
 </div>
</div>
<script>
function pct(x){return Math.round(x*100)}
function metricCard(name,value,suffix=''){return `<div class="metric"><div class="name">${name}</div><div class="value">${value}${suffix}</div></div>`}
function drawChart(points,warning,failure){const c=document.getElementById('chart'),ctx=c.getContext('2d'),W=c.width,H=c.height,p={l:48,r:22,t:22,b:38};ctx.clearRect(0,0,W,H);ctx.strokeStyle='#294359';ctx.lineWidth=1;ctx.fillStyle='#7890a4';ctx.font='12px system-ui';for(let i=0;i<=4;i++){let y=p.t+(H-p.t-p.b)*i/4;ctx.beginPath();ctx.moveTo(p.l,y);ctx.lineTo(W-p.r,y);ctx.stroke();ctx.fillText((1-i/4).toFixed(2),8,y+4)}const maxT=points[points.length-1].time_hours;for(let i=0;i<=6;i++){let t=maxT*i/6,x=p.l+(W-p.l-p.r)*t/maxT;ctx.fillText(t.toFixed(1)+'h',x-10,H-12)}function vx(t){return p.l+(W-p.l-p.r)*t/maxT}function vy(v){return p.t+(H-p.t-p.b)*(1-v)}function marker(t,label,color){if(t==null)return;let x=vx(t);ctx.strokeStyle=color;ctx.setLineDash([6,5]);ctx.beginPath();ctx.moveTo(x,p.t);ctx.lineTo(x,H-p.b);ctx.stroke();ctx.setLineDash([]);ctx.fillStyle=color;ctx.fillText(label,x+5,p.t+13)}marker(warning,'SENTINEL WARNING','#f5b95f');marker(failure,'CONVENTIONAL FAILURE','#ef7c7c');const series=[['pressure','#69d7ff'],['entropy','#f5b95f'],['control','#93e7b9'],['instability','#ef7c7c']];for(const [key,color] of series){ctx.strokeStyle=color;ctx.lineWidth=3;ctx.beginPath();points.forEach((d,i)=>{const x=vx(d.time_hours),y=vy(d[key]);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke()}}
function drawControl(noCtl,ctl,controlTime){const c=document.getElementById('controlChart'),ctx=c.getContext('2d'),W=c.width,H=c.height,p={l:48,r:22,t:22,b:38};ctx.clearRect(0,0,W,H);ctx.strokeStyle='#294359';ctx.fillStyle='#7890a4';ctx.font='12px system-ui';for(let i=0;i<=4;i++){let y=p.t+(H-p.t-p.b)*i/4;ctx.beginPath();ctx.moveTo(p.l,y);ctx.lineTo(W-p.r,y);ctx.stroke();ctx.fillText((1-i/4).toFixed(2),8,y+4)}const maxT=noCtl[noCtl.length-1].time_hours;function vx(t){return p.l+(W-p.l-p.r)*t/maxT}function vy(v){return p.t+(H-p.t-p.b)*(1-v)}let x=vx(controlTime);ctx.strokeStyle='#f5b95f';ctx.setLineDash([6,5]);ctx.beginPath();ctx.moveTo(x,p.t);ctx.lineTo(x,H-p.b);ctx.stroke();ctx.setLineDash([]);ctx.fillStyle='#f5b95f';ctx.fillText('CONTROL',x+5,p.t+13);for(const [pts,color] of [[noCtl,'#ef7c7c'],[ctl,'#93e7b9']]){ctx.strokeStyle=color;ctx.lineWidth=3;ctx.beginPath();pts.forEach((d,i)=>{const xx=vx(d.time_hours),yy=vy(d.instability);i?ctx.lineTo(xx,yy):ctx.moveTo(xx,yy)});ctx.stroke()}ctx.fillStyle='#8fa7bc';ctx.fillText('red: no control   green: Sentinel control',p.l,H-12)}
let activeIncident='truck_breakdown';
async function setupScenarioLab(){const data=await fetch('/incidents').then(r=>r.json());const host=document.getElementById('scenarioButtons');host.innerHTML=data.incidents.map((x,i)=>`<button class="scenario-btn ${i===0?'active':''}" data-kind="${x.kind}" title="${x.description}" onclick="selectIncident('${x.kind}',this)">${x.label}</button>`).join('');await runScenario()}
function selectIncident(kind,el){activeIncident=kind;document.querySelectorAll('.scenario-btn[data-kind]').forEach(b=>b.classList.remove('active'));el.classList.add('active');runScenario()}
function drawScenario(noCtl,ctl,controlTime){const c=document.getElementById('scenarioChart'),ctx=c.getContext('2d'),W=c.width,H=c.height,p={l:48,r:22,t:22,b:38};ctx.clearRect(0,0,W,H);ctx.strokeStyle='#294359';ctx.fillStyle='#7890a4';ctx.font='12px system-ui';for(let i=0;i<=4;i++){let y=p.t+(H-p.t-p.b)*i/4;ctx.beginPath();ctx.moveTo(p.l,y);ctx.lineTo(W-p.r,y);ctx.stroke();ctx.fillText((1-i/4).toFixed(2),8,y+4)}const maxT=noCtl[noCtl.length-1].time_hours;function vx(t){return p.l+(W-p.l-p.r)*t/maxT}function vy(v){return p.t+(H-p.t-p.b)*(1-v)}if(controlTime!=null){let x=vx(controlTime);ctx.strokeStyle='#f5b95f';ctx.setLineDash([6,5]);ctx.beginPath();ctx.moveTo(x,p.t);ctx.lineTo(x,H-p.b);ctx.stroke();ctx.setLineDash([]);ctx.fillStyle='#f5b95f';ctx.fillText('SENTINEL',x+5,p.t+13)}for(const [pts,color] of [[noCtl,'#ef7c7c'],[ctl,'#93e7b9']]){ctx.strokeStyle=color;ctx.lineWidth=3;ctx.beginPath();pts.forEach((d,i)=>{const x=vx(d.time_hours),y=vy(d.instability);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke()}}
async function runScenario(){const sev=parseFloat(document.getElementById('severity').value);document.getElementById('scenarioStatus').textContent='SIMULATING';document.getElementById('scenarioSummary').innerHTML='Running CPU Monte Carlo counterfactuals...';const d=await fetch(`/interactive-scenario?incident=${activeIncident}&severity=${sev}`).then(r=>r.json());if(d.error){document.getElementById('scenarioSummary').textContent=d.error;return}document.getElementById('scenarioStatus').textContent=d.status;drawScenario(d.no_control,d.sentinel_control,d.warning_time_hours);const action=d.selected_action;document.getElementById('scenarioSummary').innerHTML=`<strong>${activeIncident.replaceAll('_',' ')}</strong><div class="small">Severity ${d.severity.toFixed(1)} · warning ${d.warning_time_hours==null?'not triggered':d.warning_time_hours.toFixed(2)+' h'} · hard failure ${d.failure_time_hours==null?'not reached':d.failure_time_hours.toFixed(2)+' h'}${d.warning_lead_time_hours==null?'':` · lead ${d.warning_lead_time_hours.toFixed(2)} h`}</div><div class="small" style="margin-top:8px"><b>Recommended:</b> ${action.label}</div><div class="small">Projected objective improvement vs no action: ${action.improvement_vs_no_action.toFixed(3)}</div>`;document.getElementById('operatorOutcome').innerHTML='';document.getElementById('scenarioRecs').innerHTML=d.recommendations.filter(x=>x.kind!=='none').slice(0,4).map((r,i)=>`<button class="choice ${r.action_id===action.action_id?'sentinel':''}" onclick="simulateOperatorChoice('${r.action_id.replaceAll("'","\\'")}')"><span class="rank">${i+1}.</span>${r.label}${r.action_id===action.action_id?' <span class="pill">Sentinel pick</span>':''}<div class="small">trajectory objective ${r.objective.toFixed(3)} · click to simulate override</div></button>`).join('')||'<div class="rec">No intervention required at this severity.</div>'}
async function simulateOperatorChoice(actionId){const sev=parseFloat(document.getElementById('severity').value);document.getElementById('operatorOutcome').innerHTML='<div class="rec">Simulating dispatcher choice...</div>';const d=await fetch(`/operator-counterfactual?incident=${activeIncident}&severity=${sev}&action_id=${encodeURIComponent(actionId)}`).then(r=>r.json());if(d.error){document.getElementById('operatorOutcome').innerHTML=`<div class="rec">${d.error}</div>`;return}drawScenario(d.no_control,d.chosen_control,d.control_time_hours);const regret=d.regret_vs_sentinel;const verdict=regret<=0.000001?'Matches Sentinel optimum':regret<0.03?'Near-optimal override':'Higher-risk override';document.getElementById('operatorOutcome').innerHTML=`<div class="alert" style="margin-top:10px"><strong>${verdict}</strong><div class="small">Chosen: ${d.chosen_action.label}</div><div class="decision-grid"><div class="small">vs no action<b>${d.chosen_improvement_vs_no_action>=0?'+':''}${d.chosen_improvement_vs_no_action.toFixed(3)}</b></div><div class="small">regret vs Sentinel<b>${d.regret_vs_sentinel.toFixed(3)}</b></div><div class="small">failure time<b>${d.chosen_failure_time_hours==null?'avoided / not reached':d.chosen_failure_time_hours.toFixed(2)+' h'}</b></div></div><div class="small" style="margin-top:8px">Sentinel pick: ${d.sentinel_action.label}. The chart now shows <b>your chosen control</b> in green against no control in red.</div></div>`} 
async function askSentinel(){const q=document.getElementById('question').value;document.getElementById('answer').textContent='Thinking over the simulated fleet state...';const r=await fetch('/dispatcher-query',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})}).then(r=>r.json());document.getElementById('answer').innerHTML=`<strong>${r.intent.replaceAll('_',' ')}</strong><div class="small">${r.answer}</div>`}
async function load(){const d=await fetch('/trajectory').then(r=>r.json()),pts=d.points,last=pts[pts.length-1],warn=pts.find(x=>x.early_warning)||pts[0];document.getElementById('metrics').innerHTML=[metricCard('Pressure',pct(warn.pressure)),metricCard('Entropy',pct(warn.entropy)),metricCard('dH/dt',warn.entropy_rate.toFixed(2),'/h'),metricCard('Control',pct(warn.control)),metricCard('Instability',pct(warn.instability)),metricCard('Cascade risk',pct(warn.cascade_risk))].join('');document.getElementById('warningTime').textContent=d.warning_time_hours==null?'—':d.warning_time_hours.toFixed(2)+' h';document.getElementById('failureTime').textContent=d.failure_time_hours==null?'—':d.failure_time_hours.toFixed(2)+' h';document.getElementById('leadTime').textContent=d.lead_time_hours==null?'—':d.lead_time_hours.toFixed(2)+' h';document.getElementById('alert').innerHTML=d.lead_time_hours!=null?`<strong>Early warning demonstrated:</strong> predictive entropy accelerates and Sentinel flags instability <strong>${d.lead_time_hours.toFixed(2)} hours</strong> before the synthetic conventional-failure threshold.`:`<strong>No pre-failure lead time</strong> in this seeded run.`;drawChart(pts,d.warning_time_hours,d.failure_time_hours);const recs=d.warning_recommendations.filter(x=>x.action.kind!=='none').slice(0,5);const cl=await fetch('/closed-loop').then(r=>r.json());const db=await fetch('/dispatcher-brief').then(r=>r.json());document.getElementById('selectedControl').innerHTML=`<strong>${db.headline}</strong><div class="small">${db.summary}</div><div class="small">Confidence: <b>${db.confidence}</b> · ${db.confidence_reason}</div>`;document.getElementById('dispatcherBrief').innerHTML=`<div class="rec"><strong>Why now?</strong><div class="small">${db.why_now}</div></div><div class="rec"><strong>No-action counterfactual</strong><div class="small">${db.no_action_consequence}</div></div><div class="rec"><strong>Expected effect</strong><div class="small">${db.expected_effect.join('<br>')}</div></div>`;document.getElementById('causalChain').innerHTML=db.causal_chain.map(x=>`<div class="factor"><strong>${x.name}</strong><span class="pill">${x.severity}</span><div class="small">${x.evidence}</div><div class="small">${x.implication}</div></div>`).join('');const alts=cl.alternatives.filter(x=>x.action.kind!=='none').slice(0,5);document.getElementById('recommendations').innerHTML=alts.map((r,i)=>`<div class="rec"><span class="rank">${i+1}.</span>${r.action.label}<div class="small">objective ${r.objective.toFixed(3)} · mean instability ${pct(r.mean_instability)}% · mean cascade ${pct(r.mean_cascade_risk)}%</div></div>`).join('')||'<div class="rec">No intervention generated.</div>';drawControl(cl.no_control,cl.sentinel_control,cl.control_time_hours);document.getElementById('controlStory').innerHTML=`<div class="small">Instability burden reduction<b>${pct(cl.instability_auc.reduction)}%</b></div><div class="small">Cascade burden reduction<b>${pct(cl.cascade_auc.reduction)}%</b></div><div class="small">Failure outcome<b>${cl.selected_outcome.failure_avoided?'Avoided':(cl.selected_outcome.failure_delay_hours!=null?'Delayed '+cl.selected_outcome.failure_delay_hours.toFixed(2)+' h':'Not avoided')}</b></div>`;const e=await fetch('/experiment').then(r=>r.json());document.getElementById('experiment').innerHTML=`<div class="small">Scenarios<b>${e.scenarios}</b></div><div class="small">Failures<b>${e.failures}</b></div><div class="small">Non-failures<b>${e.non_failures}</b></div>`;document.getElementById('detectors').innerHTML=e.detectors.map(x=>`<div class="rec"><strong>${x.name}</strong><div class="small">precision ${pct(x.precision)}% · recall ${pct(x.recall)}% · false-positive rate ${pct(x.false_positive_rate)}% · median lead ${x.median_lead_time_hours==null?'—':x.median_lead_time_hours.toFixed(2)+' h'}</div></div>`).join('')}
setupScenarioLab();load();askSentinel();
</script>
</body></html>'''
