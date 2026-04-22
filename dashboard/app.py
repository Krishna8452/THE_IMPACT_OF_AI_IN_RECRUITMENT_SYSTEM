"""
dashboard/app.py
Flask recruiter dashboard — uses real Kaggle dataset columns.
Columns: Resume_ID, Name, Skills, Experience (Years), Education,
         Certifications, Job Role, Recruiter Decision,
         Salary Expectation ($), Projects Count
"""

import json, sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template_string, request

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.preprocessing.resume_processor import ResumeProcessor

app  = Flask(__name__)

MODELS_DIR = Path("models")
RESULTS_DIR= Path("results")
CSV_PATH   = "data/kaggle_resumes.csv"

model     = joblib.load(MODELS_DIR / "best_model.pkl")
scaler    = joblib.load(MODELS_DIR / "scaler.pkl")
processor = ResumeProcessor()

with open(MODELS_DIR / "model_info.json") as f:
    model_info = json.load(f)

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AI Recruitment System — Dashboard</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',Arial,sans-serif;background:#f4f6f9;color:#333}
header{background:#1F3A52;color:#fff;padding:16px 28px;display:flex;align-items:center;justify-content:space-between}
header h1{font-size:1.3rem}
header span{font-size:.82rem;opacity:.75}
.wrap{max-width:1100px;margin:24px auto;padding:0 18px}
.card{background:#fff;border-radius:8px;padding:22px;box-shadow:0 2px 8px rgba(0,0,0,.08);margin-bottom:22px}
.card h2{font-size:1rem;color:#1F3A52;margin-bottom:14px;border-bottom:2px solid #e8edf2;padding-bottom:7px}
.g4{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.stat{background:#f0f4f8;border-radius:6px;padding:14px;text-align:center}
.stat .v{font-size:1.9rem;font-weight:700;color:#1F3A52}
.stat .l{font-size:.78rem;color:#666;margin-top:3px}
label{font-size:.83rem;color:#555;display:block;margin-bottom:4px}
input,select{width:100%;padding:7px 10px;border:1px solid #ccc;border-radius:5px;font-size:.88rem;margin-bottom:12px}
button{background:#1F3A52;color:#fff;padding:9px 22px;border:none;border-radius:5px;cursor:pointer;font-size:.92rem;width:100%}
button:hover{background:#2E5878}
#res{display:none;margin-top:16px;padding:14px;border-radius:6px}
.ok{background:#e8f5e9;border-left:4px solid #4CAF50}
.no{background:#fce4ec;border-left:4px solid #F44336}
.rt{font-weight:700;font-size:1.05rem;margin-bottom:6px}
.bar{background:#e0e0e0;border-radius:8px;height:9px;margin:6px 0}
.fill{height:9px;border-radius:8px}
.expl{font-size:.87rem;color:#444;margin-top:8px;line-height:1.5}
.factor{display:flex;justify-content:space-between;padding:3px 0;font-size:.83rem;border-bottom:1px solid #f0f0f0}
.pos{color:#388E3C}.neg{color:#D32F2F}
.badge{display:inline-block;padding:2px 9px;border-radius:10px;font-size:.72rem;font-weight:700}
.b-ok{background:#e8f5e9;color:#2E7D32}.b-warn{background:#fff3e0;color:#E65100}
table{width:100%;border-collapse:collapse;font-size:.83rem}
th{background:#1F3A52;color:#fff;padding:7px 10px;text-align:left}
td{padding:7px 10px;border-bottom:1px solid #f0f0f0}
</style>
</head>
<body>
<header>
  <h1>🤖 AI Recruitment System</h1>
  <span>Real Kaggle Dataset · Explainable · Fair</span>
</header>
<div class="wrap">

  <div class="card">
    <h2>Model Performance</h2>
    <div class="g4">
      <div class="stat"><div class="v" id="sf1">—</div><div class="l">F1 Score</div></div>
      <div class="stat"><div class="v" id="sau">—</div><div class="l">ROC-AUC</div></div>
      <div class="stat"><div class="v" id="spr">—</div><div class="l">Precision</div></div>
      <div class="stat"><div class="v" id="sre">—</div><div class="l">Recall</div></div>
    </div>
    <p style="font-size:.8rem;color:#888;margin-top:10px" id="mname"></p>
  </div>

  <div class="g2">
    <div class="card">
      <h2>Score a Candidate</h2>
      <label>Years of Experience</label>
      <input type="number" id="exp" min="0" max="40" value="4">
      <label>Education</label>
      <select id="edu">
        <option>Bachelor</option><option>Master</option><option>PhD</option>
        <option>Associate</option><option>High School</option>
      </select>
      <label>Skills (comma or pipe separated)</label>
      <input type="text" id="skills" value="Python, Machine Learning, SQL">
      <label>Certifications (comma separated or None)</label>
      <input type="text" id="certs" value="AWS Certified, PMP">
      <label>Job Role</label>
      <select id="role">
        <option>Data Scientist</option><option>Software Engineer</option>
        <option>Data Analyst</option><option>ML Engineer</option>
        <option>Business Analyst</option><option>Product Manager</option>
      </select>
      <label>Salary Expectation ($)</label>
      <input type="number" id="salary" value="85000">
      <label>Projects Count</label>
      <input type="number" id="projects" min="0" value="5">
      <button onclick="score()">Score Candidate</button>

      <div id="res">
        <div class="rt" id="rtitle"></div>
        <div style="font-size:.88rem;margin-bottom:4px">
          Probability: <strong id="rprob"></strong>
        </div>
        <div class="bar"><div class="fill" id="rfill" style="width:0%"></div></div>
        <div class="expl" id="rexpl"></div>
        <div id="rfactors"></div>
      </div>
    </div>

    <div class="card">
      <h2>Fairness Summary</h2>
      <div id="fairness-content"><p style="color:#888;font-size:.88rem">Loading...</p></div>
    </div>
  </div>

  <div class="card">
    <h2>Dataset Overview</h2>
    <div id="ds" style="color:#888;font-size:.88rem">Loading...</div>
  </div>

</div>
<script>
async function loadStats(){
  const d = await (await fetch('/api/model-info')).json();
  document.getElementById('sf1').textContent  = d.metrics?.f1 ?? '—';
  document.getElementById('sau').textContent  = d.metrics?.roc_auc ?? '—';
  document.getElementById('spr').textContent  = d.metrics?.precision ?? '—';
  document.getElementById('sre').textContent  = d.metrics?.recall ?? '—';
  document.getElementById('mname').textContent= 'Best model: ' + (d.best_model_name ?? '');
}
async function loadFairness(){
  try{
    const d = await (await fetch('/api/fairness-summary')).json();
    let h='';
    for(const [attr,data] of Object.entries(d)){
      const dpd=data.demographic_parity_difference, dir=data.disparate_impact_ratio;
      h+=`<div style="margin-bottom:12px"><strong>${attr.replace(/_/g,' ').replace(/\b\w/g,c=>c.toUpperCase())}</strong>
      <table style="margin-top:5px">
        <tr><th>Metric</th><th>Value</th><th>Status</th></tr>
        <tr><td>Demographic Parity Diff</td><td>${dpd}</td>
          <td><span class="badge ${data.dpd_acceptable?'b-ok':'b-warn'}">${data.dpd_acceptable?'✅ OK':'⚠️ Review'}</span></td></tr>
        <tr><td>Disparate Impact Ratio</td><td>${dir}</td>
          <td><span class="badge ${data.dir_acceptable?'b-ok':'b-warn'}">${data.dir_acceptable?'✅ OK':'⚠️ Review'}</span></td></tr>
      </table>
      <div style="font-size:.78rem;color:#666;margin-top:4px">Highest: ${data.highest_rate_group} | Lowest: ${data.lowest_rate_group}</div>
      </div>`;
    }
    document.getElementById('fairness-content').innerHTML = h || '<p>Run fairness evaluation first.</p>';
  }catch{
    document.getElementById('fairness-content').innerHTML='<p style="color:red">Run: python3 src/fairness/fairness_evaluator.py</p>';
  }
}
async function loadDS(){
  try{
    const d = await (await fetch('/api/candidates')).json();
    document.getElementById('ds').innerHTML=
      `Total candidates: <strong>${d.total}</strong> &nbsp;|&nbsp;
       Selected rate: <strong>${d.selected_rate}</strong> &nbsp;|&nbsp;
       Avg experience: <strong>${d.avg_experience} yrs</strong> &nbsp;|&nbsp;
       Avg projects: <strong>${d.avg_projects}</strong>`;
  }catch{}
}
async function score(){
  const payload={
    "Experience (Years)": parseFloat(document.getElementById('exp').value),
    "Education":          document.getElementById('edu').value,
    "Skills":             document.getElementById('skills').value,
    "Certifications":     document.getElementById('certs').value,
    "Job Role":           document.getElementById('role').value,
    "Salary Expectation ($)": parseFloat(document.getElementById('salary').value),
    "Projects Count":     parseInt(document.getElementById('projects').value),
    "Resume_ID": "LIVE-001", "Name": "Candidate",
    "Recruiter Decision": "Selected"
  };
  try{
    const d = await (await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)})).json();
    const box=document.getElementById('res');
    box.style.display='block';
    box.className=d.shortlisted?'ok':'no';
    document.getElementById('rtitle').textContent=d.shortlisted?'✅ Recommended for Shortlisting':'❌ Not Recommended';
    const pct=Math.round(d.shortlist_probability*100);
    document.getElementById('rprob').textContent=pct+'%';
    document.getElementById('rfill').style.width=pct+'%';
    document.getElementById('rfill').style.background=d.shortlisted?'#4CAF50':'#F44336';
    document.getElementById('rexpl').textContent=d.explanation||'';
    let fh='<div style="margin-top:9px;font-weight:700;font-size:.83rem">Key factors:</div>';
    (d.top_positive||[]).slice(0,3).forEach(f=>{fh+=`<div class="factor"><span class="pos">▲ ${f.display_name}</span><span>+${f.shap_value.toFixed(3)}</span></div>`;});
    (d.top_negative||[]).slice(0,3).forEach(f=>{fh+=`<div class="factor"><span class="neg">▼ ${f.display_name}</span><span>${f.shap_value.toFixed(3)}</span></div>`;});
    document.getElementById('rfactors').innerHTML=fh;
  }catch(e){alert('Error: '+e.message);}
}
window.onload=()=>{loadStats();loadFairness();loadDS();};
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        df_input = pd.DataFrame([{
            "Resume_ID":             "LIVE",
            "Name":                  data.get("Name", "Candidate"),
            "Skills":                data.get("Skills", ""),
            "Experience (Years)":    data.get("Experience (Years)", 0),
            "Education":             data.get("Education", "Bachelor"),
            "Certifications":        data.get("Certifications", "None"),
            "Job Role":              data.get("Job Role", "Data Analyst"),
            "Recruiter Decision":    "Selected",
            "Salary Expectation ($)":data.get("Salary Expectation ($)", 60000),
            "Projects Count":        data.get("Projects Count", 0),
        }])
        X    = processor.process_dataframe(df_input)
        X_sc = scaler.transform(X)
        prob = float(model.predict_proba(X_sc)[0][1])
        pred = prob >= 0.5

        explanation_text = ""
        top_positive, top_negative = [], []
        try:
            import shap
            from src.explainability.shap_explainer import FEATURE_DISPLAY_NAMES
            model_type = type(model).__name__
            if model_type in ["RandomForestClassifier","GradientBoostingClassifier","DecisionTreeClassifier"]:
                exp = shap.TreeExplainer(model)
                sv  = exp.shap_values(X_sc)
                shap_row = sv[1][0] if isinstance(sv, list) else sv[0]
            else:
                exp = shap.LinearExplainer(model, X_sc)
                shap_row = exp.shap_values(X_sc)[0]

            fn = processor.get_feature_names()
            for name, val, sv_val in zip(fn, X_sc[0], shap_row):
                entry = {
                    "feature":      name,
                    "display_name": FEATURE_DISPLAY_NAMES.get(name, name),
                    "shap_value":   round(float(sv_val), 4),
                    "raw_value":    round(float(val), 4),
                }
                (top_positive if sv_val > 0 else top_negative).append(entry)
            top_positive = sorted(top_positive, key=lambda x: x["shap_value"], reverse=True)[:5]
            top_negative = sorted(top_negative, key=lambda x: x["shap_value"])[:5]
            pos = ", ".join(e["display_name"] for e in top_positive[:3])
            neg = ", ".join(e["display_name"] for e in top_negative[:3])
            explanation_text = (
                f"{'Recommended' if pred else 'Not recommended'} "
                f"({prob:.1%} probability). "
                f"{'Strengths: ' + pos + '.' if pred else 'Limiting factors: ' + neg + '.'}"
            )
        except Exception:
            explanation_text = f"{'Recommended' if pred else 'Not recommended'} ({prob:.1%} probability)."

        return jsonify({
            "shortlist_probability": round(prob, 4),
            "shortlisted":           bool(pred),
            "confidence":            "High" if abs(prob - 0.5) > 0.3 else "Medium",
            "explanation":           explanation_text,
            "top_positive":          top_positive,
            "top_negative":          top_negative,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/model-info")
def model_info_route():
    return jsonify(model_info)


@app.route("/api/fairness-summary")
def fairness_summary():
    p = RESULTS_DIR / "fairness_report.json"
    if not p.exists():
        return jsonify({"error": "Run: python3 src/fairness/fairness_evaluator.py"}), 404
    with open(p) as f:
        return jsonify(json.load(f).get("fairness_summary", {}))


@app.route("/api/candidates")
def candidates_stats():
    try:
        df = pd.read_csv(CSV_PATH)
        df.columns = [c.strip() for c in df.columns]
        y  = ResumeProcessor.get_target(df)
        return jsonify({
            "total":          len(df),
            "selected_rate":  f"{y.mean()*100:.1f}%",
            "avg_experience": round(pd.to_numeric(df["Experience (Years)"], errors="coerce").mean(), 1),
            "avg_projects":   round(pd.to_numeric(df["Projects Count"], errors="coerce").mean(), 1),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("\n AI Recruitment Dashboard")
    print("=" * 40)
    print("Open: http://localhost:5000")
    print("=" * 40)
    app.run(debug=True, host="0.0.0.0", port=5000)
