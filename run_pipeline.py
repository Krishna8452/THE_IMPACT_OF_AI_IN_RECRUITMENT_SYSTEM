"""
run_pipeline.py
Complete AI Recruitment pipeline using the real Kaggle dataset.

Usage:
    python3 run_pipeline.py              # full pipeline
    python3 run_pipeline.py --dashboard  # full pipeline + launch web UI
"""

import argparse, sys, time, json
from pathlib import Path

KAGGLE_CSV = "data/kaggle_resumes.csv"

def banner(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")

def check_dataset():
    banner("CHECKING DATASET")
    if not Path(KAGGLE_CSV).exists():
        print(f"""
  ❌  Dataset not found at: {KAGGLE_CSV}

  Please download the dataset from Kaggle:
  https://www.kaggle.com/datasets/mdtalhask/ai-powered-resume-screening-dataset-2025

  Then save it as:
  {KAGGLE_CSV}

  Expected columns:
  Resume_ID, Name, Skills, Experience (Years), Education,
  Certifications, Job Role, Recruiter Decision,
  Salary Expectation ($), Projects Count
""")
        sys.exit(1)

    import pandas as pd
    df = pd.read_csv(KAGGLE_CSV)
    df.columns = [c.strip() for c in df.columns]

    required = [
        "Resume_ID", "Name", "Skills", "Experience (Years)", "Education",
        "Certifications", "Job Role", "Recruiter Decision",
        "Salary Expectation ($)", "Projects Count"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  ❌  Missing columns: {missing}")
        print(f"  Found columns: {list(df.columns)}")
        sys.exit(1)

    print(f"  ✅  Dataset found: {len(df)} candidates, {len(df.columns)} columns")
    print(f"  Columns: {list(df.columns)}")
    sr = df["Recruiter Decision"].astype(str).str.strip().str.lower()
    selected = sr.isin(["hire","hired","selected","yes","accept","accepted","1"]).sum()
    print(f"  Selected: {selected} ({selected/len(df)*100:.1f}%)")
    print(f"  Rejected: {len(df)-selected} ({(len(df)-selected)/len(df)*100:.1f}%)")

def step1_preprocess():
    banner("STEP 1: Feature Extraction (Real Kaggle Data)")
    import pandas as pd
    from src.preprocessing.resume_processor import ResumeProcessor
    df = pd.read_csv(KAGGLE_CSV)
    processor = ResumeProcessor()
    X = processor.process_dataframe(df)
    y = ResumeProcessor.get_target(df)
    print(f"  Feature matrix: {X.shape[0]} rows × {X.shape[1]} features")
    print(f"  Features: {processor.get_feature_names()}")
    print(f"  Target — Selected: {y.sum()} | Rejected: {(y==0).sum()}")

def step2_train():
    banner("STEP 2: Train & Evaluate ML Models")
    from src.ml.ranking_engine import RecruitmentRankingEngine
    engine = RecruitmentRankingEngine()
    results = engine.train_evaluate()
    print(f"\n  {'Model':<30} {'F1':>6} {'AUC':>8} {'Precision':>10} {'Recall':>8}")
    print(f"  {'-'*60}")
    for name, m in results.items():
        print(f"  {name:<30} {str(m.get('f1','—')):>6} {str(m.get('roc_auc','—')):>8} "
              f"{str(m.get('precision','—')):>10} {str(m.get('recall','—')):>8}")
    return engine

def step3_fairness():
    banner("STEP 3: Fairness Evaluation")
    from src.fairness.fairness_evaluator import FairnessEvaluator
    report = FairnessEvaluator().evaluate()
    for attr, data in report.get("fairness_summary", {}).items():
        print(f"\n  {attr.upper().replace('_',' ')}:")
        print(f"    DPD: {data['demographic_parity_difference']}  {'✅' if data['dpd_acceptable'] else '⚠️'}")
        print(f"    DIR: {data['disparate_impact_ratio']}  {'✅' if data['dir_acceptable'] else '⚠️'}")
        print(f"    EOD: {data['equal_opportunity_difference']}")

def step4_explain():
    banner("STEP 4: SHAP Explainability")
    try:
        import shap
        from src.explainability.shap_explainer import SHAPExplainer
        SHAPExplainer().run_full()
    except ImportError:
        print("  SHAP not installed. Run: pip3 install shap")
    except Exception as e:
        print(f"  SHAP skipped: {e}")

def step5_summary():
    banner("STEP 5: Summary")
    try:
        with open("models/model_info.json") as f:
            info = json.load(f)
        m = info["metrics"]
        print(f"\n  Best Model:  {info['best_model_name']}")
        print(f"  F1 Score:    {m['f1']}")
        print(f"  ROC-AUC:     {m['roc_auc']}")
        print(f"  Precision:   {m['precision']}")
        print(f"  Recall:      {m['recall']}")
    except Exception as e:
        print(f"  Could not load model info: {e}")

    print("\n  Output files:")
    for f in sorted(Path("results").iterdir()):
        print(f"    📄 {f.name}")
    print("\n  Launch dashboard:  python3 dashboard/app.py")
    print("  Open browser:      http://localhost:5000")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dashboard", action="store_true")
    args = parser.parse_args()

    start = time.time()
    check_dataset()
    step1_preprocess()
    step2_train()
    step3_fairness()
    step4_explain()
    step5_summary()

    print(f"\n{'='*65}")
    print(f"  Pipeline complete in {time.time()-start:.1f} seconds")
    print(f"{'='*65}")

    if args.dashboard:
        import subprocess
        subprocess.run([sys.executable, "dashboard/app.py"])
