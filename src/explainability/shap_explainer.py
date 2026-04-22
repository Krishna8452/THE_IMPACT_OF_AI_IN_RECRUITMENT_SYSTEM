"""
explainability/shap_explainer.py
SHAP explanations for the Kaggle recruitment dataset.
"""

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.preprocessing.resume_processor import ResumeProcessor

MODELS_DIR  = Path("models")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

CSV_PATH = "data/kaggle_resumes.csv"

FEATURE_DISPLAY_NAMES = {
    "experience_years":        "Years of Experience",
    "education_level":         "Education Level",
    "certifications_count":    "Number of Certifications",
    "projects_count":          "Projects Completed",
    "salary_expectation_k":    "Salary Expectation (K)",
    "job_role_rank":           "Job Role Level",
    "total_skills":            "Total Skills Listed",
    "exp_x_edu":               "Experience × Education",
    "cert_x_proj":             "Certifications × Projects",
    "skill_python":            "Python Skill",
    "skill_java":              "Java Skill",
    "skill_sql":               "SQL Skill",
    "skill_machine_learning":  "Machine Learning Skill",
    "skill_data_analysis":     "Data Analysis Skill",
    "skill_communication":     "Communication Skill",
    "skill_leadership":        "Leadership Skill",
    "skill_project_management":"Project Management",
    "skill_deep_learning":     "Deep Learning Skill",
    "skill_excel":             "Excel Skill",
    "skill_r":                 "R Skill",
    "skill_tensorflow":        "TensorFlow Skill",
    "skill_aws":               "AWS Skill",
    "skill_javascript":        "JavaScript Skill",
    "skill_c++":               "C++ Skill",
    "skill_statistics":        "Statistics Skill",
    "skill_nlp":               "NLP Skill",
}


class SHAPExplainer:

    def __init__(self):
        self.model     = joblib.load(MODELS_DIR / "best_model.pkl")
        self.scaler    = joblib.load(MODELS_DIR / "scaler.pkl")
        self.processor = ResumeProcessor()
        self.explainer   = None
        self.shap_values = None
        with open(MODELS_DIR / "model_info.json") as f:
            info = json.load(f)
        self.model_name = info.get("best_model_name", "Model")

    def fit(self, csv_path=CSV_PATH):
        import shap
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        X  = self.processor.process_dataframe(df)
        self.feature_names_ = self.processor.get_feature_names()
        self.X_sc_ = self.scaler.transform(X)
        self.df_   = df

        print(f"Computing SHAP values for {self.model_name}...")
        model_type = type(self.model).__name__
        if model_type in ["RandomForestClassifier", "GradientBoostingClassifier", "DecisionTreeClassifier"]:
            self.explainer   = shap.TreeExplainer(self.model)
            sv = self.explainer.shap_values(self.X_sc_)
            self.shap_values = sv[1] if isinstance(sv, list) else sv
        else:
            self.explainer   = shap.LinearExplainer(
                self.model, self.X_sc_, feature_perturbation="correlation_dependent"
            )
            self.shap_values = self.explainer.shap_values(self.X_sc_)

        print(f"SHAP values computed. Shape: {self.shap_values.shape}")
        self._save_shap_csv()
        return self

    def _save_shap_csv(self):
        mean_abs = np.abs(self.shap_values).mean(axis=0)
        pd.DataFrame({
            "feature":        self.feature_names_,
            "display_name":   [FEATURE_DISPLAY_NAMES.get(f, f) for f in self.feature_names_],
            "mean_abs_shap":  mean_abs.round(5),
        }).sort_values("mean_abs_shap", ascending=False).to_csv(
            RESULTS_DIR / "shap_feature_importance.csv", index=False
        )
        print("Saved: results/shap_feature_importance.csv")

    def plot_global_summary(self):
        import shap
        display = [FEATURE_DISPLAY_NAMES.get(f, f) for f in self.feature_names_]
        plt.figure(figsize=(9, 7))
        shap.summary_plot(self.shap_values, self.X_sc_, feature_names=display,
                          plot_type="dot", show=False, max_display=15)
        plt.title(f"SHAP Feature Impact — {self.model_name}", pad=15)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved: results/shap_summary.png")

    def plot_bar_importance(self):
        import shap
        display = [FEATURE_DISPLAY_NAMES.get(f, f) for f in self.feature_names_]
        plt.figure(figsize=(8, 6))
        shap.summary_plot(self.shap_values, self.X_sc_, feature_names=display,
                          plot_type="bar", show=False, max_display=15)
        plt.title(f"SHAP Global Feature Importance — {self.model_name}", pad=15)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "shap_bar_importance.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved: results/shap_bar_importance.png")

    def explain_candidate(self, idx: int) -> dict:
        shap_row = self.shap_values[idx]
        fvals    = self.X_sc_[idx]
        prob     = float(self.model.predict_proba(fvals.reshape(1, -1))[0][1])

        positives, negatives = [], []
        for name, val, sv in zip(self.feature_names_, fvals, shap_row):
            entry = {
                "feature":      name,
                "display_name": FEATURE_DISPLAY_NAMES.get(name, name),
                "shap_value":   round(float(sv), 4),
                "raw_value":    round(float(val), 4),
            }
            (positives if sv > 0 else negatives).append(entry)

        positives = sorted(positives, key=lambda x: x["shap_value"], reverse=True)[:5]
        negatives = sorted(negatives, key=lambda x: x["shap_value"])[:5]
        pos_str = ", ".join(e["display_name"] for e in positives[:3])
        neg_str = ", ".join(e["display_name"] for e in negatives[:3])

        if prob >= 0.5:
            explanation = (
                f"This candidate is recommended for shortlisting "
                f"(probability {prob:.1%}). "
                f"Key supporting factors: {pos_str}. "
                f"Minor concerns: {neg_str}."
            )
        else:
            explanation = (
                f"This candidate is not recommended "
                f"(probability {prob:.1%}). "
                f"Main limiting factors: {neg_str}. "
                f"Positive signals: {pos_str}."
            )

        return {
            "candidate_index": idx,
            "shortlist_prob":  round(prob, 4),
            "shortlisted":     prob >= 0.5,
            "top_positive":    positives,
            "top_negative":    negatives,
            "explanation":     explanation,
        }

    def run_full(self):
        self.fit()
        self.plot_global_summary()
        self.plot_bar_importance()
        preds = self.model.predict(self.X_sc_)
        for idx in [np.where(preds == 1)[0][0], np.where(preds == 0)[0][0]]:
            result = self.explain_candidate(idx)
            print(f"\n--- Candidate {idx} ---")
            print(result["explanation"])
        return self


if __name__ == "__main__":
    SHAPExplainer().run_full()
    print("\nSHAP analysis complete.")
