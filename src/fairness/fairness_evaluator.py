"""
fairness/fairness_evaluator.py
Evaluates the trained model for bias across Job Role groups.

Note: The Kaggle dataset does not contain gender/ethnicity columns.
Fairness is evaluated across:
  - Job Role (hiring fairness across different job types)
  - Education level groups
  - Experience bands (junior / mid / senior)

Metrics:
  - Demographic Parity Difference (DPD)
  - Disparate Impact Ratio (DIR)
  - Equal Opportunity Difference (EOD)
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.preprocessing.resume_processor import ResumeProcessor

MODELS_DIR  = Path("models")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

CSV_PATH = "data/kaggle_resumes.csv"


class FairnessEvaluator:
    """Evaluates trained model for bias across available demographic groups."""

    def __init__(self):
        self.model     = joblib.load(MODELS_DIR / "best_model.pkl")
        self.scaler    = joblib.load(MODELS_DIR / "scaler.pkl")
        self.processor = ResumeProcessor()

    def evaluate(self, csv_path=CSV_PATH) -> dict:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]

        X    = self.processor.process_dataframe(df)
        X_sc = self.scaler.transform(X)

        df["predicted"]    = self.model.predict(X_sc)
        df["predict_prob"] = self.model.predict_proba(X_sc)[:, 1]
        df["shortlisted"]  = ResumeProcessor.get_target(df)

        # Create grouping columns
        df["experience_band"] = pd.cut(
            pd.to_numeric(df["Experience (Years)"], errors="coerce").fillna(0),
            bins=[-1, 2, 5, 10, 100],
            labels=["0–2 yrs", "3–5 yrs", "6–10 yrs", "10+ yrs"]
        )
        df["education_group"] = df["Education"].astype(str).str.strip()
        df["job_role_group"]  = df["Job Role"].astype(str).str.strip()

        report = {}
        report["job_role"]       = self._group_metrics(df, "job_role_group")
        report["education"]      = self._group_metrics(df, "education_group")
        report["experience_band"]= self._group_metrics(df, "experience_band")
        report["fairness_summary"] = self._fairness_summary(df, report)

        self._save_report(report)
        self._plot_selection_rates(df)
        self._plot_group_metrics(report)
        return report

    def _group_metrics(self, df, group_col):
        results = {}
        for group in sorted(df[group_col].dropna().unique(), key=str):
            mask   = df[group_col] == group
            y_true = df.loc[mask, "shortlisted"].values
            y_pred = df.loc[mask, "predicted"].values
            if len(y_true) < 3:
                continue
            results[str(group)] = {
                "n":              int(len(y_true)),
                "selection_rate": round(float(y_pred.mean()), 4),
                "true_rate":      round(float(y_true.mean()), 4),
                "precision":      round(precision_score(y_true, y_pred, zero_division=0), 4),
                "recall":         round(recall_score(y_true, y_pred, zero_division=0), 4),
                "f1":             round(f1_score(y_true, y_pred, zero_division=0), 4),
            }
        return results

    def _fairness_summary(self, df, report):
        summary = {}
        for attr in ["job_role", "education", "experience_band"]:
            groups = report.get(attr, {})
            if len(groups) < 2:
                continue
            rates   = {g: m["selection_rate"] for g, m in groups.items()}
            recalls = {g: m["recall"] for g, m in groups.items()}
            max_r, min_r = max(rates.values()), min(rates.values())
            dpd      = round(max_r - min_r, 4)
            dir_ratio= round(min_r / max_r, 4) if max_r > 0 else 1.0
            eod      = round(max(recalls.values()) - min(recalls.values()), 4)
            summary[attr] = {
                "demographic_parity_difference": dpd,
                "disparate_impact_ratio":        dir_ratio,
                "equal_opportunity_difference":  eod,
                "selection_rates":               rates,
                "dpd_acceptable":                dpd < 0.10,
                "dir_acceptable":                dir_ratio >= 0.80,
                "highest_rate_group":            max(rates, key=rates.get),
                "lowest_rate_group":             min(rates, key=rates.get),
            }
            print(f"\n--- Fairness: {attr.upper()} ---")
            print(f"  DPD: {dpd}  {'✅' if dpd < 0.10 else '⚠️  BIAS DETECTED'}")
            print(f"  DIR: {dir_ratio}  {'✅' if dir_ratio >= 0.80 else '⚠️  BIAS DETECTED'}")
            print(f"  EOD: {eod}")
        return summary

    def _save_report(self, report):
        out = RESULTS_DIR / "fairness_report.json"
        with open(out, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nFairness report → {out}")

    def _plot_selection_rates(self, df):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax, col, title in zip(
            axes,
            ["job_role_group", "education_group", "experience_band"],
            ["Job Role", "Education Level", "Experience Band"]
        ):
            grp = df.groupby(col)["predicted"].mean().reset_index()
            grp.columns = [col, "AI Selection Rate"]
            tru = df.groupby(col)["shortlisted"].mean().reset_index()
            tru.columns = [col, "Actual Rate"]
            merged = grp.merge(tru, on=col)

            x = np.arange(len(merged))
            ax.bar(x - 0.2, merged["AI Selection Rate"], 0.35,
                   label="AI Predicted", color="#2196F3", alpha=0.85)
            ax.bar(x + 0.2, merged["Actual Rate"], 0.35,
                   label="Actual Rate", color="#FF5722", alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(merged[col], rotation=20, ha="right", fontsize=9)
            ax.set_ylabel("Selection Rate")
            ax.set_title(f"Selection Rate by {title}")
            ax.set_ylim(0, 1.0)
            ax.axhline(df["predicted"].mean(), color="grey", linestyle="--",
                       alpha=0.5, label="Overall rate")
            ax.legend(fontsize=8)

        plt.suptitle("Fairness Analysis — Kaggle AI Recruitment Dataset",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "fairness_selection_rates.png", dpi=150)
        plt.close()
        print("Saved: results/fairness_selection_rates.png")

    def _plot_group_metrics(self, report):
        attrs  = [a for a in ["job_role", "education", "experience_band"] if report.get(a)]
        n      = len(attrs)
        if n == 0: return
        fig, axes = plt.subplots(1, n, figsize=(7*n, 5))
        if n == 1: axes = [axes]
        for ax, attr in zip(axes, attrs):
            groups = report[attr]
            rows = [{"Group": g, "Precision": m["precision"],
                     "Recall": m["recall"], "F1": m["f1"],
                     "Selection Rate": m["selection_rate"]}
                    for g, m in groups.items()]
            df_hm = pd.DataFrame(rows).set_index("Group")
            sns.heatmap(df_hm, ax=ax, annot=True, fmt=".3f",
                        cmap="YlGnBu", vmin=0, vmax=1, linewidths=0.5)
            ax.set_title(f"Per-Group Metrics — {attr.replace('_', ' ').title()}")
        plt.suptitle("Fairness Metrics Heatmap", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "fairness_heatmap.png", dpi=150)
        plt.close()
        print("Saved: results/fairness_heatmap.png")


if __name__ == "__main__":
    evaluator = FairnessEvaluator()
    report = evaluator.evaluate()
    print("\nFairness evaluation complete.")
