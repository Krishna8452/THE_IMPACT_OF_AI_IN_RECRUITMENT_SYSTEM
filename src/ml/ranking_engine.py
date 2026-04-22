"""
ml/ranking_engine.py
Trains and evaluates ML models on the real Kaggle recruitment dataset.
Compares 4 ML models against a rule-based keyword baseline.

Dataset: data/kaggle_resumes.csv
Columns: Resume_ID, Name, Skills, Experience (Years), Education,
         Certifications, Job Role, Recruiter Decision,
         Salary Expectation ($), Projects Count
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
    roc_curve, ConfusionMatrixDisplay
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.preprocessing.resume_processor import ResumeProcessor

MODELS_DIR  = Path("models")
RESULTS_DIR = Path("results")
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

CSV_PATH = "data/kaggle_resumes.csv"


# ── Rule-based baseline ───────────────────────────────────────────────────────
class RuleBasedBaseline:
    """
    Traditional keyword-based recruitment filter.
    Mirrors conventional HR screening using Experience + Education + Skills.
    """
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        preds = []
        for _, row in df.iterrows():
            skills_text = str(row.get("Skills", "")).lower()
            exp         = float(str(row.get("Experience (Years)", 0)).replace(",", "") or 0)
            edu         = str(row.get("Education", "")).lower()
            certs       = str(row.get("Certifications", ""))
            projects    = float(str(row.get("Projects Count", 0)).replace(",", "") or 0)

            has_degree  = any(k in edu for k in ["bachelor", "master", "phd", "bsc", "msc"])
            has_skills  = len([s for s in ["python","sql","java","machine learning","data"] if s in skills_text]) >= 2
            has_cert    = certs.strip().lower() not in ("none", "0", "", "nan")

            score = (
                min(exp, 10) * 0.4
                + (4 if has_degree else 0)
                + (2 if has_skills else 0)
                + (1 if has_cert else 0)
                + min(projects, 10) * 0.2
            )
            preds.append(1 if score >= 6.0 else 0)
        return np.array(preds)


# ── ML Ranking Engine ─────────────────────────────────────────────────────────
class RecruitmentRankingEngine:

    CANDIDATES = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, random_state=42, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=8, random_state=42,
            class_weight="balanced", n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            random_state=42
        ),
    }

    def __init__(self):
        self.best_model_name = None
        self.best_model      = None
        self.scaler          = StandardScaler()
        self.processor       = ResumeProcessor()
        self.results_        = {}
        self.feature_names_  = None

    def load_data(self, csv_path=CSV_PATH):
        if not Path(csv_path).exists():
            raise FileNotFoundError(
                f"\n  Dataset not found: {csv_path}\n"
                f"  Download from: https://www.kaggle.com/datasets/mdtalhask/ai-powered-resume-screening-dataset-2025\n"
                f"  Save as: {csv_path}"
            )
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]

        print(f"Loaded Kaggle dataset: {len(df)} candidates")
        print(f"Columns: {list(df.columns)}")

        X = self.processor.process_dataframe(df)
        y = ResumeProcessor.get_target(df)

        print(f"\nTarget distribution:")
        print(f"  Selected (1): {y.sum()} ({y.mean()*100:.1f}%)")
        print(f"  Rejected (0): {(y==0).sum()} ({(y==0).mean()*100:.1f}%)")

        self.feature_names_ = self.processor.get_feature_names()
        self.df_raw_        = df
        return X, y, df

    def train_evaluate(self, csv_path=CSV_PATH):
        X, y, df_raw = self.load_data(csv_path)

        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, np.arange(len(y)), test_size=0.2, random_state=42, stratify=y
        )
        X_train_sc = self.scaler.fit_transform(X_train)
        X_test_sc  = self.scaler.transform(X_test)

        print(f"\nTraining set: {len(X_train)} | Test set: {len(X_test)}")
        print("=" * 65)

        best_f1 = 0
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for name, model in self.CANDIDATES.items():
            model.fit(X_train_sc, y_train)
            y_pred = model.predict(X_test_sc)
            y_prob = model.predict_proba(X_test_sc)[:, 1]
            cv_scores = cross_val_score(model, X_train_sc, y_train, cv=cv, scoring="f1")

            metrics = {
                "precision":  round(precision_score(y_test, y_pred, zero_division=0), 4),
                "recall":     round(recall_score(y_test, y_pred, zero_division=0), 4),
                "f1":         round(f1_score(y_test, y_pred, zero_division=0), 4),
                "roc_auc":    round(roc_auc_score(y_test, y_prob), 4),
                "cv_f1_mean": round(cv_scores.mean(), 4),
                "cv_f1_std":  round(cv_scores.std(), 4),
            }
            self.results_[name] = metrics

            print(f"\n{name}")
            print(f"  Precision:  {metrics['precision']}")
            print(f"  Recall:     {metrics['recall']}")
            print(f"  F1 Score:   {metrics['f1']}")
            print(f"  ROC-AUC:    {metrics['roc_auc']}")
            print(f"  CV F1:      {metrics['cv_f1_mean']} ± {metrics['cv_f1_std']}")

            if metrics["f1"] > best_f1:
                best_f1              = metrics["f1"]
                self.best_model_name = name
                self.best_model      = model
                self.X_test_         = X_test_sc
                self.y_test_         = y_test
                self.y_pred_         = y_pred
                self.y_prob_         = y_prob
                self.idx_test_       = idx_test
                self.df_test_        = df_raw.iloc[idx_test].copy()

        # Rule-based baseline
        baseline = RuleBasedBaseline()
        bp = baseline.predict(self.df_test_)
        self.results_["Rule-Based Baseline"] = {
            "precision": round(precision_score(y_test, bp, zero_division=0), 4),
            "recall":    round(recall_score(y_test, bp, zero_division=0), 4),
            "f1":        round(f1_score(y_test, bp, zero_division=0), 4),
            "roc_auc":   "N/A",
            "cv_f1_mean":"N/A",
            "cv_f1_std": "N/A",
        }

        print(f"\n{'='*65}")
        print(f"Best model: {self.best_model_name} (F1={best_f1})")

        self._save_results()
        self._save_model()
        self._plot_all()
        return self.results_

    def _save_model(self):
        joblib.dump(self.best_model, MODELS_DIR / "best_model.pkl")
        joblib.dump(self.scaler,     MODELS_DIR / "scaler.pkl")
        with open(MODELS_DIR / "feature_names.json", "w") as f:
            json.dump(self.feature_names_, f, indent=2)
        with open(MODELS_DIR / "model_info.json", "w") as f:
            json.dump({
                "best_model_name": self.best_model_name,
                "metrics": self.results_[self.best_model_name]
            }, f, indent=2)
        print(f"\nModel saved → {MODELS_DIR}/")

    def _save_results(self):
        rows = [{"Model": k, **v} for k, v in self.results_.items()]
        pd.DataFrame(rows).to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    def _plot_all(self):
        self._plot_model_comparison()
        self._plot_confusion_matrix()
        self._plot_roc_curve()
        self._plot_feature_importance()

    def _plot_model_comparison(self):
        models = [k for k in self.results_ if k != "Rule-Based Baseline"]
        baseline = self.results_.get("Rule-Based Baseline", {})
        metrics = ["precision", "recall", "f1"]
        x = np.arange(len(models))
        w = 0.25
        fig, ax = plt.subplots(figsize=(11, 5))
        for i, metric in enumerate(metrics):
            vals = [self.results_[m][metric] for m in models]
            ax.bar(x + i*w, vals, w, label=metric.capitalize(), alpha=0.85)
            if isinstance(baseline.get(metric), float):
                ax.axhline(y=baseline[metric], color=f"C{i}", linestyle="--",
                           alpha=0.4, label=f"Baseline {metric}" if i==0 else "")
        ax.set_xticks(x + w)
        ax.set_xticklabels(models, rotation=12, ha="right")
        ax.set_ylabel("Score")
        ax.set_title("Model Comparison: AI vs Rule-Based Baseline\n(Kaggle AI Recruitment Dataset)")
        ax.legend(loc="lower right")
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "model_comparison.png", dpi=150)
        plt.close()
        print("Saved: results/model_comparison.png")

    def _plot_confusion_matrix(self):
        fig, ax = plt.subplots(figsize=(5, 4))
        cm = confusion_matrix(self.y_test_, self.y_pred_)
        ConfusionMatrixDisplay(cm, display_labels=["Rejected", "Selected"]).plot(
            ax=ax, colorbar=False, cmap="Blues"
        )
        ax.set_title(f"Confusion Matrix — {self.best_model_name}")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
        plt.close()
        print("Saved: results/confusion_matrix.png")

    def _plot_roc_curve(self):
        fpr, tpr, _ = roc_curve(self.y_test_, self.y_prob_)
        auc = roc_auc_score(self.y_test_, self.y_prob_)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, lw=2, label=f"{self.best_model_name} (AUC={auc:.3f})")
        plt.plot([0,1],[0,1],"k--", alpha=0.4, label="Random (AUC=0.5)")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve — Best Model")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "roc_curve.png", dpi=150)
        plt.close()
        print("Saved: results/roc_curve.png")

    def _plot_feature_importance(self):
        if not hasattr(self.best_model, "feature_importances_"):
            return
        importances = self.best_model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]
        names   = [self.feature_names_[i] for i in indices]
        plt.figure(figsize=(8, 5))
        plt.barh(names[::-1], importances[indices][::-1], color="#2196F3", alpha=0.85)
        plt.xlabel("Importance")
        plt.title(f"Top 15 Feature Importances — {self.best_model_name}")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "feature_importance.png", dpi=150)
        plt.close()
        print("Saved: results/feature_importance.png")

    def predict_candidate(self, candidate_dict: dict) -> dict:
        df_single = pd.DataFrame([candidate_dict])
        X  = self.processor.process_dataframe(df_single)
        Xs = self.scaler.transform(X)
        prob = float(self.best_model.predict_proba(Xs)[0][1])
        return {
            "shortlist_probability": round(prob, 4),
            "shortlisted": prob >= 0.5,
            "confidence": "High" if abs(prob - 0.5) > 0.3 else "Medium"
        }


if __name__ == "__main__":
    engine = RecruitmentRankingEngine()
    results = engine.train_evaluate()
    print("\n--- Final Comparison ---")
    for model, m in results.items():
        print(f"{model:<30}  F1={m['f1']}  AUC={m.get('roc_auc','N/A')}")
