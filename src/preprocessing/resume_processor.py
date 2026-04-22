import re
from pathlib import Path
import numpy as np
import pandas as pd

EDUCATION_MAP = {
    "phd": 5, "doctorate": 5,
    "master": 4, "msc": 4, "mba": 4, "m.sc": 4, "m.tech": 4,
    "bachelor": 3, "bsc": 3, "b.sc": 3, "b.tech": 3, "b.e": 3, "b.com": 3,
    "associate": 2, "diploma": 2,
    "high school": 1, "secondary": 1,
    "none": 0,
}

KEY_SKILLS = [
    "python", "java", "sql", "machine learning", "data analysis",
    "communication", "leadership", "project management",
    "deep learning", "excel", "r", "tensorflow", "aws",
    "javascript", "c++", "statistics", "nlp"
]

class ResumeProcessor:

    def __init__(self):
        self.feature_names_ = None

    @staticmethod
    def normalise_columns(df):
        df.columns = [c.strip() for c in df.columns]
        return df

    @staticmethod
    def encode_target(series):
        cleaned = series.astype(str).str.strip().str.lower()
        print(f"\n  [Target] Unique values in CSV : {series.unique().tolist()}")
        POSITIVE = {"hire", "hired", "selected", "yes", "1", "accept", "accepted"}
        result = cleaned.apply(lambda v: 1 if v in POSITIVE else 0)
        print(f"  [Target] Selected(1): {result.sum()}")
        print(f"  [Target] Rejected(0): {(result==0).sum()}")
        return result

    @staticmethod
    def encode_education(series):
        def _encode(val):
            if pd.isna(val): return 0
            v = str(val).lower().strip()
            for key, score in EDUCATION_MAP.items():
                if key in v: return score
            return 1
        return series.apply(_encode)

    @staticmethod
    def count_certifications(series):
        def _count(val):
            if pd.isna(val) or str(val).strip().lower() in ("none","0","","nan"):
                return 0
            v = str(val)
            for sep in [";", ",", "|"]:
                if sep in v:
                    return len([x for x in v.split(sep) if x.strip()])
            return 1
        return series.apply(_count)

    @staticmethod
    def extract_skill_features(series):
        rows = []
        for val in series:
            skills_lower = set()
            if pd.notna(val):
                v = str(val).lower()
                for sep in ["|", ",", ";"]:
                    v = v.replace(sep, "|")
                skills_lower = {s.strip() for s in v.split("|") if s.strip()}
            row = {"total_skills": len(skills_lower)}
            for skill in KEY_SKILLS:
                row[f"skill_{skill.replace(' ', '_')}"] = (
                    1 if any(skill in s for s in skills_lower) else 0
                )
            rows.append(row)
        return pd.DataFrame(rows, index=series.index)

    @staticmethod
    def encode_job_role(series):
        role_rank = {
            "data scientist": 5, "ml engineer": 5, "machine learning": 5,
            "software engineer": 4, "data engineer": 4,
            "data analyst": 3, "business analyst": 3, "product manager": 3,
            "developer": 3, "analyst": 2, "manager": 2, "intern": 1,
        }
        def _rank(val):
            if pd.isna(val): return 2
            v = str(val).lower()
            for key, score in role_rank.items():
                if key in v: return score
            return 2
        return series.apply(_rank)

    def process_dataframe(self, df):
        df = self.normalise_columns(df.copy())
        features = pd.DataFrame(index=df.index)
        features["experience_years"] = (
            pd.to_numeric(df["Experience (Years)"], errors="coerce")
            .fillna(0).clip(0, 40)
        )
        features["education_level"]      = self.encode_education(df["Education"])
        features["certifications_count"] = self.count_certifications(df["Certifications"])
        features["projects_count"] = (
            pd.to_numeric(df["Projects Count"], errors="coerce")
            .fillna(0).clip(0, 50)
        )
        features["salary_expectation_k"] = (
            pd.to_numeric(df["Salary Expectation ($)"], errors="coerce")
            .fillna(0) / 1000.0
        )
        features["job_role_rank"] = self.encode_job_role(df["Job Role"])
        if "AI Score (0-100)" in df.columns:
            features["ai_score"] = (
                pd.to_numeric(df["AI Score (0-100)"], errors="coerce")
                .fillna(0).clip(0, 100)
            )
        skill_df = self.extract_skill_features(df["Skills"])
        features = pd.concat([features, skill_df], axis=1)
        features["exp_x_edu"]   = features["experience_years"] * features["education_level"]
        features["cert_x_proj"] = features["certifications_count"] * features["projects_count"]
        self.feature_names_ = list(features.columns)
        return features.astype(float)

    def get_feature_names(self):
        if self.feature_names_ is None:
            raise ValueError("Call process_dataframe() first.")
        return self.feature_names_

    @staticmethod
    def get_target(df):
        df = ResumeProcessor.normalise_columns(df.copy())
        return ResumeProcessor.encode_target(df["Recruiter Decision"])
