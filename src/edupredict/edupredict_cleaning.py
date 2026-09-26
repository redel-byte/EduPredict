from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "dataset.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "edupredict_pipeline.joblib"

NUMERIC_COLUMNS = [
    "Hours_Studied",
    "Attendance",
    "Sleep_Hours",
    "Previous_Scores",
    "Tutoring_Sessions",
    "Physical_Activity",
]

ORDINAL_COLUMNS = [
    "Parental_Involvement",
    "Access_to_Resources",
    "Motivation_Level",
    "Family_Income",
    "Teacher_Quality",
    "Parental_Education_Level",
    "Distance_from_Home",
]

ORDINAL_CATEGORIES = [
    ["Low", "Medium", "High"],
    ["Low", "Medium", "High"],
    ["Low", "Medium", "High"],
    ["Low", "Medium", "High"],
    ["Low", "Medium", "High"],
    ["High School", "College", "Postgraduate"],
    ["Near", "Moderate", "Far"],
]

NOMINAL_COLUMNS = [
    "Gender",
    "School_Type",
    "Extracurricular_Activities",
    "Internet_Access",
    "Learning_Disabilities",
    "Peer_Influence",
]


def build_model_pipeline():
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    ordinal_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    categories=ORDINAL_CATEGORIES,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )
    nominal_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
            ("ordinal", ordinal_pipeline, ORDINAL_COLUMNS),
            ("nominal", nominal_pipeline, NOMINAL_COLUMNS),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=300,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def main():
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    valid_target = df["Exam_Score"].between(0, 100)
    removed_rows = int((~valid_target).sum())
    df = df.loc[valid_target].copy()

    X = df.drop(columns="Exam_Score")
    y = df["Exam_Score"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = build_model_pipeline()
    model.fit(X_train, y_train)
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    print(f"Rows removed for invalid scores: {removed_rows}")
    print(f"Rows used: {len(df)}")
    print(f"Train R²: {r2_score(y_train, train_predictions):.4f}")
    print(f"Test R²: {r2_score(y_test, test_predictions):.4f}")
    print(f"Test MAE: {mean_absolute_error(y_test, test_predictions):.4f}")
    print(f"Test RMSE: {root_mean_squared_error(y_test, test_predictions):.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved pipeline: {MODEL_PATH}")

    return model


if __name__ == "__main__":
    main()
