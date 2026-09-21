# Student Exam Score Prediction

Machine Learning regression project that predicts a student's final exam score (`Exam_Score`) from study habits, previous academic results, and socio-familial information.

The project covers the complete workflow:

- Data loading and exploratory data analysis (EDA)
- Data cleaning and preprocessing
- Training multiple regression models
- Hyperparameter tuning
- Model evaluation and comparison
- Export of the final preprocessing + prediction pipeline
- Interactive prediction interface with Streamlit
- Optional Docker containerization and Docker Hub publication

---

## 1. Project Context

A higher-education institution wants an intelligent system capable of estimating the final exam score of each student.

The objectives are to:

- anticipate academic results;
- identify students who may need additional support;
- help the educational team make data-informed decisions;
- build a reproducible Machine Learning workflow;
- expose the final model through a simple Streamlit application.

> The prediction is an assistance tool. It should not be used as the sole basis for educational decisions about a student.

---

## 2. Machine Learning Problem

This is a **supervised regression** problem.

- **Target:** `Exam_Score`
- **Input:** student characteristics and study-related variables
- **Output:** predicted exam score

Models evaluated:

1. Linear Regression
2. Random Forest Regressor
3. XGBoost Regressor
4. Support Vector Regressor (SVR)

Evaluation metrics:

- RMSE
- MAE
- R²

---

## 3. Dataset

The dataset contains information related to:

- study habits;
- previous academic performance;
- parental and socio-familial context;
- school environment;
- access to resources;
- motivation and extracurricular activities.

The detailed description of every variable is available in:

```text
data/dictionnaire_des_donnees.md
```

Missing values are expected in:

- `Teacher_Quality`
- `Parental_Education_Level`
- `Distance_from_Home`

---

## 4. Recommended Project Structure

```text
student-exam-score-prediction/
│
├── data/
│   ├── raw/
│   │   └── student_performance.csv
│   ├── processed/
│   └── dictionnaire_des_donnees.md
│
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_model_experiments.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   └── load_data.py
│   ├── features/
│   │   └── preprocessing.py
│   ├── models/
│   │   ├── train.py
│   │   ├── tune.py
│   │   └── evaluate.py
│   └── utils/
│       └── config.py
│
├── models/
│   ├── final_pipeline.joblib
│   └── feature_metadata.json
│
├── reports/
│   ├── figures/
│   └── model_results.csv
│
├── app.py
├── requirements.txt
├── .gitignore
├── Dockerfile
└── README.md
```

---

## 5. Installation

### Clone the repository

```bash
git clone <repository-url>
cd student-exam-score-prediction
```

### Create a virtual environment

Using `venv`:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on Linux / WSL:

```bash
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

Example dependencies:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
joblib
streamlit
jupyter
```

---

## 6. Project Workflow

```text
RAW DATA
   │
   ▼
DATA UNDERSTANDING
   │
   ▼
EDA
   │
   ▼
TRAIN / TEST SPLIT
   │
   ▼
PREPROCESSING PIPELINE
   │
   ├── Missing-value imputation
   ├── Ordinal encoding
   ├── One-hot encoding
   └── Numeric scaling
   │
   ▼
BASELINE MODELS
   │
   ▼
HYPERPARAMETER TUNING
   │
   ▼
MODEL COMPARISON
   │
   ▼
FINAL PIPELINE
   │
   ▼
JOBLIB EXPORT
   │
   ▼
STREAMLIT APPLICATION
```

---

## 7. Data Analysis

The EDA includes:

- dataset shape;
- column names;
- data types;
- sample rows;
- descriptive statistics;
- categorical frequencies;
- missing values;
- duplicated rows;
- target distribution;
- `Hours_Studied` distribution;
- outlier analysis;
- correlation analysis.

Example checks:

```python
df.shape
df.head()
df.info()
df.describe()
df.describe(include="object")
df.isna().sum()
df.duplicated().sum()
```

---

## 8. Preprocessing

### Missing Values

Categorical missing values are imputed using their most frequent value.

Columns concerned:

```text
Teacher_Quality
Parental_Education_Level
Distance_from_Home
```

### Duplicates

Duplicated observations are identified and removed when they represent true duplicate records.

### Outliers

`Exam_Score` is explored with:

- boxplots;
- IQR;
- z-score.

If the project specification requires IQR filtering, the decision and number of removed observations must be documented.

> Outlier removal must be performed carefully because an unusual but valid exam score is not necessarily erroneous.

### Nominal Variables

One-hot encoding:

```text
Gender
School_Type
Extracurricular_Activities
Internet_Access
Learning_Disabilities
Peer_Influence
```

### Ordinal Variables

Ordinal encoding:

```text
Parental_Involvement
Access_to_Resources
Motivation_Level
Family_Income
Teacher_Quality
Parental_Education_Level
Distance_from_Home
```

The category order must be defined explicitly according to the data dictionary.

### Numerical Variables

Numerical variables used by models requiring scaling are standardized with `StandardScaler`.

All preprocessing must be fitted using the training data only.

---

## 9. Train / Test Split

The dataset is separated into:

- 80% training data
- 20% test data

Example:

```python
from sklearn.model_selection import train_test_split

X = df.drop(columns="Exam_Score")
y = df["Exam_Score"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
```

---

## 10. Scikit-learn Pipeline

The recommended architecture is:

```text
Raw student data
       │
       ▼
ColumnTransformer
 ├── numeric pipeline
 │    ├── imputation if required
 │    └── StandardScaler
 │
 ├── ordinal pipeline
 │    ├── SimpleImputer
 │    └── OrdinalEncoder
 │
 └── nominal pipeline
      ├── SimpleImputer
      └── OneHotEncoder
       │
       ▼
Regression Model
```

This complete object is trained, tuned, evaluated, and finally exported.

---

## 11. Baseline Models

Models are first trained with default or controlled baseline parameters:

```python
LinearRegression()
RandomForestRegressor(random_state=42)
XGBRegressor(random_state=42)
SVR()
```

For each model calculate:

```text
RMSE
MAE
R²
```

Results are stored in a comparison DataFrame and optionally exported to:

```text
reports/model_results.csv
```

---

## 12. Hyperparameter Tuning

`GridSearchCV` is applied only to the training set using 3-fold cross-validation.

Scoring:

```python
scoring="r2"
```

### Linear Regression

```text
fit_intercept
positive
```

### Random Forest

```text
n_estimators
max_depth
min_samples_split
```

### XGBoost

```text
n_estimators
learning_rate
max_depth
subsample
```

### SVR

```text
C
kernel
epsilon
```

The best estimator from each search is retained and evaluated once on the untouched test set.

---

## 13. Model Evaluation

Optimized models are compared using:

| Model             | RMSE | MAE |  R² | Residual Std | CV R² Std |
| ----------------- | ---: | --: | --: | -----------: | --------: |
| Linear Regression |    - |   - |   - |            - |         - |
| Random Forest     |    - |   - |   - |            - |         - |
| XGBoost           |    - |   - |   - |            - |         - |
| SVR               |    - |   - |   - |            - |         - |

Visualizations:

- predicted values vs real values;
- residual distribution;
- model metric comparison.

The final model is selected from empirical results and the choice is documented.

---

## 14. Exporting the Final Model

Export the **complete fitted pipeline**, including preprocessing and the final regressor.

```python
import joblib

joblib.dump(final_pipeline, "models/final_pipeline.joblib")
```

Metadata describing expected raw fields and allowed categories can also be saved.

Example:

```text
models/feature_metadata.json
```

---

## 15. Streamlit Application

Run the application with:

```bash
streamlit run app.py
```

The application should:

- collect student characteristics;
- validate user inputs;
- construct a one-row DataFrame using the raw feature names;
- load the saved pipeline;
- predict the exam score;
- display the estimated score;
- optionally display a configurable support/risk indicator.

Because preprocessing is part of the saved pipeline, `app.py` should not manually reproduce model transformations.

---

## 16. Reproducibility

Recommended practices:

- use `random_state=42` when supported;
- keep the raw dataset unchanged;
- fit preprocessing only on training data;
- keep the test set untouched during tuning;
- store dependencies in `requirements.txt`;
- document preprocessing decisions;
- version the project with Git;
- save model results and generated figures.

---

## 17. Jira Organization

Recommended Epics:

1. Data Understanding & Preparation
2. Baseline Model Training
3. Hyperparameter Optimization
4. Model Evaluation & Selection
5. Model Export & Streamlit Application
6. Documentation & Reproducibility
7. Docker Deployment (Bonus)

See `jira_backlog.md` for the detailed backlog.

---

## 18. Optional Docker Deployment

The Streamlit application can optionally be containerized.

Typical workflow:

```text
Create Dockerfile
      ↓
Build image
      ↓
Test locally
      ↓
Tag image
      ↓
Push image to Docker Hub
```

---

## 19. Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Joblib
- Streamlit
- Jupyter
- Git / GitHub
- Jira
- Docker (bonus)

---

## 20. Educational Purpose

This project is intended to demonstrate a complete supervised Machine Learning workflow, from raw tabular data to a deployable regression application.
