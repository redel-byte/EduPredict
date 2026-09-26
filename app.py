from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = Path(__file__).resolve().parent / "models" / "best_model.joblib"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.set_page_config(page_title="EduPredict", page_icon="🎓", layout="centered")
st.title("🎓 EduPredict")
st.write("Enter a student's information to estimate their exam score.")

if not MODEL_PATH.exists():
    st.error(
        "The best model has not been saved yet. Run `python sklearn_pipline.py` first."
    )
    st.stop()

model = load_model()

with st.form("student_information"):
    st.subheader("Student information")

    col1, col2 = st.columns(2)
    with col1:
        hours_studied = st.number_input(
            "Hours studied per week",
            min_value=0.0,
            max_value=60.0,
            value=20.0,
            step=1.0,
        )
        attendance = st.number_input(
            "Attendance (%)", min_value=0.0, max_value=100.0, value=80.0, step=1.0
        )
        sleep_hours = st.number_input(
            "Sleep per night (hours)",
            min_value=0.0,
            max_value=24.0,
            value=8.0,
            step=1.0,
        )
        previous_scores = st.number_input(
            "Previous exam score", min_value=0.0, max_value=100.0, value=70.0, step=1.0
        )
        tutoring_sessions = st.number_input(
            "Tutoring sessions per month", min_value=0, max_value=20, value=1
        )
        physical_activity = st.number_input(
            "Physical activity (hours per week)",
            min_value=0.0,
            max_value=24.0,
            value=3.0,
            step=1.0,
        )

    with col2:
        gender = st.selectbox("Gender", ["Female", "Male"])
        school_type = st.selectbox("School type", ["Public", "Private"])
        extracurricular_activities = st.selectbox(
            "Extracurricular activities", ["Yes", "No"]
        )
        internet_access = st.selectbox("Internet access", ["Yes", "No"])
        learning_disabilities = st.selectbox("Learning disabilities", ["No", "Yes"])
        peer_influence = st.selectbox(
            "Peer influence", ["Positive", "Neutral", "Negative"]
        )

    st.subheader("Learning environment")
    col3, col4 = st.columns(2)
    with col3:
        parental_involvement = st.selectbox(
            "Parental involvement", ["Low", "Medium", "High"]
        )
        access_to_resources = st.selectbox(
            "Access to resources", ["Low", "Medium", "High"]
        )
        motivation_level = st.selectbox("Motivation level", ["Low", "Medium", "High"])
        family_income = st.selectbox("Family income", ["Low", "Medium", "High"])
    with col4:
        teacher_quality = st.selectbox("Teacher quality", ["Low", "Medium", "High"])
        parental_education_level = st.selectbox(
            "Parental education level", ["High School", "College", "Postgraduate"]
        )
        distance_from_home = st.selectbox(
            "Distance from home", ["Near", "Moderate", "Far"]
        )

    submitted = st.form_submit_button(
        "Estimate exam score", type="primary", use_container_width=True
    )

if submitted:
    student = pd.DataFrame(
        [
            {
                "Hours_Studied": hours_studied,
                "Attendance": attendance,
                "Sleep_Hours": sleep_hours,
                "Previous_Scores": previous_scores,
                "Tutoring_Sessions": tutoring_sessions,
                "Physical_Activity": physical_activity,
                "Parental_Involvement": parental_involvement,
                "Access_to_Resources": access_to_resources,
                "Motivation_Level": motivation_level,
                "Family_Income": family_income,
                "Teacher_Quality": teacher_quality,
                "Parental_Education_Level": parental_education_level,
                "Distance_from_Home": distance_from_home,
                "Gender": gender,
                "School_Type": school_type,
                "Extracurricular_Activities": extracurricular_activities,
                "Internet_Access": internet_access,
                "Learning_Disabilities": learning_disabilities,
                "Peer_Influence": peer_influence,
            }
        ]
    )
    predicted_score = float(model.predict(student)[0])
    st.metric("Estimated exam score", f"{predicted_score:.1f} / 100")
