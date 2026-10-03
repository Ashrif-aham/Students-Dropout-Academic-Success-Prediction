import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Student Dropout Prediction",
    page_icon="🎓",
    layout="wide"
)


# =====================================================
# 2. CUSTOM UI DESIGN
# =====================================================

st.markdown("""
<style>

/* Main application background */

.stApp {
    background-color: #F5F7FB;
    color: #1E293B;
}

/* Main content text */

.stApp p,
.stApp label,
.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4,
.stApp span,
.stApp div[data-testid="stMarkdownContainer"] {
    color: #1E293B;
}

/* Main heading */

h1 {
    color: #172554 !important;
    font-weight: 700 !important;
}

/* Metric cards */

div[data-testid="stMetric"] {
    background-color: #FFFFFF;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

/* Metric values */

div[data-testid="stMetricValue"] {
    color: #2563EB !important;
    font-weight: bold;
}

/* Sidebar */

[data-testid="stSidebar"] {
    background-color: #172554;
}

/* Sidebar text */

[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

/* ============================= */
/* ALL STREAMLIT BUTTONS */
/* ============================= */

/* Normal buttons */
.stButton > button,
button[kind="secondary"],
button[kind="primary"] {
    background-color: #2563EB !important;
    color: #FFFFFF !important;
    border: 1px solid #2563EB !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    min-height: 45px !important;
}

/* Button text and icons */
.stButton > button *,
button[kind="secondary"] *,
button[kind="primary"] * {
    color: #FFFFFF !important;
}

/* Hover */
.stButton > button:hover,
button[kind="secondary"]:hover,
button[kind="primary"]:hover {
    background-color: #1D4ED8 !important;
    border-color: #1D4ED8 !important;
    color: #FFFFFF !important;
}

/* ============================= */
/* FILE UPLOADER BUTTON */
/* ============================= */

[data-testid="stFileUploader"] button {
    background-color: #2563EB !important;
    color: #FFFFFF !important;
    border: 1px solid #2563EB !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

/* File uploader button text + icon */
[data-testid="stFileUploader"] button * {
    color: #FFFFFF !important;
}

/* File uploader hover */
[data-testid="stFileUploader"] button:hover {
    background-color: #1D4ED8 !important;
    border-color: #1D4ED8 !important;
}

/* ============================= */
/* DOWNLOAD BUTTON */
/* ============================= */

[data-testid="stDownloadButton"] button {
    background-color: #2563EB !important;
    color: #FFFFFF !important;
    border: 1px solid #2563EB !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

[data-testid="stDownloadButton"] button:hover {
    background-color: #1D4ED8 !important;
    border-color: #1D4ED8 !important;
}

[data-testid="stDownloadButton"] button * {
    color: #FFFFFF !important;
}

/* Dataframes */

[data-testid="stDataFrame"] {
    background-color: white;
    border-radius: 10px;
}

/* Tabs */

button[data-baseweb="tab"] {
    font-weight: 600;
}

/* ============================= */
/* ALL STREAMLIT BUTTONS */
/* ============================= */

/* Normal buttons */
.stButton > button,
button[kind="secondary"],
button[kind="primary"] {
    background-color: #2563EB !important;
    color: #FFFFFF !important;
    border: 1px solid #2563EB !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    min-height: 45px !important;
}

/* Button text and icons */
.stButton > button *,
button[kind="secondary"] *,
button[kind="primary"] * {
    color: #FFFFFF !important;
}

/* Hover */
.stButton > button:hover,
button[kind="secondary"]:hover,
button[kind="primary"]:hover {
    background-color: #1D4ED8 !important;
    border-color: #1D4ED8 !important;
    color: #FFFFFF !important;
}

/* ============================= */
/* FILE UPLOADER BUTTON */
/* ============================= */

[data-testid="stFileUploader"] button {
    background-color: #2563EB !important;
    color: #FFFFFF !important;
    border: 1px solid #2563EB !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

/* File uploader button text + icon */
[data-testid="stFileUploader"] button * {
    color: #FFFFFF !important;
}

/* File uploader hover */
[data-testid="stFileUploader"] button:hover {
    background-color: #1D4ED8 !important;
    border-color: #1D4ED8 !important;
}

/* ============================= */
/* DOWNLOAD BUTTON */
/* ============================= */

[data-testid="stDownloadButton"] button {
    background-color: #2563EB !important;
    color: #FFFFFF !important;
    border: 1px solid #2563EB !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

[data-testid="stDownloadButton"] button:hover {
    background-color: #1D4ED8 !important;
    border-color: #1D4ED8 !important;
}

[data-testid="stDownloadButton"] button * {
    color: #FFFFFF !important;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# 3. FILE PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "student_dropout_model.pkl"

##DATA_PATH = BASE_DIR / "Student dropout and academic success.csv"

# Automatically find the dataset CSV file

csv_files = list(BASE_DIR.glob("*.csv"))

if not csv_files:
    st.error(
        f"No CSV file found in: {BASE_DIR}"
    )
    st.stop()

DATA_PATH = csv_files[0]

st.write("Dataset found:", DATA_PATH.name)

# =====================================================
# 4. LOAD YOUR SAVED BEST MODEL
# =====================================================

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)

    return model


@st.cache_data
def load_data():

    data = pd.read_csv(
        DATA_PATH,
        sep=";"
    )

    data.columns = (
        data.columns
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    return data


try:

    best_model = load_model()

    df = load_data()

except FileNotFoundError as e:

    st.error(
        f"File not found: {e.filename}. "
        "Check that the model and dataset "
        "are inside the Student_Dropout_App folder."
    )

    st.stop()


# =====================================================
# 5. PREPARE DATA
# =====================================================

df = df.drop_duplicates()

if "Target" not in df.columns:

    st.error("Target column not found in dataset.")

    st.stop()


X = df.drop(columns=["Target"])

y = df["Target"]


# Use the exact feature names from training

feature_names = list(X.columns)


# =====================================================
# 6. SIDEBAR
# =====================================================

with st.sidebar:

    st.title("🎓 Student Success AI")

    st.write("Student Dropout Prediction System")

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Predict Student",
            "Batch Prediction",
            "About"
        ]
    )

    st.markdown("---")

    st.caption("Machine Learning Project")


# =====================================================
# 7. DASHBOARD
# =====================================================

if page == "Dashboard":

    st.title("📊 Student Analytics Dashboard")

    st.write(
        "Analyze student academic outcomes "
        "and dropout patterns."
    )

    total_students = len(df)

    dropout_count = (y == "Dropout").sum()

    graduate_count = (y == "Graduate").sum()

    enrolled_count = (y == "Enrolled").sum()

    dropout_rate = (
        dropout_count / total_students * 100
    )

    # KPI cards

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Students",
        f"{total_students:,}"
    )

    c2.metric(
        "Dropout Students",
        f"{dropout_count:,}"
    )

    c3.metric(
        "Graduates",
        f"{graduate_count:,}"
    )

    c4.metric(
        "Dropout Rate",
        f"{dropout_rate:.2f}%"
    )

    st.markdown("---")

    # Target distribution

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Student Outcome Distribution")

        target_counts = y.value_counts()

        fig, ax = plt.subplots(figsize=(7, 5))

        ax.pie(
            target_counts.values,
            labels=target_counts.index,
            autopct="%1.1f%%",
            startangle=90
        )

        ax.set_title("Student Outcomes")

        st.pyplot(fig)

        plt.close(fig)

    with col2:

        st.subheader("Age Distribution")

        fig, ax = plt.subplots(figsize=(7, 5))

        for target in y.unique():

            ages = df.loc[
                y == target,
                "Age at enrollment"
            ]

            ax.hist(
                ages,
                bins=20,
                alpha=0.5,
                label=target
            )

        ax.set_xlabel("Age at Enrollment")

        ax.set_ylabel("Number of Students")

        ax.legend()

        st.pyplot(fig)

        plt.close(fig)

    # Academic performance

    st.markdown("---")

    st.subheader("Academic Performance")

    academic_cols = [
        "Curricular units 1st sem (approved)",
        "Curricular units 2nd sem (approved)",
        "Curricular units 1st sem (grade)",
        "Curricular units 2nd sem (grade)"
    ]

    academic_cols = [
        c for c in academic_cols
        if c in df.columns
    ]

    academic_summary = (
        df.groupby("Target")[academic_cols]
        .mean()
        .round(2)
    )

    st.dataframe(
        academic_summary,
        use_container_width=True
    )

    st.bar_chart(academic_summary)

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


# =====================================================
# 8. INDIVIDUAL STUDENT PREDICTION
# =====================================================

elif page == "Predict Student":

    st.title("🔍 Predict Student Outcome")

    st.write(
        "Enter student details to predict "
        "the student's academic outcome."
    )

    st.info(
        "Use the same feature coding and units "
        "as your training dataset."
    )

    input_data = {}

    with st.form("prediction_form"):

        st.subheader("Student Information")

        columns = st.columns(3)

        for i, col in enumerate(feature_names):

            with columns[i % 3]:

                if col in [
                    "Marital status",
                    "Application mode",
                    "Application order",
                    "Course",
                    "Daytime/evening attendance",
                    "Previous qualification",
                    "Nacionality",
                    "Mother's qualification",
                    "Father's qualification",
                    "Mother's occupation",
                    "Father's occupation",
                    "Displaced",
                    "Educational special needs",
                    "Debtor",
                    "Tuition fees up to date",
                    "Gender",
                    "Scholarship holder",
                    "International"
                ]:

                    options = sorted(
                        X[col].dropna().unique().tolist()
                    )

                    if options:

                        input_data[col] = st.selectbox(
                            col,
                            options=options,
                            format_func=lambda x: str(x)
                        )

                    else:

                        input_data[col] = 0

                else:

                    values = pd.to_numeric(
                        X[col],
                        errors="coerce"
                    )

                    median = values.median()

                    min_val = values.min()

                    max_val = values.max()

                    if pd.isna(median):
                        median = 0.0

                    if pd.isna(min_val):
                        min_val = 0.0

                    if pd.isna(max_val):
                        max_val = 100.0

                    padding = max(
                        (max_val - min_val) * 0.1,
                        1
                    )

                    input_data[col] = st.number_input(
                        col,
                        min_value=float(min_val - padding),
                        max_value=float(max_val + padding),
                        value=float(median),
                        step=1.0
                    )

        st.markdown("---")

        submit = st.form_submit_button(
            "🚀 Predict Student Outcome",
            type="primary",
            use_container_width=True
        )

    if submit:

        new_student = pd.DataFrame(
            [input_data],
            columns=feature_names
        )

        # Prediction using your saved best_model

        prediction = best_model.predict(
            new_student
        )[0]

        probabilities = best_model.predict_proba(
            new_student
        )[0]

        classes = best_model.classes_

        st.markdown("---")

        st.subheader("Prediction Result")

        if prediction == "Dropout":

            st.error(
                "⚠️ Predicted Outcome: Dropout"
            )

        elif prediction == "Graduate":

            st.success(
                "🎓 Predicted Outcome: Graduate"
            )

        else:

            st.info(
                "📚 Predicted Outcome: Enrolled"
            )

        # Probability cards

        st.subheader("Prediction Probabilities")

        prob_cols = st.columns(len(classes))

        for i, class_name in enumerate(classes):

            prob_cols[i].metric(
                class_name,
                f"{probabilities[i] * 100:.2f}%"
            )

        # Probability chart

        probability_df = pd.DataFrame({
            "Outcome": classes,
            "Probability (%)": probabilities * 100
        })

        st.bar_chart(
            probability_df.set_index("Outcome")
        )

        # Download individual prediction

        result = new_student.copy()

        result["Predicted Target"] = prediction

        for i, class_name in enumerate(classes):

            result[
                f"Probability {class_name} (%)"
            ] = round(
                probabilities[i] * 100,
                2
            )

        csv = result.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "📥 Download Prediction",
            data=csv,
            file_name="student_prediction.csv",
            mime="text/csv"
        )

        with st.expander("View Student Details"):

            st.dataframe(
                new_student,
                use_container_width=True
            )


# =====================================================
# 9. BATCH PREDICTION
# =====================================================

elif page == "Batch Prediction":

    st.title("📁 Batch Student Prediction")

    st.write(
        "Upload a CSV file containing multiple "
        "students and predict their outcomes."
    )

    st.info(
        "Upload a CSV with all training feature "
        "columns, excluding Target."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            batch_df = pd.read_csv(
                uploaded_file,
                sep=None,
                engine="python"
            )

            batch_df.columns = (
                batch_df.columns
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
            )

            # Remove target if provided

            if "Target" in batch_df.columns:

                batch_df = batch_df.drop(
                    columns=["Target"]
                )

            missing_columns = [
                c for c in feature_names
                if c not in batch_df.columns
            ]

            if missing_columns:

                st.error(
                    "Missing columns: "
                    + ", ".join(missing_columns)
                )

            else:

                # Keep correct feature order

                batch_df = batch_df[feature_names]

                st.success(
                    f"{len(batch_df)} students uploaded."
                )

                st.dataframe(
                    batch_df.head(10),
                    use_container_width=True
                )

                if st.button(
                    "Predict All Students",
                    type="primary"
                ):

                    predictions = best_model.predict(
                        batch_df
                    )

                    probabilities = (
                        best_model.predict_proba(
                            batch_df
                        )
                    )

                    result = batch_df.copy()

                    result["Predicted Target"] = (
                        predictions
                    )

                    for i, class_name in enumerate(
                        best_model.classes_
                    ):

                        result[
                            f"Probability {class_name} (%)"
                        ] = (
                            probabilities[:, i] * 100
                        ).round(2)

                    st.subheader("Prediction Results")

                    st.dataframe(
                        result,
                        use_container_width=True
                    )

                    st.subheader("Predicted Outcome Summary")

                    st.bar_chart(
                        result[
                            "Predicted Target"
                        ].value_counts()
                    )

                    csv = result.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        "📥 Download All Predictions",
                        data=csv,
                        file_name="student_predictions.csv",
                        mime="text/csv"
                    )

        except Exception as e:

            st.error(f"Error: {e}")


# =====================================================
# 10. ABOUT PROJECT
# =====================================================

elif page == "About":

    st.title("ℹ️ About Project")

    st.write("""
    Student Dropout Prediction is a machine learning
    application that predicts student academic
    outcomes based on historical student data.

    The model predicts three categories:

    1. Dropout
    2. Enrolled
    3. Graduate

    The application includes data visualization,
    individual student prediction, batch prediction,
    and downloadable prediction reports.
    """)

    st.subheader("Machine Learning Model")

    st.write(
        "Loaded model: "
        + type(
            best_model.named_steps["model"]
        ).__name__
        if hasattr(best_model, "named_steps")
        else type(best_model).__name__
    )

    st.write(
        f"Dataset records: {len(df):,}"
    )

    st.caption(
        "Predictions are estimates based on historical "
        "data and should not be treated as guaranteed "
        "student outcomes."
    )