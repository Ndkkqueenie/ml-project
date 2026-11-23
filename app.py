from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC


DATA_PATH = Path(__file__).with_name("mushrooms.csv")
METRICS = ["Confusion matrix", "ROC curve", "Precision-recall curve"]
CLASS_LABELS = {"e": "Edible", "p": "Poisonous"}


def display_class_name(label):
    return CLASS_LABELS.get(str(label).strip().lower(), str(label))


def build_model(model_name, features):
    numeric_features = features.select_dtypes(include="number").columns.tolist()
    categorical_features = features.select_dtypes(exclude="number").columns.tolist()

    transformers = []
    if numeric_features:
        transformers.append(
            ("numeric", SimpleImputer(strategy="median"), numeric_features)
        )
    if categorical_features:
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(
            ("categorical", categorical_pipeline, categorical_features)
        )

    preprocessing = ColumnTransformer(transformers)
    classifiers = {
        "Logistic regression": LogisticRegression(max_iter=1000),
        "Random forest": RandomForestClassifier(
            n_estimators=200, random_state=42
        ),
        "Support vector machine": SVC(probability=True),
    }
    return Pipeline(
        [
            ("preprocessing", preprocessing),
            ("classifier", classifiers[model_name]),
        ]
    )


def train_model(data, target_column, model_name):
    frame = data.copy()
    for column in frame.select_dtypes(include=["object", "string"]).columns:
        frame[column] = frame[column].str.strip()
    frame = frame.replace("?", np.nan).dropna(subset=[target_column])

    features = frame.drop(columns=[target_column]).dropna(axis=1, how="all")
    target = frame[target_column]
    if features.empty:
        raise ValueError("The dataset must contain at least one usable feature.")
    if target.nunique() != 2:
        raise ValueError("Choose a target column with exactly two classes.")
    if target.value_counts().min() < 2:
        raise ValueError(
            "Each target class needs at least two rows for a stratified split."
        )

    test_size = max(0.3, 2 / len(target))
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=42,
        stratify=target,
    )
    model = build_model(model_name, X_train)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    positive_class = model.named_steps["classifier"].classes_[1]

    scores = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, pos_label=positive_class, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, pos_label=positive_class, zero_division=0
        ),
    }
    return model, X_test, y_test, scores, positive_class, len(X_train)


def main():
    st.title("Binary Classification Web App")
    st.markdown(
        "Explore whether mushrooms are edible or poisonous. Load the bundled "
        "dataset or upload a CSV, choose a binary target and classifier, then "
        "review its performance on a held-out test set."
    )
    st.sidebar.title("Dataset and model")

    source = st.sidebar.radio(
        "Data source", ["Bundled mushrooms.csv", "Upload a CSV"]
    )
    if source == "Upload a CSV":
        uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is None:
            st.info("Upload a CSV file to get started.")
            return
        try:
            data = pd.read_csv(uploaded_file)
        except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as error:
            st.error(f"Could not read the uploaded CSV: {error}")
            return
    else:
        try:
            data = pd.read_csv(DATA_PATH)
        except FileNotFoundError:
            st.error(f"Bundled dataset not found: {DATA_PATH}")
            return

    if data.empty or len(data.columns) < 2:
        st.error("The dataset must contain rows and at least two columns.")
        return

    default_target = next(
        (name for name in ("class", "type") if name in data.columns),
        data.columns[0],
    )
    target_column = st.sidebar.selectbox(
        "Target column",
        data.columns,
        index=data.columns.get_loc(default_target),
    )
    model_name = st.sidebar.selectbox(
        "Classifier",
        [
            "Logistic regression",
            "Random forest",
            "Support vector machine",
        ],
    )
    selected_metrics = st.sidebar.multiselect(
        "Metrics to display", METRICS, default=METRICS
    )

    with st.expander("Preview dataset"):
        st.write(data.head(20))
        st.caption(f"{len(data):,} rows · {len(data.columns)} columns")

    try:
        model, X_test, y_test, scores, positive_class, train_rows = train_model(
            data, target_column, model_name
        )
    except ValueError as error:
        st.error(f"Could not train the model: {error}")
        return

    class_names = [
        display_class_name(label)
        for label in model.named_steps["classifier"].classes_
    ]
    st.subheader("Test-set performance")
    st.caption(
        f"Trained on {train_rows:,} rows and evaluated on {len(y_test):,} rows. "
        f"Precision and recall treat {display_class_name(positive_class)} "
        "as the positive class."
    )
    score_columns = st.columns(len(scores))
    for column, (name, value) in zip(score_columns, scores.items()):
        column.metric(name, f"{value:.3f}")

    if "Confusion matrix" in selected_metrics:
        st.subheader("Confusion matrix")
        display = ConfusionMatrixDisplay.from_estimator(
            model, X_test, y_test, display_labels=class_names
        )
        st.pyplot(display.figure_)

    if "ROC curve" in selected_metrics:
        st.subheader("ROC curve")
        st.caption(f"Positive class: {display_class_name(positive_class)}")
        display = RocCurveDisplay.from_estimator(model, X_test, y_test)
        st.pyplot(display.figure_)

    if "Precision-recall curve" in selected_metrics:
        st.subheader("Precision-recall curve")
        st.caption(f"Positive class: {display_class_name(positive_class)}")
        display = PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
        st.pyplot(display.figure_)


if __name__ == "__main__":
    main()
