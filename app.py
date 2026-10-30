from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

MODEL_PATH = Path("models/churn_pipeline.joblib")
DATA_PATH = Path("data/raw/Telco-Customer-Churn.csv")
CLASSIFICATION_THRESHOLD = 0.50

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      .block-container {max-width: 1050px; padding-top: 2rem; padding-bottom: 3rem;}
      .hero {padding: 1.25rem 1.4rem; border: 1px solid rgba(128,128,128,.28);
             border-radius: 16px; margin-bottom: 1.2rem;}
      .hero p {margin-bottom: .25rem; opacity: .82;}
      .eyebrow {text-transform: uppercase; letter-spacing: .12em; font-size: .75rem;
                font-weight: 700; opacity: .7;}
      .risk-low {padding: 1rem; border-radius: 12px; background: rgba(22,163,74,.12);
                 border: 1px solid rgba(22,163,74,.35);}
      .risk-moderate {padding: 1rem; border-radius: 12px; background: rgba(234,179,8,.12);
                      border: 1px solid rgba(234,179,8,.4);}
      .risk-high {padding: 1rem; border-radius: 12px; background: rgba(220,38,38,.10);
                  border: 1px solid rgba(220,38,38,.35);}
      div[data-testid="stForm"] {border-radius: 14px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Machine Learning · Explainable AI</div>
      <h1 style="margin:.35rem 0 .5rem 0;">📊 Customer Churn Prediction</h1>
      <p>Estimate churn probability and inspect SHAP explanations for a fictional telecom customer
      using a model trained on IBM's sample Telco Customer Churn dataset.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists():
    st.error("Model file not found. From this project folder, run `python train.py` first.")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as exc:
    st.error(
        "The saved model could not be loaded. Retrain and run the app with the same Python "
        "environment: `python train.py`, then `python -m streamlit run app.py`."
    )
    st.exception(exc)
    st.stop()



with st.form("customer_form"):
    st.subheader("1 · Customer profile")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior citizen?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        partner = st.selectbox("Has partner?", ["Yes", "No"])
        dependents = st.selectbox("Has dependents?", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless billing?", ["Yes", "No"])
    with col2:
        monthly = st.number_input("Monthly charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=1.0)
        total = st.number_input("Total charges ($)", min_value=0.0, max_value=10000.0, value=840.0, step=10.0)
        payment = st.selectbox("Payment method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        phone = st.selectbox("Phone service?", ["Yes", "No"])
        multiple = st.selectbox("Multiple lines", ["No phone service", "No", "Yes"])
        internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])

    st.subheader("2 · Internet services")
    col3, col4 = st.columns(2)
    with col3:
        security = st.selectbox("Online security", ["No internet service", "No", "Yes"])
        backup = st.selectbox("Online backup", ["No internet service", "No", "Yes"])
        protection = st.selectbox("Device protection", ["No internet service", "No", "Yes"])
        support = st.selectbox("Tech support", ["No internet service", "No", "Yes"])
    with col4:
        tv = st.selectbox("Streaming TV", ["No internet service", "No", "Yes"])
        movies = st.selectbox("Streaming movies", ["No internet service", "No", "Yes"])

    submitted = st.form_submit_button("Estimate churn probability", type="primary", use_container_width=True)

if submitted:
    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": 0.0 if tenure == 0 else total,
    }])
    st.session_state["last_customer"] = customer

# Keep the result visible when a user interacts with the explanation section.
if "last_customer" in st.session_state:
    customer = st.session_state["last_customer"]
    probability = float(model.predict_proba(customer)[0, 1])
    predicted_churn = probability >= CLASSIFICATION_THRESHOLD

    if probability < 0.20:
        band, css_class = "Low estimated probability", "risk-low"
        band_note = "The model estimates a relatively low probability of churn for this profile."
    elif probability < CLASSIFICATION_THRESHOLD:
        band, css_class = "Moderate estimated probability", "risk-moderate"
        band_note = "The estimate is above the illustrative low band, but below the demo classification threshold."
    else:
        band, css_class = "High estimated probability", "risk-high"
        band_note = "The estimate is at or above the demo classification threshold."

    st.divider()
    st.subheader("Prediction result")
    st.markdown(
        f'<div class="{css_class}"><div class="eyebrow">Illustrative probability band</div>'
        f'<h3 style="margin:.35rem 0;">{band}</h3><p style="margin:0;">{band_note}</p></div>',
        unsafe_allow_html=True,
    )
    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        st.metric("Estimated churn probability", f"{probability:.1%}")
        st.progress(max(0, min(100, int(round(probability * 100)))))
    with metric_col2:
        st.metric("Demo classification (50% threshold)", "Churn" if predicted_churn else "No churn")
        st.caption("This label uses a fixed 50% cutoff for demonstration only.")

    st.caption(
        "The probability-band boundaries (20% and 50%) are illustrative UI labels, not validated "
        "business risk categories. The 50% threshold is a demonstration choice. A real deployment "
        "should select thresholds using validation data, calibration checks, and business costs."
    )

    st.divider()
    st.subheader("Why did the model make this prediction?")
    st.write("SHAP estimates how each encoded feature contributes to the model's output. These are model explanations, not proof of cause and effect.")

    try:
        import shap
        from src.data_utils import load_data, clean_data

        if not hasattr(model, "named_steps") or "preprocessor" not in model.named_steps or "model" not in model.named_steps:
            raise ValueError("The saved model is not the expected scikit-learn preprocessing + classifier pipeline.")

        preprocessor = model.named_steps["preprocessor"]
        classifier = model.named_steps["model"]
        feature_names = preprocessor.get_feature_names_out()
        transformed_customer = preprocessor.transform(customer)
        if hasattr(transformed_customer, "toarray"):
            transformed_customer = transformed_customer.toarray()
        transformed_customer = np.asarray(transformed_customer)

        # This implementation uses LinearExplainer for the Logistic Regression model trained by train.py.
        # Its SHAP values are contributions in the model's log-odds output space.
        if classifier.__class__.__name__ != "LogisticRegression":
            st.warning("The current SHAP explanation code is configured for the Logistic Regression model. Retrain/select Logistic Regression or adapt the explainer for the selected estimator.")
        else:
            @st.cache_resource(show_spinner="Preparing SHAP explainer…")
            def make_shap_explainer():
                data = clean_data(load_data())
                if "Churn" in data.columns:
                    data = data.drop(columns=["Churn"])
                # Use a fixed, reproducible background sample to keep the app responsive.
                background_raw = data.sample(n=min(150, len(data)), random_state=42)
                background_transformed = preprocessor.transform(background_raw)
                if hasattr(background_transformed, "toarray"):
                    background_transformed = background_transformed.toarray()
                background_transformed = np.asarray(background_transformed)
                explainer = shap.LinearExplainer(classifier, background_transformed)
                return explainer, background_transformed, feature_names

            with st.spinner("Calculating SHAP explanations…"):
                explainer, background_transformed, feature_names = make_shap_explainer()
                local_exp = explainer(transformed_customer)

            values = np.asarray(local_exp.values).reshape(-1)
            display_names = [str(name).replace("numeric__", "").replace("categorical__", "") for name in feature_names]
            local_df = pd.DataFrame({
                "Feature": display_names,
                "Encoded value": transformed_customer[0],
                "SHAP value (log-odds)": values,
                "Absolute impact": np.abs(values),
            }).sort_values("Absolute impact", ascending=False)

            with st.expander("Individual customer explanation", expanded=True):
                st.caption("Positive SHAP values push the model output toward churn; negative values push it away from churn. Values are in log-odds space, not percentage-point changes in probability.")
                top_local = local_df.head(12).sort_values("SHAP value (log-odds)")
                fig, ax = plt.subplots(figsize=(9, max(4, 0.36 * len(top_local))))
                bar_colors = ["#D9534F" if value > 0 else "#238B57" for value in top_local["SHAP value (log-odds)"]]
                ax.barh(top_local["Feature"], top_local["SHAP value (log-odds)"], color=bar_colors)
                ax.axvline(0, linewidth=1, color="gray")
                ax.set_xlabel("SHAP value (log-odds contribution)")
                ax.set_ylabel("")
                ax.set_title("Features with the strongest local contributions")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                st.dataframe(local_df[["Feature", "Encoded value", "SHAP value (log-odds)"]].head(12), use_container_width=True, hide_index=True)

            with st.expander("Global feature importance", expanded=False):
                data = clean_data(load_data())
                if "Churn" in data.columns:
                    data = data.drop(columns=["Churn"])
                global_raw = data.sample(n=min(200, len(data)), random_state=123)
                global_transformed = preprocessor.transform(global_raw)
                if hasattr(global_transformed, "toarray"):
                    global_transformed = global_transformed.toarray()
                global_transformed = np.asarray(global_transformed)
                global_exp = explainer(global_transformed)
                global_values = np.asarray(global_exp.values)
                if global_values.ndim == 3:
                    global_values = global_values[:, :, -1]
                importance = np.mean(np.abs(global_values), axis=0)
                global_df = pd.DataFrame({"Feature": display_names, "Mean absolute SHAP value": importance})
                global_df = global_df.sort_values("Mean absolute SHAP value", ascending=False).head(15).sort_values("Mean absolute SHAP value")
                fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(global_df))))
                ax.barh(global_df["Feature"], global_df["Mean absolute SHAP value"], color="#4472C4")
                ax.set_xlabel("Mean absolute SHAP value (log-odds)")
                ax.set_ylabel("")
                ax.set_title("Global feature importance (sample of up to 200 customers)")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                st.dataframe(global_df.sort_values("Mean absolute SHAP value", ascending=False), use_container_width=True, hide_index=True)
                st.caption("Global importance is calculated on a reproducible sample of up to 200 rows from the IBM sample dataset; it may vary if the dataset or model changes.")

    except ImportError:
        st.warning("SHAP is not installed in this Python environment. In the project terminal, run `python -m pip install shap`, then restart Streamlit.")
    except Exception as exc:
        st.warning("The prediction worked, but SHAP explanations could not be generated. Check the terminal for details.")
        st.exception(exc)
