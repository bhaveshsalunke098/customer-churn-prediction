# Customer Churn Prediction

An end-to-end machine learning project that analyzes customer churn using IBM's Telco Customer Churn dataset. The project compares classification models, evaluates performance using multiple metrics, and provides an interactive Streamlit application with SHAP-based explainability.

## Project Overview

Customer churn occurs when a customer stops using a company's services. Identifying customers with a higher estimated probability of churn can help businesses understand customer behavior and explore potential retention strategies.

This project uses customer demographics, service subscriptions, contract details, tenure, and billing information to classify customers according to their churn status.

**Key objectives:**
- Perform exploratory data analysis (EDA) and data quality checks.
- Build and compare multiple machine learning classification models.
- Evaluate performance using classification and ranking metrics.
- Develop an interactive application for customer-level predictions.
- Explain model predictions using SHAP.

## Dataset

**IBM Telco Customer Churn Dataset**

- Official IBM repository: https://github.com/IBM/telco-customer-churn-on-icp4d
- Direct CSV: https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
- Alternative source: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

The dataset contains 7,043 customer records and 21 columns, including the target variable `Churn`.

The raw CSV is not included in this repository. The project downloads it automatically when the data-loading script runs, provided the source is accessible and the local dataset is not already present.

### Data Preprocessing

- Converts `TotalCharges` from text to numeric values.
- Handles blank `TotalCharges` values associated with zero-tenure customers.
- Separates numerical and categorical features.
- Applies preprocessing through a Scikit-learn pipeline.
- Uses one-hot encoding for categorical variables.

## Technologies Used

- **Programming:** Python
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib
- **Machine Learning:** Scikit-learn
- **Explainable AI:** SHAP
- **Web Application:** Streamlit
- **Model Persistence:** Joblib
- **Version Control:** Git and GitHub

## Machine Learning Workflow

1. Load and validate the dataset.
2. Perform exploratory data analysis and visualize churn patterns.
3. Preprocess numerical and categorical features.
4. Compare Logistic Regression, Random Forest, and HistGradientBoosting.
5. Select a candidate model using validation ROC-AUC.
6. Refit the selected model on the combined training and validation data.
7. Evaluate the final model on the held-out test set.
8. Save the fitted pipeline and evaluation outputs.
9. Generate interactive predictions and SHAP explanations in Streamlit.

## Model Evaluation

The selected Logistic Regression model achieved the following reported results on the held-out test set:

| Metric | Test Result |
|---|---:|
| Accuracy | 73.81% |
| Precision | 50.43% |
| Recall | 78.34% |
| F1-score | 61.36% |
| ROC-AUC | 84.16% |
| PR-AUC | 63.27% |

These metrics reflect the reported model run. Results may vary if the dataset, dependencies, data split, or training configuration changes.

**Interpretation:** The model identified approximately 78% of actual churn customers in the test set. Its precision indicates that some customers predicted as churners did not belong to the actual churn class. Therefore, accuracy alone is insufficient to assess the model's usefulness.

The classification threshold is configurable in the application. The current 50% threshold is a demonstration setting, not a business-validated decision threshold.

## Streamlit Application

The application allows users to enter a customer profile and estimate the probability of churn.

### Features

- Interactive customer profile form.
- Estimated churn probability.
- Illustrative probability bands.
- Classification result using a 50% threshold.
- Individual SHAP explanations for a selected customer.
- Global feature importance based on a sample of customers.
- Visualizations showing the direction and magnitude of encoded feature contributions.

### Understanding SHAP

SHAP estimates how individual features contribute to the model's output.

- **Positive SHAP values** push the Logistic Regression output toward the churn class.
- **Negative SHAP values** push the output away from the churn class.
- SHAP values are expressed in log-odds space, not as direct percentage-point changes in probability.
- Global feature importance summarizes the average absolute SHAP contribution across the sampled customers.

These explanations describe model behavior and do not establish causal relationships.

## Getting Started

### Prerequisites

- Python 3.10 or newer recommended.
- Git (optional, for cloning the repository).
- Internet access for the initial dataset download.

### 1. Clone the Repository

```bash
git clone https://github.com/bhaveshsalunke098/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Create a Virtual Environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run Exploratory Data Analysis

```bash
python eda.py
```

The script loads or downloads the dataset and generates exploratory outputs in the configured reports directory.

### 5. Train and Evaluate the Models

```bash
python train.py
```

This step generates the fitted model and evaluation outputs. The Streamlit application expects the saved model at:

```text
models/churn_pipeline.joblib
```

### 6. Launch the Application

```bash
python -m streamlit run app.py
```

Open the local URL printed in the terminal by Streamlit.

**Note:** If the trained model artifact is not included in the repository, run the training step before launching the application.

## Repository Structure

```text
customer-churn-prediction/
├── app.py
├── eda.py
├── train.py
├── requirements.txt
├── README.md
├── .gitignore
├── src/
│   ├── __init__.py
│   └── data_utils.py
├── models/
│   ├── churn_pipeline.joblib
│   └── model_info.json
├── reports/
│   ├── churn_by_contract.png
│   ├── churn_distribution.png
│   ├── tenure_by_churn.png
│   ├── validation_model_comparison.csv
│   └── test_predictions.csv
└── data/
    └── raw/
        └── Telco-Customer-Churn.csv  # Downloaded locally; not committed
```

Generated files may vary depending on the training and analysis scripts. The raw dataset, virtual environment, cache files, secrets, and unnecessary temporary files should not be committed.

## Limitations

- This is a classification demonstration using a sample dataset, not a time-ordered monthly churn forecasting system.
- Predictions are estimates and depend on the training data and model.
- SHAP explanations describe patterns learned by the model; they do not prove cause and effect.
- Probability bands and the 50% classification threshold are illustrative rather than validated business policies.
- Real-world deployment would require further validation, probability calibration, monitoring, privacy safeguards, and business-specific threshold selection.

## Future Improvements

- Hyperparameter tuning and cross-validation.
- Probability calibration and business-specific threshold selection.
- Automated testing and continuous integration.
- Batch prediction functionality.
- SQL integration for storing prediction results.
- Deployment to a cloud hosting platform.

## Responsible Use

This project is intended for educational and portfolio demonstration purposes. Model outputs should not be treated as certainty or used as the sole basis for consequential decisions about real customers.

## Author

**Bhavesh Salunke**

- GitHub: https://github.com/bhaveshsalunke098
- LinkedIn: https://linkedin.com/in/bhavesh-salunke-7a4163283
- Portfolio: https://techbhavesh.netlify.app
