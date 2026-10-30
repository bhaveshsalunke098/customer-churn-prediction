from pathlib import Path
import pandas as pd

DATA_URL = (
    "https://raw.githubusercontent.com/IBM/"
    "telco-customer-churn-on-icp4d/master/data/"
    "Telco-Customer-Churn.csv"
)
DATA_PATH = Path("data/raw/Telco-Customer-Churn.csv")

def load_data(download_if_missing=True):
    """Load the IBM sample CSV, downloading it if it is not present."""
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        if not download_if_missing:
            raise FileNotFoundError(
                f"Dataset not found at {DATA_PATH}. Run the project or download it from IBM."
            )
        print("Downloading the IBM Telco Customer Churn dataset...")
        df = pd.read_csv(DATA_URL)
        df.to_csv(DATA_PATH, index=False)
        print(f"Saved dataset to {DATA_PATH}")
    else:
        df = pd.read_csv(DATA_PATH)
    return df

def clean_data(df):
    """Apply documented, minimal cleaning without changing the raw file."""
    df = df.copy()
    if "customerID" in df.columns:
        df = df.drop_duplicates(subset=["customerID"])
        df = df.drop(columns=["customerID"])

    # TotalCharges is stored as text in the source; blank values become NaN.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # In this dataset, blank TotalCharges are associated with zero-tenure customers.
    # Fill those values with 0 because no tenure means no accumulated charges yet.
    zero_tenure = df["tenure"].eq(0)
    df.loc[zero_tenure & df["TotalCharges"].isna(), "TotalCharges"] = 0

    # If any other missing values exist, the model pipeline imputes them.
    return df
