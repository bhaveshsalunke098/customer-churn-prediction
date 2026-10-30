from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_utils import load_data, clean_data

Path("reports").mkdir(exist_ok=True)
sns.set_theme()

df_raw = load_data()
print("\n--- Shape ---")
print(df_raw.shape)
print("\n--- First five rows ---")
print(df_raw.head())
print("\n--- Data types ---")
print(df_raw.dtypes)
print("\n--- Missing values ---")
print(df_raw.isna().sum().sort_values(ascending=False))
print("\n--- Duplicate customer IDs ---")
print(df_raw["customerID"].duplicated().sum())
print("\n--- Target distribution ---")
print(df_raw["Churn"].value_counts(dropna=False))
print("\n--- Target proportions (%) ---")
print((df_raw["Churn"].value_counts(normalize=True) * 100).round(2))

df = clean_data(df_raw)

# 1. Churn distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Churn")
plt.title("Customer Churn Distribution")
plt.tight_layout()
plt.savefig("reports/churn_distribution.png", dpi=150)
plt.close()

# 2. Churn by contract type
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Contract", hue="Churn")
plt.title("Churn by Contract Type")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("reports/churn_by_contract.png", dpi=150)
plt.close()

# 3. Tenure distribution by churn
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="tenure", hue="Churn", bins=25, stat="density",
             common_norm=False, element="step")
plt.title("Tenure Distribution by Churn")
plt.tight_layout()
plt.savefig("reports/tenure_by_churn.png", dpi=150)
plt.close()

print("\nCharts saved in reports/. Inspect them before writing business conclusions.")
