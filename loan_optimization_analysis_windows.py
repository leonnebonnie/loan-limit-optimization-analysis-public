"""
Windows-friendly smoke-test runner for Loan Limit Optimization Analysis.
This is a safe, reduced-size run of the original `loan_optimization_analysis.py` suitable
for verifying environment, imports, and basic logic.

It uses relative input/output paths and a small synthetic dataset if the real Excel
is not present in the workspace.
"""
import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Smaller imports used in the main script to validate availability
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
except Exception as e:
    print("Missing optional plotting or ML packages or other import error:", e)
    print("Install the packages in requirements.txt and re-run.")
    sys.exit(1)

WORKDIR = os.path.abspath(os.path.dirname(__file__))
INPUT_XLSX = os.path.join(WORKDIR, 'loan_limit_increases.xlsx')
OUT_RESULTS = os.path.join(WORKDIR, 'loan_optimization_results.csv')
OUT_RECOMM = os.path.join(WORKDIR, 'recommended_increases.csv')
OUT_SIM = os.path.join(WORKDIR, 'simulation_results.csv')

print("Working directory:", WORKDIR)

# If real input is missing, create a tiny synthetic dataset for smoke test
if not os.path.exists(INPUT_XLSX):
    print("Input Excel not found. Creating a small synthetic dataset for a smoke test.")
    n = 50
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        'Customer ID': [f'CUST_{i:05d}' for i in range(n)],
        'Initial Loan ($)': rng.integers(100, 5000, size=n),
        'Days Since Last Loan': rng.integers(0, 400, size=n),
        'On-time Payments (%)': rng.integers(50, 100, size=n),
        'No. of Increases in 2023': rng.integers(0, 4, size=n),
        'Total Profit Contribution ($)': rng.integers(0, 500, size=n)
    })
    # Save to Excel so downstream code can read it if desired
    try:
        df.to_excel(INPUT_XLSX, index=False)
        print(f"Synthetic input written to {INPUT_XLSX}")
    except Exception as e:
        print("Unable to write Excel file (openpyxl missing?). Will proceed with DataFrame in memory.")
        # we'll continue using df in memory
else:
    print("Found input:", INPUT_XLSX)
    try:
        df = pd.read_excel(INPUT_XLSX)
    except Exception as e:
        print("Failed to read the provided Excel file:", e)
        sys.exit(1)

# Basic checks and minimal feature engineering to emulate the original script
required_columns = ['Customer ID', 'Initial Loan ($)', 'Days Since Last Loan', 'On-time Payments (%)', 'No. of Increases in 2023', 'Total Profit Contribution ($)']
for c in required_columns:
    if c not in df.columns:
        print(f"Input is missing required column: {c}")
        sys.exit(1)

print(f"Dataset shape: {df.shape}")

# Small subset of original feature engineering
df['Eligible'] = (df['Days Since Last Loan'] >= 60).astype(int)

def assign_risk_category(payment_rate):
    if payment_rate >= 95:
        return 'Prime'
    elif payment_rate >= 85:
        return 'Near-Prime'
    else:
        return 'Sub-Prime'

df['Risk_Category'] = df['On-time Payments (%)'].apply(assign_risk_category)

# Build tiny uptake model to validate ML path
uptake_features = ['Initial Loan ($)', 'Days Since Last Loan', 'On-time Payments (%)']
X = df[uptake_features].fillna(0)
y = (df['No. of Increases in 2023'] > 0).astype(int)

if len(df) < 10:
    print("Not enough rows to train models reliably. Exiting.")
    sys.exit(1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

lr = LogisticRegression(max_iter=500)
try:
    lr.fit(X_train_sc, y_train)
    proba = lr.predict_proba(X_test_sc)[:, 1]
    auc = roc_auc_score(y_test, proba) if len(np.unique(y_test)) > 1 else None
    print("LogisticRegression trained. Test AUC:", auc)
except Exception as e:
    print("Model training failed:", e)

# Minimal expected value calc
PROFIT_PER_INCREASE = 40

def calc_default_prob(row):
    score = (100 - row['On-time Payments (%)'])
    return 1 / (1 + np.exp(-0.1 * (score - 50)))

df['Uptake_Probability'] = 0.2 + 0.6 * lr.predict_proba(scaler.transform(df[uptake_features]))[:, 1]

df['Default_Probability'] = df.apply(calc_default_prob, axis=1)

def calculate_expected_value(row):
    uptake_prob = row['Uptake_Probability']
    default_prob = row['Default_Probability']
    expected_profit = PROFIT_PER_INCREASE * uptake_prob * (1 - default_prob)
    expected_loss = row['Initial Loan ($)'] * 0.5 * uptake_prob * default_prob
    return expected_profit - expected_loss

df['Expected_Value'] = df.apply(calculate_expected_value, axis=1)

# Create a tiny optimization: recommend increases where EV>0 and Eligible
rec_df = df[(df['Eligible'] == 1) & (df['Expected_Value'] > 0)].copy()
rec_df['Recommended_Increases'] = 1
rec_df['Total_Expected_Value'] = rec_df['Expected_Value'] * rec_df['Recommended_Increases']

# Monte Carlo smoke simulation with tiny sizes
def simulate_row(row, n_sim=10, periods=2):
    rng = np.random.default_rng(42)
    vals = []
    for i in range(n_sim):
        profit = 0
        loss = 0
        for p in range(periods):
            if rng.random() < row['Uptake_Probability']:
                if rng.random() < row['Default_Probability']:
                    loss += row['Initial Loan ($)'] * 0.5
                else:
                    profit += PROFIT_PER_INCREASE
        vals.append(profit - loss)
    return np.mean(vals)

sim_results = []
for _, r in df.iterrows():
    sim_results.append({'Customer ID': r['Customer ID'], 'net_value': simulate_row(r)})

sim_df = pd.DataFrame(sim_results)

# Save outputs
try:
    df.to_csv(OUT_RESULTS, index=False)
    rec_df.to_csv(OUT_RECOMM, index=False)
    sim_df.to_csv(OUT_SIM, index=False)
    print(f"Outputs written to: {OUT_RESULTS}, {OUT_RECOMM}, {OUT_SIM}")
except Exception as e:
    print("Failed to write outputs:", e)
    sys.exit(1)

print("Smoke test completed successfully.")
