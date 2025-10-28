"""
Cleaned Loan Limit Optimization Analysis
- Keeps original logic and defaults
- Reorganized into functions with small docstrings and type hints
- Adds a CLI (argparse) for input/output and simulation parameters
"""
from __future__ import annotations

import argparse
import os
import random
from typing import Optional

import numpy as np
import pandas as pd

# ML and plotting imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# Constants (keep same defaults as original)
RANDOM_SEED = 42
PROFIT_PER_INCREASE = 40
MAX_INCREASES_PER_YEAR = 6
DISCOUNT_RATE = 0.19
ELIGIBILITY_THRESHOLD_DAYS = 60

# Set global RNGs for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def load_data(path: str) -> pd.DataFrame:
    """Load Excel data and normalize header if needed."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_excel(path, skiprows=0)
    # If first row accidentally contains headers, detect and adjust
    if 'Customer ID' not in df.columns:
        first_row = df.iloc[0].astype(str).str.lower()
        if first_row.str.contains('customer id').any():
            df.columns = df.iloc[0]
            df = df.iloc[1:].reset_index(drop=True)
    # Ensure numeric columns are numeric where appropriate
    for col in df.columns:
        if col != 'Customer ID':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features used by models and optimization."""
    df = df.copy()
    df['Eligible'] = (df['Days Since Last Loan'] >= ELIGIBILITY_THRESHOLD_DAYS).astype(int)
    df['Received_Increase'] = (df['No. of Increases in 2023'] > 0).astype(int)

    def assign_risk_category(payment_rate: float) -> str:
        if payment_rate >= 95:
            return 'Prime'
        elif payment_rate >= 85:
            return 'Near-Prime'
        else:
            return 'Sub-Prime'

    df['Risk_Category'] = df['On-time Payments (%)'].apply(assign_risk_category)
    df['Loan_Size_Category'] = pd.cut(df['Initial Loan ($)'], bins=[0, 1500, 3000, 5000], labels=['Small', 'Medium', 'Large'])

    df['Credit_Score_Proxy'] = (
        df['On-time Payments (%)'] * 0.6 +
        (df['Days Since Last Loan'] / df['Days Since Last Loan'].max() * 100) * 0.2 +
        ((df['Initial Loan ($)'] / df['Initial Loan ($)'].max()) * 100) * 0.2
    )

    df['Payment_Days_Interaction'] = df['On-time Payments (%)'] * df['Days Since Last Loan'] / 100
    df['Loan_Payment_Ratio'] = df['Initial Loan ($)'] / (df['On-time Payments (%)'] + 1)
    return df


def build_uptake_models(df: pd.DataFrame):
    """Train three uptake models and return the best one (by AUC) plus a mapping."""
    features = ['Initial Loan ($)', 'Days Since Last Loan', 'On-time Payments (%)',
                'Credit_Score_Proxy', 'Payment_Days_Interaction', 'Loan_Payment_Ratio']
    X = df[features]
    y = df['Received_Increase']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    lr = LogisticRegression(random_state=RANDOM_SEED, max_iter=1000)
    lr.fit(X_train_sc, y_train)

    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, max_depth=10)
    rf.fit(X_train, y_train)

    gb = GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_SEED, max_depth=5)
    gb.fit(X_train, y_train)

    models = {
        'Logistic Regression': (lr, X_test_sc),
        'Random Forest': (rf, X_test),
        'Gradient Boosting': (gb, X_test)
    }

    best_name = None
    best_auc = -1.0
    best_model = None

    for name, (model, Xt) in models.items():
        try:
            proba = model.predict_proba(Xt)[:, 1]
            auc = roc_auc_score(y_test, proba)
        except Exception:
            auc = 0.0
        if auc > best_auc:
            best_auc = auc
            best_name = name
            best_model = model

    # Attach predicted probabilities to df
    if best_name == 'Logistic Regression':
        df['Uptake_Probability'] = lr.predict_proba(scaler.transform(df[features]))[:, 1]
    else:
        df['Uptake_Probability'] = best_model.predict_proba(df[features])[:, 1]

    return best_name, best_auc, df


def build_default_model(df: pd.DataFrame) -> pd.DataFrame:
    """Construct a simple default probability model used in the original script."""
    df = df.copy()
    df['Default_Risk_Score'] = (
        (100 - df['On-time Payments (%)']) * 0.5 +
        (100 - df['Credit_Score_Proxy']) * 0.3 +
        (df['Initial Loan ($)'] / df['Initial Loan ($)'].max() * 100) * 0.2
    )
    df['Default_Probability'] = 1 / (1 + np.exp(-0.1 * (df['Default_Risk_Score'] - 50)))
    df['Adjusted_Default_Probability'] = df['Default_Probability'] * (1 + 0.05 * df['No. of Increases in 2023'])
    df['Adjusted_Default_Probability'] = df['Adjusted_Default_Probability'].clip(0, 0.95)
    return df


def calculate_expected_value(row: pd.Series) -> float:
    uptake_prob = float(row['Uptake_Probability'])
    default_prob = float(row['Adjusted_Default_Probability'])
    expected_profit = PROFIT_PER_INCREASE * uptake_prob * (1 - default_prob)
    expected_loss = float(row['Initial Loan ($)']) * 0.5 * uptake_prob * default_prob
    return expected_profit - expected_loss


def optimize_loan_increases(df_input: pd.DataFrame, max_high_risk_pct: float = 0.25, capital_constraint: Optional[float] = None):
    """Greedy optimization logic used by the original script."""
    eligible = df_input[df_input['Eligible'] == 1].copy()
    eligible = eligible.sort_values('Risk_Adjusted_Score', ascending=False)

    eligible['Recommended_Increases'] = 0
    eligible['Total_Expected_Value'] = 0

    total_value = 0.0
    total_exposure = 0.0
    high_risk_count = 0
    total_approvals = 0

    for idx, row in eligible.iterrows():
        is_high_risk = row['Risk_Category'] == 'Sub-Prime'
        if is_high_risk and high_risk_count >= len(eligible) * max_high_risk_pct:
            continue
        if row['Expected_Value'] <= 0:
            continue
        optimal_increases = min(MAX_INCREASES_PER_YEAR, int(row['Uptake_Probability'] * MAX_INCREASES_PER_YEAR) + 1)
        if capital_constraint:
            projected_exposure = row['Initial Loan ($)'] * optimal_increases * 0.5
            if total_exposure + projected_exposure > capital_constraint:
                continue
        eligible.at[idx, 'Recommended_Increases'] = optimal_increases
        eligible.at[idx, 'Total_Expected_Value'] = row['Expected_Value'] * optimal_increases
        total_value += eligible.at[idx, 'Total_Expected_Value']
        total_exposure += row['Initial Loan ($)'] * optimal_increases * 0.5
        if is_high_risk:
            high_risk_count += 1
        total_approvals += 1

    results = {
        'eligible_df': eligible,
        'total_expected_value': total_value,
        'total_approvals': total_approvals,
        'total_exposure': total_exposure,
        'high_risk_count': high_risk_count,
        'high_risk_pct': high_risk_count / total_approvals if total_approvals > 0 else 0
    }
    return results


def simulate_loan_lifecycle(customer_row: pd.Series, n_simulations: int = 100, time_periods: int = 4) -> pd.DataFrame:
    """Perform Monte Carlo simulations for a single customer and return a DataFrame of results."""
    rows = []
    for sim in range(n_simulations):
        total_profit = 0.0
        total_losses = 0.0
        defaults = 0
        increases_granted = 0
        current_risk_state = customer_row['Risk_Category']
        for quarter in range(time_periods):
            if quarter > 0:
                state_idx = {'Prime': 0, 'Near-Prime': 1, 'Sub-Prime': 2}[current_risk_state]
                probs = np.array([[0.85, 0.12, 0.03], [0.15, 0.7, 0.15], [0.05, 0.25, 0.7]])[state_idx]
                next_state_idx = np.random.choice(3, p=probs)
                current_risk_state = ['Prime', 'Near-Prime', 'Sub-Prime'][next_state_idx]
            risk_multiplier = {'Prime': 0.8, 'Near-Prime': 1.0, 'Sub-Prime': 1.3}[current_risk_state]
            adjusted_default_prob = min(customer_row['Default_Probability'] * risk_multiplier, 0.95)
            accepts_increase = np.random.random() < customer_row['Uptake_Probability']
            if accepts_increase:
                increases_granted += 1
                defaults_this_period = np.random.random() < adjusted_default_prob
                if defaults_this_period:
                    defaults += 1
                    total_losses += customer_row['Initial Loan ($)'] * 0.5
                else:
                    total_profit += PROFIT_PER_INCREASE
        net_value = total_profit - total_losses
        rows.append({'simulation': sim, 'total_profit': total_profit, 'total_losses': total_losses,
                     'net_value': net_value, 'defaults': defaults, 'increases_granted': increases_granted})
    return pd.DataFrame(rows)


def calculate_npv(cash_flows: list[float], discount_rate: float = DISCOUNT_RATE) -> float:
    npv = 0.0
    for t, cf in enumerate(cash_flows):
        npv += cf / ((1 + discount_rate) ** (t / 4))
    return npv


def main():
    parser = argparse.ArgumentParser(description='Loan limit optimization analysis (clean)')
    parser.add_argument('--input', default='loan_limit_increases.xlsx', help='Input Excel file')
    parser.add_argument('--sim-customers', type=int, default=int(os.environ.get('SIM_CUSTOMER_COUNT', '1000')),
                        help='Number of customers to simulate (sampled from eligible)')
    parser.add_argument('--sims-per-customer', type=int, default=int(os.environ.get('N_SIMULATIONS_PER_CUSTOMER', '100')),
                        help='Number of Monte Carlo simulations per customer')
    parser.add_argument('--quarters', type=int, default=int(os.environ.get('SIM_QUARTERS', '4')),
                        help='Number of quarters per simulation')
    args = parser.parse_args()

    df = load_data(args.input)
    print(f"Dataset Shape: {df.shape}")
    df = feature_engineering(df)
    print(f"Eligible Customers (>=60 days): {df['Eligible'].sum():,} ({df['Eligible'].mean()*100:.1f}%)")

    best_name, best_auc, df = build_uptake_models(df)
    print(f"Best Uptake Model: {best_name} (AUC: {best_auc:.4f})")

    df = build_default_model(df)
    df['Expected_Value'] = df.apply(calculate_expected_value, axis=1)
    df['Risk_Adjusted_Score'] = df['Expected_Value'] * (1 - df['Adjusted_Default_Probability']) * df['Uptake_Probability']

    optimization_results = optimize_loan_increases(df, max_high_risk_pct=0.25)
    print(f"Approved for Increases: {optimization_results['total_approvals']:,}")
    print(f"Total Expected Value: ${optimization_results['total_expected_value']:.2f}")

    # Sampling customers for Monte Carlo
    eligible_customers = df[df['Eligible'] == 1]
    available = len(eligible_customers)
    sample_n = args.sim_customers
    if sample_n <= available:
        sample_customers = eligible_customers.sample(n=sample_n, random_state=RANDOM_SEED)
    else:
        sample_customers = eligible_customers.sample(n=sample_n, replace=True, random_state=RANDOM_SEED)

    sim_results = []
    for _, row in sample_customers.iterrows():
        sim_df = simulate_loan_lifecycle(row, n_simulations=args.sims_per_customer, time_periods=args.quarters)
        sim_df['customer_id'] = row['Customer ID']
        sim_df['risk_category'] = row['Risk_Category']
        sim_results.append(sim_df)

    all_sims = pd.concat(sim_results, ignore_index=True)
    total_runs = sample_n * args.sims_per_customer
    total_decisions = total_runs * args.quarters
    print(f"Total Simulation Runs: {total_runs:,}")
    print(f"Total Individual Decisions: {total_decisions:,}")

    # NPV
    df['Customer_NPV'] = df.apply(lambda r: calculate_npv([r['Expected_Value']] * args.quarters), axis=1)

    # Save outputs (same filenames as original script)
    df.to_csv('loan_optimization_results.csv', index=False)
    optimization_output = optimization_results['eligible_df'][optimization_results['eligible_df']['Recommended_Increases'] > 0][
        ['Customer ID', 'Risk_Category', 'On-time Payments (%)', 'Initial Loan ($)', 'Uptake_Probability',
         'Default_Probability', 'Expected_Value', 'Recommended_Increases', 'Total_Expected_Value']]
    optimization_output.to_csv('recommended_increases.csv', index=False)
    all_sims.to_csv('simulation_results.csv', index=False)
    print('Saved outputs: loan_optimization_results.csv, recommended_increases.csv, simulation_results.csv')


if __name__ == '__main__':
    main()
