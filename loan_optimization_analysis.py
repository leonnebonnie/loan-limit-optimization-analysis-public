"""
LOAN LIMIT OPTIMIZATION MODEL - WINDOWS VERSION
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize, LinearConstraint
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import warnings
import os
import random
warnings.filterwarnings('ignore')

# Global random seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Set style for visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("="*80)
print(" LOAN LIMIT OPTIMIZATION MODEL - COMPREHENSIVE ANALYSIS")
print("="*80)

# ============================================================================
# 1. DATA LOADING AND EXPLORATORY DATA ANALYSIS
# ============================================================================

print("\n[1] LOADING AND EXPLORING DATA")
print("-"*80)

# Load data - WINDOWS COMPATIBLE PATH
# Look for Excel file in current directory
excel_file = 'loan_limit_increases.xlsx'

if not os.path.exists(excel_file):
    print(f"ERROR: Cannot find '{excel_file}' in current directory")
    print(f"Current directory: {os.getcwd()}")
    print("\nPlease ensure loan_limit_increases.xlsx is in the same folder as this script.")
    exit(1)

df = pd.read_excel(excel_file, skiprows=0)
# If the first row contains column names (some Excel exports put headers in the first row of data),
# normalize so we don't accidentally overwrite proper headers.
if 'Customer ID' not in df.columns:
    # check if first row looks like headers
    first_row = df.iloc[0].astype(str).str.lower()
    if first_row.str.contains('customer id').any():
        df.columns = df.iloc[0]
        df = df.iloc[1:].reset_index(drop=True)

# Convert to numeric
for col in df.columns:
    if col != 'Customer ID':
        df[col] = pd.to_numeric(df[col], errors='coerce')

print(f"Dataset Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nBasic Statistics:\n{df.describe()}")

# Calculate derived metrics
df['Eligible'] = (df['Days Since Last Loan'] >= 60).astype(int)
df['Received_Increase'] = (df['No. of Increases in 2023'] > 0).astype(int)
df['Average_Profit_Per_Increase'] = df['Total Profit Contribution ($)'] / df['No. of Increases in 2023'].replace(0, 1)

print(f"\nEligibility Analysis:")
print(f"Eligible Customers (>=60 days): {df['Eligible'].sum():,} ({df['Eligible'].mean()*100:.1f}%)")
print(f"Customers who received increases: {df['Received_Increase'].sum():,} ({df['Received_Increase'].mean()*100:.1f}%)")

# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================

print("\n[2] FEATURE ENGINEERING")
print("-"*80)

def assign_risk_category(payment_rate):
    if payment_rate >= 95:
        return 'Prime'
    elif payment_rate >= 85:
        return 'Near-Prime'
    else:
        return 'Sub-Prime'

df['Risk_Category'] = df['On-time Payments (%)'].apply(assign_risk_category)
df['Loan_Size_Category'] = pd.cut(df['Initial Loan ($)'], 
                                   bins=[0, 1500, 3000, 5000], 
                                   labels=['Small', 'Medium', 'Large'])

df['Credit_Score_Proxy'] = (
    df['On-time Payments (%)'] * 0.6 + 
    (df['Days Since Last Loan'] / df['Days Since Last Loan'].max() * 100) * 0.2 +
    ((df['Initial Loan ($)'] / df['Initial Loan ($)'].max()) * 100) * 0.2
)

df['Payment_Days_Interaction'] = df['On-time Payments (%)'] * df['Days Since Last Loan'] / 100
df['Loan_Payment_Ratio'] = df['Initial Loan ($)'] / (df['On-time Payments (%)'] + 1)

print("Feature Engineering Complete:")
print(f"  - Risk Categories: {df['Risk_Category'].value_counts().to_dict()}")
print(f"  - Loan Size Categories: {df['Loan_Size_Category'].value_counts().to_dict()}")
print(f"  - Credit Score Proxy: Mean={df['Credit_Score_Proxy'].mean():.2f}, Std={df['Credit_Score_Proxy'].std():.2f}")

# ============================================================================
# 3. PREDICTIVE MODELING - UPTAKE PROBABILITY
# ============================================================================

print("\n[3] BUILDING UPTAKE PROBABILITY MODEL")
print("-"*80)

uptake_features = ['Initial Loan ($)', 'Days Since Last Loan', 'On-time Payments (%)', 
                   'Credit_Score_Proxy', 'Payment_Days_Interaction', 'Loan_Payment_Ratio']

X_uptake = df[uptake_features]
y_uptake = df['Received_Increase']

X_train_uptake, X_test_uptake, y_train_uptake, y_test_uptake = train_test_split(
    X_uptake, y_uptake, test_size=0.2, random_state=42, stratify=y_uptake
)

scaler_uptake = StandardScaler()
X_train_uptake_scaled = scaler_uptake.fit_transform(X_train_uptake)
X_test_uptake_scaled = scaler_uptake.transform(X_test_uptake)

print("\nTraining Uptake Prediction Models...")

lr_uptake = LogisticRegression(random_state=42, max_iter=1000)
lr_uptake.fit(X_train_uptake_scaled, y_train_uptake)

rf_uptake = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf_uptake.fit(X_train_uptake, y_train_uptake)

gb_uptake = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5)
gb_uptake.fit(X_train_uptake, y_train_uptake)

models_uptake = {
    'Logistic Regression': (lr_uptake, X_test_uptake_scaled),
    'Random Forest': (rf_uptake, X_test_uptake),
    'Gradient Boosting': (gb_uptake, X_test_uptake)
}

print("\nUptake Model Performance:")
best_model_name_uptake = None
best_auc_uptake = 0

for name, (model, X_test) in models_uptake.items():
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test_uptake, y_pred_proba)
    accuracy = (y_pred == y_test_uptake).mean()
    
    print(f"\n  {name}:")
    print(f"    - Accuracy: {accuracy:.4f}")
    print(f"    - AUC-ROC: {auc:.4f}")
    
    if auc > best_auc_uptake:
        best_auc_uptake = auc
        best_model_name_uptake = name
        best_model_uptake = model

print(f"\nBest Uptake Model: {best_model_name_uptake} (AUC: {best_auc_uptake:.4f})")

if best_model_name_uptake == 'Logistic Regression':
    df['Uptake_Probability'] = lr_uptake.predict_proba(scaler_uptake.transform(X_uptake))[:, 1]
else:
    df['Uptake_Probability'] = best_model_uptake.predict_proba(X_uptake)[:, 1]

# ============================================================================
# 4. RISK MODELING - DEFAULT PROBABILITY
# ============================================================================

print("\n[4] BUILDING DEFAULT RISK MODEL")
print("-"*80)

df['Default_Risk_Score'] = (
    (100 - df['On-time Payments (%)']) * 0.5 +
    (100 - df['Credit_Score_Proxy']) * 0.3 +
    (df['Initial Loan ($)'] / df['Initial Loan ($)'].max() * 100) * 0.2
)

df['Default_Probability'] = 1 / (1 + np.exp(-0.1 * (df['Default_Risk_Score'] - 50)))
df['Adjusted_Default_Probability'] = df['Default_Probability'] * (1 + 0.05 * df['No. of Increases in 2023'])
df['Adjusted_Default_Probability'] = df['Adjusted_Default_Probability'].clip(0, 0.95)

print(f"Default Probability Statistics:")
print(f"  Mean: {df['Default_Probability'].mean():.4f}")
print(f"  Median: {df['Default_Probability'].median():.4f}")
print(f"  Std: {df['Default_Probability'].std():.4f}")

print(f"\nDefault Probability by Risk Category:")
print(df.groupby('Risk_Category')['Default_Probability'].describe()[['mean', '50%', 'std']])

# ============================================================================
# 5. MARKOV CHAIN MODEL
# ============================================================================

print("\n[5] MARKOV CHAIN MODEL FOR RISK STATE TRANSITIONS")
print("-"*80)

transition_matrix = np.array([
    [0.85, 0.12, 0.03],
    [0.15, 0.70, 0.15],
    [0.05, 0.25, 0.70]
])

risk_states = ['Prime', 'Near-Prime', 'Sub-Prime']

print("Transition Matrix:")
transition_df = pd.DataFrame(transition_matrix, index=risk_states, columns=risk_states)
print(transition_df)

eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
steady_state = eigenvectors[:, 0] / eigenvectors[:, 0].sum()
steady_state = np.real(steady_state)

print(f"\nSteady State Distribution:")
for state, prob in zip(risk_states, steady_state):
    print(f"  {state}: {prob:.4f}")

# ============================================================================
# 6. OPTIMIZATION MODEL
# ============================================================================

print("\n[6] OPTIMIZATION MODEL FOR LOAN LIMIT INCREASES")
print("-"*80)

PROFIT_PER_INCREASE = 40
MAX_INCREASES_PER_YEAR = 6
DISCOUNT_RATE = 0.19
ELIGIBILITY_THRESHOLD_DAYS = 60

def calculate_expected_value(row):
    uptake_prob = row['Uptake_Probability']
    default_prob = row['Adjusted_Default_Probability']
    expected_profit = PROFIT_PER_INCREASE * uptake_prob * (1 - default_prob)
    expected_loss = row['Initial Loan ($)'] * 0.5 * uptake_prob * default_prob
    expected_value = expected_profit - expected_loss
    return expected_value

df['Expected_Value'] = df.apply(calculate_expected_value, axis=1)
df['Risk_Adjusted_Score'] = (
    df['Expected_Value'] * 
    (1 - df['Adjusted_Default_Probability']) * 
    df['Uptake_Probability']
)

print(f"Expected Value Statistics:")
print(f"  Mean: ${df['Expected_Value'].mean():.2f}")
print(f"  Median: ${df['Expected_Value'].median():.2f}")
print(f"  Total Potential: ${df['Expected_Value'].sum():,.2f}")

def optimize_loan_increases(df_input, max_high_risk_pct=0.30, capital_constraint=None):
    eligible = df_input[df_input['Eligible'] == 1].copy()
    eligible = eligible.sort_values('Risk_Adjusted_Score', ascending=False)
    
    eligible['Recommended_Increases'] = 0
    eligible['Total_Expected_Value'] = 0
    
    total_value = 0
    total_exposure = 0
    high_risk_count = 0
    total_approvals = 0
    
    for idx, row in eligible.iterrows():
        is_high_risk = row['Risk_Category'] == 'Sub-Prime'
        
        if is_high_risk and high_risk_count >= len(eligible) * max_high_risk_pct:
            continue
        
        if row['Expected_Value'] <= 0:
            continue
        
        optimal_increases = min(
            MAX_INCREASES_PER_YEAR,
            int(row['Uptake_Probability'] * MAX_INCREASES_PER_YEAR) + 1
        )
        
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

optimization_results = optimize_loan_increases(df, max_high_risk_pct=0.25)

print(f"\nOptimization Results:")
print(f"  Total Eligible Customers: {df['Eligible'].sum():,}")
print(f"  Approved for Increases: {optimization_results['total_approvals']:,}")
print(f"  Approval Rate: {optimization_results['total_approvals']/df['Eligible'].sum()*100:.1f}%")
print(f"  Total Expected Value: ${optimization_results['total_expected_value']:,.2f}")
print(f"  Total Capital Exposure: ${optimization_results['total_exposure']:,.2f}")
print(f"  High-Risk Customers: {optimization_results['high_risk_count']} ({optimization_results['high_risk_pct']*100:.1f}%)")

# ============================================================================
# 7. MONTE CARLO SIMULATION
# ============================================================================

print("\n[7] MONTE CARLO SIMULATION FOR LOAN LIFECYCLE")
print("-"*80)

def simulate_loan_lifecycle(customer_row, n_simulations=1000, time_periods=4):
    results = []
    
    for sim in range(n_simulations):
        total_profit = 0
        total_losses = 0
        defaults = 0
        increases_granted = 0
        current_risk_state = customer_row['Risk_Category']
        
        for quarter in range(time_periods):
            if quarter > 0:
                state_idx = {'Prime': 0, 'Near-Prime': 1, 'Sub-Prime': 2}[current_risk_state]
                probs = transition_matrix[state_idx]
                next_state_idx = np.random.choice(3, p=probs)
                current_risk_state = risk_states[next_state_idx]
            
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
        
        results.append({
            'simulation': sim,
            'total_profit': total_profit,
            'total_losses': total_losses,
            'net_value': net_value,
            'defaults': defaults,
            'increases_granted': increases_granted
        })
    
    return pd.DataFrame(results)

print("Running Monte Carlo Simulations (this may take a moment)...")

# Simulation configuration (can be overridden with environment variables):
#   SIM_CUSTOMER_COUNT - total customers to simulate (default 1000)
#   N_SIMULATIONS_PER_CUSTOMER - simulations per customer (default 100)
#   SIM_QUARTERS - quarters per simulation (default 4)
SIM_CUSTOMER_COUNT = int(os.environ.get('SIM_CUSTOMER_COUNT', '1000'))
N_SIMULATIONS_PER_CUSTOMER = int(os.environ.get('N_SIMULATIONS_PER_CUSTOMER', '100'))
SIM_QUARTERS = int(os.environ.get('SIM_QUARTERS', '4'))

eligible_customers = df[df['Eligible'] == 1]
available = len(eligible_customers)
if available == 0:
    raise RuntimeError("No eligible customers available for simulation")

# If requested sample size is larger than available, sample with replacement
if SIM_CUSTOMER_COUNT <= available:
    sample_customers = eligible_customers.sample(n=SIM_CUSTOMER_COUNT, random_state=42)
else:
    sample_customers = eligible_customers.sample(n=SIM_CUSTOMER_COUNT, replace=True, random_state=42)

simulation_results_list = []
for idx, row in sample_customers.iterrows():
    sim_df = simulate_loan_lifecycle(row, n_simulations=N_SIMULATIONS_PER_CUSTOMER, time_periods=SIM_QUARTERS)
    sim_df['customer_id'] = row['Customer ID']
    sim_df['risk_category'] = row['Risk_Category']
    simulation_results_list.append(sim_df)

all_simulations = pd.concat(simulation_results_list, ignore_index=True)

num_customers = sample_customers['Customer ID'].nunique()
total_sim_runs = SIM_CUSTOMER_COUNT * N_SIMULATIONS_PER_CUSTOMER
total_individual_decisions = total_sim_runs * SIM_QUARTERS
print(f"\nSimulation Summary ({num_customers} unique customers sampled, {N_SIMULATIONS_PER_CUSTOMER} simulations each):")
print(f"  Total Simulation Runs: {total_sim_runs:,}")
print(f"  Total Individual Decisions (runs × quarters): {total_individual_decisions:,}")
print(f"  Mean Net Value per Customer: ${all_simulations.groupby('customer_id')['net_value'].mean().mean():.2f}")
print(f"  Median Net Value per Customer: ${all_simulations.groupby('customer_id')['net_value'].mean().median():.2f}")
print(f"  Mean Default Rate: {all_simulations.groupby('customer_id')['defaults'].mean().mean():.2f}")

print(f"\nSimulation Results by Risk Category:")
risk_sim_summary = all_simulations.groupby('risk_category').agg({
    'net_value': ['mean', 'median', 'std'],
    'defaults': 'mean',
    'increases_granted': 'mean'
})
print(risk_sim_summary)

# ============================================================================
# 8. NPV CALCULATION
# ============================================================================

print("\n[8] NET PRESENT VALUE ANALYSIS")
print("-"*80)

def calculate_npv(cash_flows, discount_rate=0.19, periods=4):
    npv = 0
    for t in range(len(cash_flows)):
        npv += cash_flows[t] / ((1 + discount_rate) ** (t / 4))
    return npv

def calculate_customer_npv(row, time_periods=4):
    quarterly_cash_flow = row['Expected_Value']
    cash_flows = [quarterly_cash_flow] * time_periods
    return calculate_npv(cash_flows, DISCOUNT_RATE, time_periods)

df['Customer_NPV'] = df.apply(calculate_customer_npv, axis=1)

print(f"NPV Analysis:")
print(f"  Mean Customer NPV: ${df['Customer_NPV'].mean():.2f}")
print(f"  Median Customer NPV: ${df['Customer_NPV'].median():.2f}")
print(f"  Total Portfolio NPV: ${df['Customer_NPV'].sum():,.2f}")

print(f"\nSensitivity Analysis - Impact of Discount Rate:")
discount_rates = [0.10, 0.15, 0.19, 0.25, 0.30]
for rate in discount_rates:
    test_npv = calculate_npv([PROFIT_PER_INCREASE] * 4, rate, 4)
    print(f"  Discount Rate {rate*100:.0f}%: NPV = ${test_npv:.2f}")

# ============================================================================
# 9. STRATEGY RECOMMENDATIONS
# ============================================================================

print("\n[9] STRATEGIC RECOMMENDATIONS")
print("-"*80)

segment_analysis = df.groupby('Risk_Category').agg({
    'Expected_Value': ['mean', 'sum'],
    'Uptake_Probability': 'mean',
    'Default_Probability': 'mean',
    'Customer_NPV': ['mean', 'sum'],
    'Customer ID': 'count'
}).round(2)

print("\nPerformance by Risk Segment:")
print(segment_analysis)

top_customers = df[df['Eligible'] == 1].nlargest(100, 'Risk_Adjusted_Score')[
    ['Customer ID', 'Risk_Category', 'On-time Payments (%)', 
     'Uptake_Probability', 'Default_Probability', 'Expected_Value', 'Customer_NPV']
]

print(f"\nTop 100 Customers for Targeting:")
print(top_customers.head(10))
print(f"  Total Expected Value: ${top_customers['Expected_Value'].sum():,.2f}")
print(f"  Average Uptake Probability: {top_customers['Uptake_Probability'].mean():.2%}")
print(f"  Average Default Risk: {top_customers['Default_Probability'].mean():.2%}")

# ============================================================================
# 10. SAVE RESULTS
# ============================================================================

print("\n[10] SAVING RESULTS")
print("-"*80)

output_df = df.copy()
output_df.to_csv('loan_optimization_results.csv', index=False)
print("Saved: loan_optimization_results.csv")

optimization_output = optimization_results['eligible_df'][
    optimization_results['eligible_df']['Recommended_Increases'] > 0
][['Customer ID', 'Risk_Category', 'On-time Payments (%)', 'Initial Loan ($)',
   'Uptake_Probability', 'Default_Probability', 'Expected_Value', 
   'Recommended_Increases', 'Total_Expected_Value']]

optimization_output.to_csv('recommended_increases.csv', index=False)
print("Saved: recommended_increases.csv")

all_simulations.to_csv('simulation_results.csv', index=False)
print("Saved: simulation_results.csv")

print("\n" + "="*80)
print(" ANALYSIS COMPLETE")
print("="*80)
print(f"\nKey Findings:")
print(f"  1. {optimization_results['total_approvals']:,} customers recommended for increases")
print(f"  2. Expected value: ${optimization_results['total_expected_value']:,.2f}")
print(f"  3. Best uptake model: {best_model_name_uptake} (AUC: {best_auc_uptake:.4f})")
print(f"  4. Mean customer NPV: ${df['Customer_NPV'].mean():.2f}")
print(f"  5. High-risk customer allocation: {optimization_results['high_risk_pct']*100:.1f}%")