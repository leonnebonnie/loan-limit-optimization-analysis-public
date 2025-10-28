"""
Visualization Script for Loan Optimization Analysis
Generates charts and graphs for the technical assessment report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import os

# Set style
sns.set_style('whitegrid')
sns.set_palette("husl")

# Load results (use workspace-relative paths)
WORKDIR = os.path.abspath(os.path.dirname(__file__))
df_path = os.path.join(WORKDIR, 'loan_optimization_results.csv')
rec_path = os.path.join(WORKDIR, 'recommended_increases.csv')
sim_path = os.path.join(WORKDIR, 'simulation_results.csv')

df = pd.read_csv(df_path)
recommendations = pd.read_csv(rec_path)
simulations = pd.read_csv(sim_path)

print("Creating visualizations...")

# ============================================================================
# Figure 1: Dataset Overview and Distribution Analysis
# ============================================================================

fig1 = plt.figure(figsize=(16, 10))
gs = GridSpec(3, 3, figure=fig1, hspace=0.3, wspace=0.3)

# 1.1 Distribution of On-time Payment Rates
ax1 = fig1.add_subplot(gs[0, 0])
df['On-time Payments (%)'].hist(bins=50, ax=ax1, edgecolor='black', alpha=0.7)
ax1.set_xlabel('On-time Payment Rate (%)')
ax1.set_ylabel('Frequency')
ax1.set_title('Distribution of Payment Rates')
ax1.axvline(df['On-time Payments (%)'].mean(), color='red', linestyle='--', label=f'Mean: {df["On-time Payments (%)"].mean():.1f}%')
ax1.legend()

# 1.2 Initial Loan Amount Distribution
ax2 = fig1.add_subplot(gs[0, 1])
df['Initial Loan ($)'].hist(bins=50, ax=ax2, edgecolor='black', alpha=0.7, color='green')
ax2.set_xlabel('Initial Loan Amount ($)')
ax2.set_ylabel('Frequency')
ax2.set_title('Distribution of Loan Amounts')
ax2.axvline(df['Initial Loan ($)'].mean(), color='red', linestyle='--', label=f'Mean: ${df["Initial Loan ($)"].mean():.0f}')
ax2.legend()

# 1.3 Days Since Last Loan
ax3 = fig1.add_subplot(gs[0, 2])
df['Days Since Last Loan'].hist(bins=50, ax=ax3, edgecolor='black', alpha=0.7, color='orange')
ax3.set_xlabel('Days Since Last Loan')
ax3.set_ylabel('Frequency')
ax3.set_title('Time Since Last Loan')
ax3.axvline(60, color='red', linestyle='--', label='Eligibility Threshold (60 days)')
ax3.legend()

# 1.4 Risk Category Distribution
ax4 = fig1.add_subplot(gs[1, 0])
risk_counts = df['Risk_Category'].value_counts()
ax4.bar(risk_counts.index, risk_counts.values, edgecolor='black', alpha=0.7)
ax4.set_xlabel('Risk Category')
ax4.set_ylabel('Number of Customers')
ax4.set_title('Customer Distribution by Risk')
for i, v in enumerate(risk_counts.values):
    ax4.text(i, v + 200, str(v), ha='center', fontweight='bold')

# 1.5 Historical Increases Distribution
ax5 = fig1.add_subplot(gs[1, 1])
increases_counts = df['No. of Increases in 2023'].value_counts().sort_index()
ax5.bar(increases_counts.index, increases_counts.values, edgecolor='black', alpha=0.7, color='purple')
ax5.set_xlabel('Number of Increases in 2023')
ax5.set_ylabel('Number of Customers')
ax5.set_title('Historical Increase Distribution')
for i, (k, v) in enumerate(increases_counts.items()):
    ax5.text(k, v + 200, str(v), ha='center', fontweight='bold')

# 1.6 Profit Contribution Distribution
ax6 = fig1.add_subplot(gs[1, 2])
profit_counts = df['Total Profit Contribution ($)'].value_counts().sort_index()
ax6.bar(profit_counts.index, profit_counts.values, edgecolor='black', alpha=0.7, color='teal')
ax6.set_xlabel('Total Profit Contribution ($)')
ax6.set_ylabel('Number of Customers')
ax6.set_title('Historical Profit Distribution')
for i, (k, v) in enumerate(profit_counts.items()):
    ax6.text(k, v + 200, f'{int(v)}', ha='center', fontweight='bold')

# 1.7 Correlation Heatmap
ax7 = fig1.add_subplot(gs[2, :2])
numeric_cols = ['Initial Loan ($)', 'Days Since Last Loan', 'On-time Payments (%)', 
                'No. of Increases in 2023', 'Credit_Score_Proxy']
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax7, 
            square=True, linewidths=1)
ax7.set_title('Feature Correlation Matrix')

# 1.8 Uptake vs Payment Performance
ax8 = fig1.add_subplot(gs[2, 2])
risk_uptake = df.groupby('Risk_Category')['Uptake_Probability'].mean().sort_values()
colors = ['#d62728', '#ff7f0e', '#2ca02c']
ax8.barh(risk_uptake.index, risk_uptake.values, edgecolor='black', alpha=0.7, color=colors)
ax8.set_xlabel('Average Uptake Probability')
ax8.set_ylabel('Risk Category')
ax8.set_title('Uptake Probability by Risk Segment')
for i, v in enumerate(risk_uptake.values):
    ax8.text(v + 0.01, i, f'{v:.2%}', va='center', fontweight='bold')

plt.suptitle('Loan Limit Optimization - Dataset Overview', fontsize=16, fontweight='bold', y=0.995)
fig1_path = os.path.join(WORKDIR, 'fig1_dataset_overview.png')
plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
print(f"Saved: {fig1_path}")

# ============================================================================
# Figure 2: Model Performance and Risk Analysis
# ============================================================================

fig2 = plt.figure(figsize=(16, 10))
gs2 = GridSpec(2, 3, figure=fig2, hspace=0.3, wspace=0.3)

# 2.1 Default Probability by Risk Category
ax1 = fig2.add_subplot(gs2[0, 0])
risk_default = df.groupby('Risk_Category')['Default_Probability'].agg(['mean', 'std', 'median'])
x_pos = np.arange(len(risk_default.index))
ax1.bar(x_pos, risk_default['mean'], yerr=risk_default['std'], capsize=5, 
        edgecolor='black', alpha=0.7, color=['#2ca02c', '#ff7f0e', '#d62728'])
ax1.set_xticks(x_pos)
ax1.set_xticklabels(risk_default.index)
ax1.set_xlabel('Risk Category')
ax1.set_ylabel('Default Probability')
ax1.set_title('Default Risk by Customer Segment')
for i, v in enumerate(risk_default['mean']):
    ax1.text(i, v + 0.01, f'{v:.2%}', ha='center', fontweight='bold')

# 2.2 Expected Value Distribution
ax2 = fig2.add_subplot(gs2[0, 1])
ev_positive = df[df['Expected_Value'] > 0]['Expected_Value']
ev_negative = df[df['Expected_Value'] <= 0]['Expected_Value']
ax2.hist([ev_positive, ev_negative], bins=50, label=['Positive EV', 'Negative EV'], 
         alpha=0.7, edgecolor='black')
ax2.set_xlabel('Expected Value ($)')
ax2.set_ylabel('Frequency')
ax2.set_title('Distribution of Expected Values')
ax2.axvline(0, color='red', linestyle='--', linewidth=2)
ax2.legend()

# 2.3 Risk-Reward Scatter
ax3 = fig2.add_subplot(gs2[0, 2])
scatter_sample = df.sample(n=min(2000, len(df)), random_state=42)
colors_map = {'Prime': '#2ca02c', 'Near-Prime': '#ff7f0e', 'Sub-Prime': '#d62728'}
for risk_cat in ['Prime', 'Near-Prime', 'Sub-Prime']:
    mask = scatter_sample['Risk_Category'] == risk_cat
    ax3.scatter(scatter_sample[mask]['Default_Probability'], 
               scatter_sample[mask]['Expected_Value'],
               alpha=0.6, label=risk_cat, c=colors_map[risk_cat], s=30)
ax3.set_xlabel('Default Probability')
ax3.set_ylabel('Expected Value ($)')
ax3.set_title('Risk-Return Trade-off')
ax3.axhline(0, color='black', linestyle='--', linewidth=1)
ax3.legend()
ax3.grid(True, alpha=0.3)

# 2.4 NPV by Risk Category
ax4 = fig2.add_subplot(gs2[1, 0])
npv_by_risk = df.groupby('Risk_Category')['Customer_NPV'].agg(['mean', 'median'])
x_pos = np.arange(len(npv_by_risk.index))
width = 0.35
ax4.bar(x_pos - width/2, npv_by_risk['mean'], width, label='Mean', 
        edgecolor='black', alpha=0.7)
ax4.bar(x_pos + width/2, npv_by_risk['median'], width, label='Median', 
        edgecolor='black', alpha=0.7)
ax4.set_xticks(x_pos)
ax4.set_xticklabels(npv_by_risk.index)
ax4.set_xlabel('Risk Category')
ax4.set_ylabel('NPV ($)')
ax4.set_title('Net Present Value by Segment')
ax4.legend()
ax4.axhline(0, color='red', linestyle='--', linewidth=1)

# 2.5 Credit Score Distribution
ax5 = fig2.add_subplot(gs2[1, 1])
for risk_cat in ['Prime', 'Near-Prime', 'Sub-Prime']:
    data = df[df['Risk_Category'] == risk_cat]['Credit_Score_Proxy']
    ax5.hist(data, alpha=0.5, label=risk_cat, bins=30, edgecolor='black')
ax5.set_xlabel('Credit Score Proxy')
ax5.set_ylabel('Frequency')
ax5.set_title('Credit Score Distribution by Risk Category')
ax5.legend()

# 2.6 Optimization Results Summary
ax6 = fig2.add_subplot(gs2[1, 2])
metrics = ['Total\nEligible', 'Approved\nfor Increase', 'High Risk\nApproved']
values = [df['Eligible'].sum(), len(recommendations), 
          len(recommendations[recommendations['Risk_Category'] == 'Sub-Prime'])]
colors_bar = ['#3498db', '#2ecc71', '#e74c3c']
bars = ax6.bar(metrics, values, edgecolor='black', alpha=0.7, color=colors_bar)
ax6.set_ylabel('Number of Customers')
ax6.set_title('Optimization Approval Summary')
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(val):,}', ha='center', va='bottom', fontweight='bold')

plt.suptitle('Model Performance and Risk Analysis', fontsize=16, fontweight='bold', y=0.995)
fig2_path = os.path.join(WORKDIR, 'fig2_model_performance.png')
plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
print(f"Saved: {fig2_path}")

# ============================================================================
# Figure 3: Monte Carlo Simulation Results
# ============================================================================

fig3 = plt.figure(figsize=(16, 10))
gs3 = GridSpec(2, 3, figure=fig3, hspace=0.3, wspace=0.3)

# 3.1 Net Value Distribution by Risk Category
ax1 = fig3.add_subplot(gs3[0, :2])
for risk_cat in ['Prime', 'Near-Prime', 'Sub-Prime']:
    data = simulations[simulations['risk_category'] == risk_cat]['net_value']
    ax1.hist(data, alpha=0.5, label=risk_cat, bins=50, edgecolor='black')
ax1.set_xlabel('Net Value ($)')
ax1.set_ylabel('Frequency')
ax1.set_title('Monte Carlo Simulation: Net Value Distribution')
ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Break-even')
ax1.legend()

# 3.2 Default Rate Distribution
ax2 = fig3.add_subplot(gs3[0, 2])
simulations.groupby('risk_category')['defaults'].mean().plot(kind='bar', ax=ax2, 
                                                              edgecolor='black', alpha=0.7)
ax2.set_xlabel('Risk Category')
ax2.set_ylabel('Average Defaults')
ax2.set_title('Simulated Default Rates')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)

# 3.3 Profit vs Loss Distribution
ax3 = fig3.add_subplot(gs3[1, 0])
profit_data = simulations.groupby('customer_id')[['total_profit', 'total_losses']].mean()
ax3.scatter(profit_data['total_profit'], profit_data['total_losses'], alpha=0.5, s=20)
ax3.set_xlabel('Average Profit ($)')
ax3.set_ylabel('Average Losses ($)')
ax3.set_title('Profit vs Loss Trade-off (Simulated)')
ax3.plot([0, profit_data['total_profit'].max()], [0, profit_data['total_profit'].max()], 
        'r--', label='Break-even line')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 3.4 Increases Granted Distribution
ax4 = fig3.add_subplot(gs3[1, 1])
simulations.groupby('risk_category')['increases_granted'].mean().plot(kind='bar', ax=ax4, 
                                                                       edgecolor='black', alpha=0.7,
                                                                       color=['#2ca02c', '#ff7f0e', '#d62728'])
ax4.set_xlabel('Risk Category')
ax4.set_ylabel('Average Increases Granted')
ax4.set_title('Simulated Loan Increases')
ax4.set_xticklabels(ax4.get_xticklabels(), rotation=0)

# 3.5 Simulation Statistics Summary
ax5 = fig3.add_subplot(gs3[1, 2])
summary_stats = simulations.groupby('risk_category')['net_value'].describe()[['mean', '50%', 'std']]
x_labels = ['Prime', 'Near-Prime', 'Sub-Prime']
x_pos = np.arange(len(x_labels))
width = 0.25

ax5.bar(x_pos - width, summary_stats['mean'], width, label='Mean', edgecolor='black', alpha=0.7)
ax5.bar(x_pos, summary_stats['50%'], width, label='Median', edgecolor='black', alpha=0.7)
ax5.bar(x_pos + width, summary_stats['std'], width, label='Std Dev', edgecolor='black', alpha=0.7)
ax5.set_xticks(x_pos)
ax5.set_xticklabels(x_labels)
ax5.set_xlabel('Risk Category')
ax5.set_ylabel('Net Value ($)')
ax5.set_title('Statistical Summary of Simulations')
ax5.legend()
ax5.axhline(0, color='red', linestyle='--', linewidth=1)

plt.suptitle('Monte Carlo Simulation Results', fontsize=16, fontweight='bold', y=0.995)
fig3_path = os.path.join(WORKDIR, 'fig3_monte_carlo.png')
plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
print(f"Saved: {fig3_path}")

# ============================================================================
# Figure 4: Optimization Strategy Visualization
# ============================================================================

fig4 = plt.figure(figsize=(16, 8))
gs4 = GridSpec(2, 3, figure=fig4, hspace=0.3, wspace=0.3)

# 4.1 Recommended Increases Distribution
ax1 = fig4.add_subplot(gs4[0, 0])
rec_dist = recommendations['Recommended_Increases'].value_counts().sort_index()
ax1.bar(rec_dist.index, rec_dist.values, edgecolor='black', alpha=0.7, color='steelblue')
ax1.set_xlabel('Number of Recommended Increases')
ax1.set_ylabel('Number of Customers')
ax1.set_title('Distribution of Recommended Increases')
for i, (k, v) in enumerate(rec_dist.items()):
    ax1.text(k, v + 20, str(v), ha='center', fontweight='bold')

# 4.2 Expected Value by Recommended Increases
ax2 = fig4.add_subplot(gs4[0, 1])
ev_by_increases = recommendations.groupby('Recommended_Increases')['Total_Expected_Value'].mean()
ax2.plot(ev_by_increases.index, ev_by_increases.values, marker='o', linewidth=2, markersize=8)
ax2.set_xlabel('Number of Recommended Increases')
ax2.set_ylabel('Average Total Expected Value ($)')
ax2.set_title('Expected Value by Increase Amount')
ax2.grid(True, alpha=0.3)

# 4.3 Risk Category in Approved Customers
ax3 = fig4.add_subplot(gs4[0, 2])
approved_risk = recommendations['Risk_Category'].value_counts()
colors_pie = ['#2ca02c', '#ff7f0e', '#d62728']
wedges, texts, autotexts = ax3.pie(approved_risk.values, labels=approved_risk.index, autopct='%1.1f%%',
                                    colors=colors_pie, startangle=90, explode=[0.05, 0.05, 0.05])
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
ax3.set_title('Risk Distribution in Approved Customers')

# 4.4 Payment Performance of Approved Customers
ax4 = fig4.add_subplot(gs4[1, 0])
recommendations['On-time Payments (%)'].hist(bins=30, ax=ax4, edgecolor='black', alpha=0.7, color='green')
ax4.set_xlabel('On-time Payment Rate (%)')
ax4.set_ylabel('Frequency')
ax4.set_title('Payment Performance of Approved Customers')
ax4.axvline(recommendations['On-time Payments (%)'].mean(), color='red', linestyle='--', 
           label=f'Mean: {recommendations["On-time Payments (%)"].mean():.1f}%')
ax4.legend()

# 4.5 Loan Amount Distribution of Approved Customers
ax5 = fig4.add_subplot(gs4[1, 1])
recommendations['Initial Loan ($)'].hist(bins=30, ax=ax5, edgecolor='black', alpha=0.7, color='purple')
ax5.set_xlabel('Initial Loan Amount ($)')
ax5.set_ylabel('Frequency')
ax5.set_title('Loan Sizes of Approved Customers')
ax5.axvline(recommendations['Initial Loan ($)'].mean(), color='red', linestyle='--', 
           label=f'Mean: ${recommendations["Initial Loan ($)"].mean():.0f}')
ax5.legend()

# 4.6 Default Risk of Approved Customers
ax6 = fig4.add_subplot(gs4[1, 2])
recommendations['Default_Probability'].hist(bins=30, ax=ax6, edgecolor='black', alpha=0.7, color='coral')
ax6.set_xlabel('Default Probability')
ax6.set_ylabel('Frequency')
ax6.set_title('Risk Profile of Approved Customers')
ax6.axvline(recommendations['Default_Probability'].mean(), color='red', linestyle='--', 
           label=f'Mean: {recommendations["Default_Probability"].mean():.2%}')
ax6.legend()

plt.suptitle('Optimization Strategy Results', fontsize=16, fontweight='bold', y=0.995)
fig4_path = os.path.join(WORKDIR, 'fig4_optimization_strategy.png')
plt.savefig(fig4_path, dpi=300, bbox_inches='tight')
print(f"Saved: {fig4_path}")

plt.close('all')
print("\nAll visualizations created successfully!")
