"""
Cleaned visualization script — clearer structure, functions, and relative paths.
"""
from __future__ import annotations

import os
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style('whitegrid')

WORKDIR = os.path.abspath(os.path.dirname(__file__))


def load_results(workdir: str = WORKDIR) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(os.path.join(workdir, 'loan_optimization_results.csv'))
    recommendations = pd.read_csv(os.path.join(workdir, 'recommended_increases.csv'))
    simulations = pd.read_csv(os.path.join(workdir, 'simulation_results.csv'))
    return df, recommendations, simulations


def fig1_dataset_overview(df: pd.DataFrame, outpath: str):
    fig, axes = plt.subplots(3, 3, figsize=(16, 10))
    df['On-time Payments (%)'].hist(bins=50, ax=axes[0, 0])
    df['Initial Loan ($)'].hist(bins=50, ax=axes[0, 1])
    df['Days Since Last Loan'].hist(bins=50, ax=axes[0, 2])
    df['Risk_Category'].value_counts().plot(kind='bar', ax=axes[1, 0])
    df['No. of Increases in 2023'].value_counts().sort_index().plot(kind='bar', ax=axes[1, 1])
    df['Total Profit Contribution ($)'].hist(bins=30, ax=axes[1, 2])
    pd.plotting.scatter_matrix(df[['Initial Loan ($)', 'Days Since Last Loan', 'On-time Payments (%)']], ax=axes[2, 0])
    sns.barplot(x=df.groupby('Risk_Category')['Uptake_Probability'].mean().index, y=df.groupby('Risk_Category')['Uptake_Probability'].mean().values, ax=axes[2, 2])
    plt.suptitle('Dataset Overview')
    fig.savefig(outpath, bbox_inches='tight')
    plt.close(fig)


def fig2_model_performance(df: pd.DataFrame, recommendations: pd.DataFrame, outpath: str):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    df.groupby('Risk_Category')['Default_Probability'].mean().plot(kind='bar', ax=axes[0, 0])
    df[df['Expected_Value'] > 0]['Expected_Value'].hist(bins=50, ax=axes[0, 1])
    df.sample(n=min(2000, len(df))).plot.scatter(x='Default_Probability', y='Expected_Value', ax=axes[0, 2])
    df.groupby('Risk_Category')['Customer_NPV'].mean().plot(kind='bar', ax=axes[1, 0])
    df.groupby('Risk_Category')['Credit_Score_Proxy'].plot(kind='line', ax=axes[1, 1])
    axes[1, 2].bar(['Total Eligible', 'Approved for Increase', 'High Risk Approved'], [df['Eligible'].sum(), len(recommendations), len(recommendations[recommendations['Risk_Category']=='Sub-Prime'])])
    plt.suptitle('Model Performance and Risk Analysis')
    fig.savefig(outpath, bbox_inches='tight')
    plt.close(fig)


def fig3_simulation_results(simulations: pd.DataFrame, outpath: str):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for i, cat in enumerate(simulations['risk_category'].unique()[:3]):
        simulations[simulations['risk_category']==cat]['net_value'].hist(bins=50, ax=axes[0, 0])
    simulations.groupby('risk_category')['defaults'].mean().plot(kind='bar', ax=axes[0, 2])
    simulations.groupby('customer_id')[['total_profit','total_losses']].mean().plot(kind='scatter', x='total_profit', y='total_losses', ax=axes[1, 0])
    simulations.groupby('risk_category')['increases_granted'].mean().plot(kind='bar', ax=axes[1, 1])
    simulations.groupby('risk_category')['net_value'].describe()[['mean','50%','std']].plot(kind='bar', ax=axes[1, 2])
    plt.suptitle('Monte Carlo Simulation Results')
    fig.savefig(outpath, bbox_inches='tight')
    plt.close(fig)


def fig4_optimization_strategy(recommendations: pd.DataFrame, outpath: str):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    recommendations['Recommended_Increases'].value_counts().sort_index().plot(kind='bar', ax=axes[0, 0])
    recommendations.groupby('Recommended_Increases')['Total_Expected_Value'].mean().plot(ax=axes[0, 1])
    recommendations['Risk_Category'].value_counts().plot(kind='pie', ax=axes[0, 2])
    recommendations['On-time Payments (%)'].hist(bins=30, ax=axes[1, 0])
    recommendations['Initial Loan ($)'].hist(bins=30, ax=axes[1, 1])
    recommendations['Default_Probability'].hist(bins=30, ax=axes[1, 2])
    plt.suptitle('Optimization Strategy Results')
    fig.savefig(outpath, bbox_inches='tight')
    plt.close(fig)


def main():
    df, recommendations, simulations = load_results()
    out1 = os.path.join(WORKDIR, 'fig1_dataset_overview.png')
    out2 = os.path.join(WORKDIR, 'fig2_model_performance.png')
    out3 = os.path.join(WORKDIR, 'fig3_monte_carlo.png')
    out4 = os.path.join(WORKDIR, 'fig4_optimization_strategy.png')
    fig1_dataset_overview(df, out1)
    fig2_model_performance(df, recommendations, out2)
    fig3_simulation_results(simulations, out3)
    fig4_optimization_strategy(recommendations, out4)
    print('Saved figures:', out1, out2, out3, out4)


if __name__ == '__main__':
    main()
