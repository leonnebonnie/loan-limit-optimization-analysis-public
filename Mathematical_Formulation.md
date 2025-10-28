# MATHEMATICAL FORMULATION
## Loan Limit Optimization Model

### Problem Statement
Determine the optimal loan limit increase allocation for a portfolio of customers to maximize expected profitability while managing risk under operational constraints.

---

## 1. NOTATION

### Sets and Indices
- **C**: Set of all customers, indexed by i ∈ C
- **T**: Time periods (quarters), indexed by t ∈ {0, 1, 2, 3}
- **S**: Risk states S = {Prime, Near-Prime, Sub-Prime}, indexed by s

### Parameters
- **π**: Profit per loan increase = $40
- **r**: Annual discount rate = 0.19 (quarterly rate = 0.0475)
- **D_min**: Minimum days for eligibility = 60
- **N_max**: Maximum increases per customer per year = 6
- **α_max**: Maximum proportion of high-risk approvals = 0.25
- **L_i**: Initial loan amount for customer i
- **d_i**: Days since last loan for customer i
- **p_i**: On-time payment rate for customer i
- **n_i**: Historical number of increases for customer i

### Decision Variables
- **x_i**: Number of loan increases allocated to customer i (integer, 0 ≤ x_i ≤ 6)
- **y_i**: Binary indicator whether customer i receives any increase (y_i ∈ {0,1})

### Derived Variables
- **P^uptake_i**: Probability customer i accepts an offer
- **P^default_i**: Probability customer i defaults
- **s_i**: Risk state of customer i
- **E_i**: Expected value per increase for customer i

---

## 2. OBJECTIVE FUNCTION

Maximize total expected net present value:

```
max Z = Σ_{i∈C} Σ_{t=0}^{T-1} [E_i × x_i / (1 + r)^t]
```

Where expected value per customer is:

```
E_i = π × P^uptake_i × (1 - P^default_i) - L_i × 0.5 × P^uptake_i × P^default_i
```

---

## 3. CONSTRAINTS

### 3.1 Eligibility Constraint
```
y_i ≤ I(d_i ≥ D_min)    ∀i ∈ C
```
Where I(·) is the indicator function.

### 3.2 Maximum Increases Constraint
```
x_i ≤ N_max × y_i    ∀i ∈ C
```

### 3.3 Risk Allocation Constraint
```
Σ_{i: s_i = Sub-Prime} y_i ≤ α_max × Σ_{i∈C} y_i
```

### 3.4 Positive Expected Value Constraint
```
E_i × y_i ≥ 0    ∀i ∈ C
```

### 3.5 Capital Constraint (Optional)
```
Σ_{i∈C} L_i × x_i × 0.5 ≤ K
```
Where K is the available capital.

### 3.6 Linking Constraint
```
x_i ≥ y_i    ∀i ∈ C
```

---

## 4. PREDICTIVE MODELS

### 4.1 Uptake Probability Model

Gradient Boosting Classifier predicts uptake probability:

```
P^uptake_i = f_GB(L_i, d_i, p_i, CS_i, I_{pd}, R_{lp})
```

Where:
- **CS_i**: Credit score proxy = 0.6×p_i + 0.2×(d_i/d_max) + 0.2×(L_i/L_max)
- **I_{pd}**: Payment-Days interaction = p_i × d_i / 100
- **R_{lp}**: Loan-Payment ratio = L_i / (p_i + 1)

### 4.2 Default Probability Model

Risk-adjusted default probability:

```
P^default_i = σ(0.1 × (DR_i - 50)) × (1 + 0.05 × n_i)
```

Where:
- **σ(x)**: Sigmoid function = 1 / (1 + e^(-x))
- **DR_i**: Default risk score = 0.5×(100-p_i) + 0.3×(100-CS_i) + 0.2×(L_i/L_max×100)

### 4.3 Risk State Classification

```
s_i = {
    Prime,      if p_i ≥ 95
    Near-Prime, if 85 ≤ p_i < 95
    Sub-Prime,  if p_i < 85
}
```

---

## 5. MARKOV CHAIN MODEL

### 5.1 Transition Matrix

Quarterly risk state transitions:

```
P = [0.85  0.12  0.03]
    [0.15  0.70  0.15]
    [0.05  0.25  0.70]
```

### 5.2 State Transition Dynamics

Probability of transitioning from state s to state s' in period t:

```
P(s_i^{t+1} = s' | s_i^t = s) = P_{ss'}
```

### 5.3 Steady-State Distribution

Long-run proportion of customers in each risk state:

```
π = [0.4268, 0.3537, 0.2195]  // [Prime, Near-Prime, Sub-Prime]
```

Computed as the eigenvector of P^T corresponding to eigenvalue 1.

---

## 6. MONTE CARLO SIMULATION

### 6.1 Lifecycle Simulation Algorithm

For each customer i and simulation run k:

```
1. Initialize: s_i^0 = current risk state, V_i^k = 0, D_i^k = 0
2. For each quarter t = 0 to T-1:
   a. If t > 0, simulate state transition:
      s_i^t ~ Categorical(P_{s_i^{t-1},:})
   
   b. Adjust default probability:
      P^default_{i,t} = P^default_i × m(s_i^t)
      where m(Prime) = 0.8, m(Near-Prime) = 1.0, m(Sub-Prime) = 1.3
   
   c. Simulate uptake decision:
      u_{i,t}^k ~ Bernoulli(P^uptake_i)
   
   d. If u_{i,t}^k = 1:
      i. Simulate default:
         δ_{i,t}^k ~ Bernoulli(P^default_{i,t})
      
      ii. Update value:
         if δ_{i,t}^k = 1:
            V_i^k -= L_i × 0.5
            D_i^k += 1
         else:
            V_i^k += π

3. Return: (V_i^k, D_i^k)
```

### 6.2 Summary Statistics

After N simulations:

```
Mean Net Value: μ_i = (1/N) Σ_{k=1}^N V_i^k
Std Dev: σ_i = sqrt((1/(N-1)) Σ_{k=1}^N (V_i^k - μ_i)^2)
Default Rate: ρ_i = (1/N) Σ_{k=1}^N D_i^k
```

---

## 7. OPTIMIZATION ALGORITHM

### 7.1 Greedy Allocation with Constraints

```
Algorithm: Constrained Greedy Optimization
Input: Customer data, constraints
Output: Allocation x*, y*

1. Filter eligible customers: C' = {i ∈ C : d_i ≥ D_min}

2. Compute risk-adjusted score for each i ∈ C':
   RAS_i = E_i × (1 - P^default_i) × P^uptake_i

3. Sort C' in descending order of RAS_i

4. Initialize: x* = 0, y* = 0, H = 0, A = 0, E_total = 0

5. For each customer i in sorted C':
   a. Check risk constraint:
      if s_i = Sub-Prime and H ≥ α_max × A:
         continue
   
   b. Check value constraint:
      if E_i ≤ 0:
         continue
   
   c. Determine optimal allocation:
      x*_i = min(N_max, ⌊P^uptake_i × N_max⌋ + 1)
   
   d. Check capital constraint (if applicable):
      if L_i × x*_i × 0.5 > K - E_exposure:
         continue
   
   e. Accept allocation:
      y*_i = 1
      E_total += E_i × x*_i
      E_exposure += L_i × x*_i × 0.5
      A += 1
      if s_i = Sub-Prime: H += 1

6. Return (x*, y*, E_total)
```

---

## 8. NPV CALCULATION

### 8.1 Customer-Level NPV

```
NPV_i = Σ_{t=0}^{T-1} [E_i / (1 + r)^{t/4}]
```

Where t/4 converts quarters to years for annual discount rate r.

### 8.2 Portfolio NPV

```
NPV_total = Σ_{i∈C} NPV_i
```

---

## 9. KEY FORMULAS SUMMARY

### Expected Value
```
E_i = π × P^uptake_i × (1 - P^default_i) - L_i × 0.5 × P^uptake_i × P^default_i
```

### Credit Score Proxy
```
CS_i = 0.6 × p_i + 0.2 × (d_i / d_max) + 0.2 × (L_i / L_max)
```

### Default Probability
```
P^default_i = (1 / (1 + e^{-0.1×(DR_i - 50)})) × (1 + 0.05 × n_i)
```

### Risk-Adjusted Score
```
RAS_i = E_i × (1 - P^default_i) × P^uptake_i
```

### NPV with Quarterly Discounting
```
NPV = Σ_{t=0}^{3} [CF_t / (1 + 0.19)^{t/4}]
```

---

## 10. MODEL VALIDATION METRICS

### Classification Performance
- **AUC-ROC**: Area under ROC curve for uptake prediction
- **Accuracy**: Proportion of correct predictions

### Optimization Quality
- **Expected Value**: Total predicted profit
- **Risk Exposure**: Percentage of high-risk customers
- **Capital Efficiency**: Profit per dollar of capital deployed

### Simulation Convergence
- **Mean Convergence**: |μ_N - μ_{N-1}| < ε for large N
- **Variance Stability**: σ²_N stabilizes as N increases

---

## 11. COMPUTATIONAL COMPLEXITY

- **Uptake Prediction**: O(n × m × d) for gradient boosting
  - n: number of samples
  - m: number of trees
  - d: tree depth

- **Optimization**: O(n log n) for sorting + O(n) for greedy allocation
  - Total: O(n log n)

- **Monte Carlo Simulation**: O(n_cust × n_sim × T)
  - n_cust: number of customers
  - n_sim: simulations per customer
  - T: time periods

---

**Implementation Note**: This formulation has been implemented in Python using:
- scikit-learn (Gradient Boosting)
- numpy/pandas (data manipulation)
- Custom optimization algorithm
- Monte Carlo simulation engine

The complete code is available in `loan_optimization_analysis.py`.
