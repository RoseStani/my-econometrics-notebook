# Part 5 – Heteroskedasticity

##  What is Heteroskedasticity?

In a classical linear regression model, one key assumption is that the **variance of the error term (u)** is constant across all observations (homoskedasticity).  
However, **heteroskedasticity** occurs when this assumption is violated — the **variance of the error term varies** across the data.

This causes:

- **Inefficient estimates** (even though they remain unbiased)
- **Invalid standard errors**, making **t-tests and F-tests unreliable**
- Misleading confidence intervals and p-values

---
##  How to Detect Heteroskedasticity?

We can inspect heteroskedasticity in **two main ways**:

1. **Visually** – by plotting residuals against fitted values.
2. **Statistically** – by conducting formal tests like:
   - Breusch-Pagan Test
   - White's Test

Both methods are used together for a more reliable diagnosis.
---

## Why is it important to detect and fix it?

- If present, your inferences about model parameters (hypothesis testing, confidence intervals) will be **misleading**.
- Correcting it ensures **robustness** and **valid inference**.

---

## Dataset Used

We’ll use the `wage1` dataset from the `wooldridge` package to Load data

```python
from wooldridge import data
import pandas as pd
df = data('wage1')
```
## 1.Visual Inspection of Heteroskedasticity

Let's run a basic regression and check the residual plot.
```python
import statsmodels.api as sm
import statsmodels.formula.api as smf
# Fit a basic regression model
model = smf.ols('wage ~ educ + exper + tenure', data=df).fit()

# Plotting residuals vs. fitted values
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

fitted_vals = model.fittedvalues
residuals = model.resid

plt.figure(figsize=(8,5))
sns.scatterplot(x=fitted_vals, y=residuals, alpha=0.7)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values')
plt.show()
```
If residuals fan out or funnel in (like a cone shape), this suggests heteroskedasticity.
A random cloud indicates homoskedasticity.

## 2.Statistical Tests for Heteroskedasticity

i) Breusch-Pagan Test
```python
from statsmodels.stats.diagnostic import het_breuschpagan

bp_test = het_breuschpagan(model.resid, model.model.exog)
labels = ['Lagrange multiplier statistic', 'p-value', 
          'f-value', 'f p-value']
dict(zip(labels, bp_test))
```
Tests whether variance of residuals depends on regressors.
if p-value < 0.05 → Heteroskedasticity is present

ii) White Test
```python
from statsmodels.stats.diagnostic import het_white

white_test = het_white(model.resid, model.model.exog)
labels = ['Test Statistic', 'Test Statistic p-value', 
          'F-Statistic', 'F-Test p-value']
dict(zip(labels, white_test))
```
More general test; doesn’t assume a specific form of heteroskedasticity.
Again,if p-value < 0.05 → Heteroskedasticity is present.

---
## 3.Fixing Heteroskedasticity

i) Robust Standard Errors
```python
robust_model = model.get_robustcov_results(cov_type='HC1')
print(robust_model.summary())
```
Adjusts for heteroskedasticity without changing coefficients

ii) Weighted Least Squares (WLS)
```python
# Suppose residuals increase with experience, try weights ~ 1/exper
weights = 1 / df['exper']
wls_model = sm.WLS(df['wage'], model.model.exog, weights=weights).fit()
print(wls_model.summary())
```
Changes estimation approach by weighting observations

---
## Summary

Heteroskedasticity refers to a violation of the classical OLS assumption where the variance of the error term is not constant across observations. Using the wage1 dataset, we examined this issue by visually inspecting a residuals vs fitted values plot and conducting formal tests like Breusch-Pagan and White. Though heteroskedasticity does not bias the OLS coefficients, it makes standard errors unreliable, leading to invalid statistical inference. To address it, we applied robust standard errors and Weighted Least Squares (WLS), ensuring more accurate and trustworthy regression results.
