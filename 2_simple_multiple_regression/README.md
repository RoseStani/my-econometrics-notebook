# 2. Simple & Multiple Regression

In this section, we dive into regression models to understand what drives **wage** — our dependent variable. We're essentially testing which factors (independent variables) might explain or influence someone's wage level.

In econometrics, wage is often seen as a **function of education, experience, gender, etc.** — and regression helps us quantify this relationship.

---

## 2.1 Simple Linear Regression

```python
import pandas as pd
import statsmodels.api as sm
# Wage as a function of education
X = df[['educ']]
y = df['wage']

# Add constant (for intercept)
X = sm.add_constant(X)

# Fit the model
model_simple = sm.OLS(y, X).fit()
print(model_simple.summary())

```
Explanation:
This is a simple linear regression with wage as the dependent variable and educ (years of education) as the only predictor.
We ask: "How much does one additional year of education increase wage?"

Use this when you want to test the effect of one key variable on the outcome.

---
## 2.2 Multiple Linear Regression

### 📘 Method 1: Using `statsmodels.api` (X, y style)
```python
# Wage as a function of multiple predictors
X = df[['educ', 'exper', 'tenure']]
y = df['wage']

X = sm.add_constant(X)
model_multiple = sm.OLS(y, X).fit()
print(model_multiple.summary())
```
This method allows manual control over independent variables. It’s useful when you want to select columns explicitly and add transformations before modeling.

### 📘 Method 2: Using statsmodels.formula.api (formula style)
```python
import statsmodels.formula.api as smf

# Fit the model using formula notation
model_formula = smf.ols('wage ~ educ + exper + tenure', data=df).fit()
print(model_formula.summary())
```
This method uses a formula string:

'dependent_variable ~ independent1 + independent2 + ...'
It’s more readable, and useful when doing interaction terms, polynomial terms, or categorical variables.

Explanation:
Here, we’re adding more explanatory variables — like experience and tenure — to better explain wage.
This is useful when you're working with multiple potential influences on the dependent variable.

Typical questions:

Does education still matter if we control for experience?
How much does tenure affect wage, holding education constant?

---

## 2.3 Log-Log Regression (Constant Elasticity)

```python
# Use log-log transformation
import numpy as np

df['log_wage'] = np.log(df['wage'])
df['log_educ'] = np.log(df['educ'])

X = df[['log_educ']]
y = df['log_wage']

X = sm.add_constant(X)
model_loglog = sm.OLS(y, X).fit()
print(model_loglog.summary())
```
Explanation:
In a log-log model, both the dependent and independent variables are logged.
This is used when we want to interpret coefficients as elasticities:

💡 In a log-log model, the coefficient tells us the percentage change in wage for a 1% change in education.
Use this when you're dealing with proportional or percentage-based interpretation.

---

## Summary

Simple regression is ideal when testing one primary influence.
Multiple regression helps when we want to control for several factors simultaneously.
Log-log models are best when we expect constant proportional relationships (elasticities).

---

[Next: F-tests and ANOVA →](../3_f_tests_anova/README.md)

