# Part 3: Hypothesis Testing & ANOVA – Validating Our Wage Model

### Why Are We Doing This?

After building a multiple regression model to explain wages, we must validate:

- Do the predictors (e.g., `educ`, `exper`, `tenure`) truly affect wages?
- Can we remove or combine variables without losing explanatory power?
- Is the model statistically sound?

We use:

- `f_test()` for **custom hypothesis testing** - testing independent variables of a model
- `anova_lm()` to **compare nested models** and check whether adding variables improves the model - basically a comparison between 2 different models altogether

---

## 1. F-Test using `model.f_test()` – Testing Linear Hypotheses

We use `model.f_test()` when we want to test if certain **linear constraints** on coefficients are valid.

```python
import statsmodels.api as sm
import statsmodels.formula.api as smf
from wooldridge import data

# Load the dataset
df = data('wage')

# Fit a full wage model
model = smf.ols('wage ~ educ + exper + tenure', data=df).fit()
```
1.1 Example 1: Is educ Significant?

Null Hypothesis (H₀): β₁ = 0 (education has no effect on wage)
```python
model.f_test('educ = 0')
```
1.2 Example 2: Are educ and exper Both Insignificant?

H₀: β₁ = 0, β₂ = 0
```python
model.f_test('educ = 0, exper = 0')
```
1.3 Example 3: Is the Effect of educ Equal to That of tenure?

H₀: β₁ = β₃
```python
model.f_test('educ = tenure')
```
1.4 Example 4: Do Combined Effects of educ + exper Equal 1?

H₀: β₁ + β₂ = 1
```python
model.f_test('educ + exper = 1')
```
🧾 Interpretation:
If p-value < 0.05, reject the null → the constraint does not hold.
If p-value > 0.05, fail to reject → the constraint may hold.

---


## 2. ANOVA: Comparing Nested Models

We use anova_lm() to compare a reduced model with a full model to see if adding variables improves performance.
```python
from statsmodels.stats.anova import anova_lm

# Reduced model (without 'tenure')
reduced_model = smf.ols('wage ~ educ + exper', data=df).fit()

# Full model (adds 'tenure')
full_model = smf.ols('wage ~ educ + exper + tenure', data=df).fit()

# Compare using ANOVA
anova_results = anova_lm(reduced_model, full_model)
print(anova_results)
```
##  Output Interpretation

| **Term**     | **Meaning**                                                        |
|--------------|---------------------------------------------------------------------|
| `df_resid`   | Residual degrees of freedom                                         |
| `ssr`        | Sum of squared residuals                                            |
| `df_diff`    | Difference in degrees of freedom between models                     |
| `ss_diff`    | Change in SSR (how much the residual sum changed between models)    |
| `F`          | F-statistic value used to test model comparison                     |
| `Pr(>F)`     | p-value for the F-test                                               |

- If **p < 0.05** → full model is significantly better → adding `tenure` improves the model  
- If **p > 0.05** → reduced model is sufficient → `tenure` does not significantly improve the model
---

## Summary
In this part, we validated our wage regression model using hypothesis testing and ANOVA. We applied F-tests to check if individual or combined predictors like educ, exper, and tenure significantly affect wages. Using model.f_test(), we tested specific coefficient constraints, while anova_lm() helped us compare full and reduced models to see if adding variables improves model performance. These tests ensure our model is both statistically sound and interpretable.

---

Next: [Part 4 – Multicollinearity](../part4_multicollinearity/README.md)









