## Part 4: Multicollinearity 

### What is Multicollinearity?
Multicollinearity occurs when two or more independent variables in a regression model are highly correlated. This can distort the estimation of coefficients, making it difficult to determine the individual effect of each predictor on the response variable.

When predictors move together, it becomes harder for the model to isolate their unique impact.

### Why is it's Detection & Treatment Important?
Multicollinearity doesn’t reduce the model’s predictive power, but it weakens the statistical significance of individual predictors. This can lead to:

(I) Inflated standard errors
(II) Unstable coefficient estimates
(III) Misleading interpretations

### How to Detect Multicollinearity?
We use two main methods:

1. Correlation Matrix : To examine pairwise correlations between predictors
2. Variance Inflation Factor (VIF) : VIF = 1 → No correlation
                                     VIF between 1 and 5 → Moderate correlation
                                     VIF > 5 (or 10) → Potential multicollinearity problem

---

### 1. Load the Dataset
```python
import pandas as pd
df = pd.read_excel('wagesmicrodata.xls')
df.head()
```
Load the Excel dataset using pandas and preview the first few rows to ensure it's read correctly.

### 2. Correlation Heatmap (Initial Check)
```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(20,10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
```
This visualizes the correlation between numerical features.
Look for high correlation (e.g., > 0.8 or < -0.8) — especially in red.

### 3. Calculate Variance Inflation Factor (VIF)
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
X= df.drop(columns=['WAGE'])
X= sm.add_constant(X)

data_vif = pd.DataFrame()
data_vif['Variables'] = X.columns
data_vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
data_vif['Tolarence'] = 1/data_vif['VIF']
print(data_vif)
```
VIF detects how strongly a feature is correlated with the other features.
VIF ≈ 1: No multicollinearity
VIF > 5: Potential concern
VIF > 10: High multicollinearity → fix needed

The column 'WAGE' is dropped by default from the set of independent variables X because it is the dependent variable (also called the response or target).
In regression modeling, the dependent variable belongs on the left-hand side of the equation, while all the features used to predict it belong on the right-hand side

### 4. Drop Problematic Column
```python
Y= df.drop(columns=['WAGE','AGE'])
```
The 'age' column had a high VIF score.
Dropping it helps prevent distorted coefficients in regression.

### 5. Replot Heatmap (After Fix)
```python
plt.figure(figsize=(20,10))
sns.heatmap(Y.corr(), annot=True, cmap='coolwarm')
```
Confirms that correlation among variables has reduced after removing 'age'.

### 6. Recalculate VIF (To Recheck)
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
Y= sm.add_constant(Y)

data_vif = pd.DataFrame()
data_vif['Variables'] = Y.columns
data_vif['VIF'] = [variance_inflation_factor(Y.values, i) for i in range(Y.shape[1])]
data_vif['Tolarence'] = 1/data_vif['VIF']
print(data_vif)
```
Ensures remaining variables have acceptable VIF values (typically < 5).

### 7. Runs OLS Regression Now
```python
import statsmodels.formula.api as smf
model1= smf.ols('WAGE~C(OCCUPATION)+C(SECTOR)+C(UNION)+C(SEX)+C(MARR)+C(RACE)+C(SOUTH)+EDUCATION+EXPERIENCE+AGE',data= df).fit()
model1.summary()
```
Reruns regression on cleaned features.
The .summary() will show more stable and interpretable coefficients now.

---

### Conclusion
Multicollinearity was identified using heatmaps and VIF.
Problematic variable 'age' was removed.
Regression was re-run with reduced multicollinearity, improving model clarity.

---
Next:[Part 5 – Heteroskedasticity](../part5_heteroscedasticity/README.md)



