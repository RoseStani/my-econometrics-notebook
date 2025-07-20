# 1. Basic Commands

This section covers the essential Python commands to get familiar with the Wages dataset from the `wooldridge` package. The focus is on understanding the structure, summary statistics, and how to explore subgroups of the data.

---

## 1.1 Import Libraries and Load Dataset

```python
import pandas as pd
import wooldridge

# Load the Wages dataset
df = wooldridge.data('wage1')

# View the first few rows
df.head()
```
We’re using the wooldridge package to load the wage1 dataset directly. This gives us quick access to real-world economic survey data.

## 1.2 Get Basic Statistics

```python
# Mean, Median, Min, Max
df['wage'].mean()
df['wage'].median()
df['wage'].min()
df['wage'].max()
```
These functions help understand the central tendency and spread of wages.

.mean() and .median() show average values
.min() and .max() give the range of values in the column

## 1.3 Describe All Columns
```python
df.describe()
```
This provides a summary of all numerical columns including count, mean, standard deviation, min, and percentiles — a quick overview of the dataset's numerical structure.

## 1.4 Check Unique & Count of Unique Values
```python
df['educ'].unique()
df['educ'].nunique()
```
.unique() returns the list of distinct values in a column
.nunique() counts how many unique values are there — useful for understanding categorical variables

## 1.5 Index-Based Row/Column Selection
```python
# View first 5 rows and first 3 columns
df.iloc[0:5, 0:3]
```
.iloc[] is used for index-based selection: The first argument selects rows, the second selects columns
0:5 means rows 0 to 4 ; 0:3 means columns 0 to 2

## 1.6 Filter Based on Conditions (e.g., Gender)
```python
# Filter only male observations (male == 1)
df_male = df[df['male'] == 1]

# Filter only female observations (male == 0)
df_female = df[df['male'] == 0]
```
We use boolean conditions to filter data:

If male == 1, the respondent is male
If male == 0, the respondent is female
This helps us analyze data based on gender-specific subsets.

---
## Summary

These are some of the most important and frequently used commands I’ve come across while working with econometric datasets. They help in quickly understanding the structure of data, spotting key patterns, and preparing the dataset for deeper analysis. Keeping it simple but meaningful was my goal here — easy to understand and easy to revisit whenever I need a quick brush-up.

---

[Next: Simple & Multiple Regression →](../2_simple_multiple_regression/README.md)







