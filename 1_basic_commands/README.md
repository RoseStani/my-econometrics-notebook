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

