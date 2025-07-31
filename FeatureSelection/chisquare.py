import pandas as pd
from scipy.stats import chi2_contingency

# New dataset
data = {
    'Gender': ['Male', 'Male', 'Male', 'Male', 'Female', 'Female', 'Female', 'Female'],
    'Purchase': ['Yes', 'Yes', 'Yes', 'No', 'No', 'No', 'No', 'Yes']
}

df = pd.DataFrame(data)

# Create a contingency table
contingency = pd.crosstab(df['Gender'], df['Purchase'])
print("Contingency Table:")
print(contingency)

# Apply Chi-square test
chi2, p, dof, expected = chi2_contingency(contingency)

print("\nChi-square value:", chi2)
print("P-value:", p)
