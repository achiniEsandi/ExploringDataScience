import pandas as pd
import numpy as np

# Step 1: Create sample data
data = {
    'Feature_A': [10, 20, 30, 40, 50],
    'Feature_B': [100, 200, 300, 400, 500],  # Highly correlated with Feature_A
    'Feature_C': [5, 3, 4, 2, 1],            # Weakly correlated with target
    'Target':     [20, 40, 60, 80, 100]
}

df = pd.DataFrame(data)

# Step 2: View correlation matrix
correlation_matrix = df.corr()
print("🔍 Full Correlation Matrix:\n")
print(correlation_matrix)

# Step 3: Check correlation of features with the target only
target_corr = correlation_matrix['Target'].drop('Target')
print("\n🎯 Correlation with Target:\n")
print(target_corr)

# Step 4: Check correlation between features
print("\n⚠️ Correlation between Features:\n")
feature_corr = df[['Feature_A', 'Feature_B', 'Feature_C']].corr()
print(feature_corr)

# Step 5: Decide what to drop
# Let's say we drop Feature_B because it's highly correlated with Feature_A
# and has equal correlation with the Target
print("\n✅ Final Feature Set (dropping Feature_B):\n")
final_df = df.drop(columns=['Feature_B'])
print(final_df)
