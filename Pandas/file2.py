import pandas as pd

data = {
    'Name': ['Achini', 'Nadeesha', 'Kasun'],
    'Age': [23, None, 27],
    'Salary': [50000, 60000, None]
}

df = pd.DataFrame(data)

# Fill missing Age with average
df['Age'].fillna(df['Age'].mean(), inplace=True)

# Fill missing Salary with 0
df['Salary'].fillna(0, inplace=True)

print(df)
