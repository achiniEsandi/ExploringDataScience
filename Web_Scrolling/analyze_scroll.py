import pandas as pd

# Load the scroll data from the JSON file
df = pd.read_json("C:/Users/Lenovo/Desktop/ExploringDataScience/Web_Scrolling/scroll_data.json")


# Convert the timestamp to readable datetime format
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

# Display the dataset
print("Scroll Data:\n", df)

# Average scroll percentage
avg_scroll = df['maxScrollPercent'].mean()
print(f"\n📊 Average Scroll Depth: {avg_scroll:.2f}%")

# Max scroll per user
max_scroll_per_user = df.groupby('userId')['maxScrollPercent'].max()
print("\n📈 Max Scroll % per User:")
print(max_scroll_per_user)

# Save the results to CSV
max_scroll_per_user.to_csv("max_scroll_per_user.csv")

# Plot a histogram of scroll percentages
import matplotlib.pyplot as plt

plt.hist(df['maxScrollPercent'], bins=5, edgecolor='black')
plt.title('Scroll Depth Distribution')
plt.xlabel('Scroll %')
plt.ylabel('Number of Sessions')
plt.grid(True)
plt.show()
