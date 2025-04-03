import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
csv_file = 'future-gc00-daily-prices.csv'
data = pd.read_csv(
    csv_file,
    parse_dates=['Date'],
    dayfirst=True,
    index_col='Date'
).sort_index()

# Data Cleaning in One Step
data['Close'] = (
    data['Close']
    .astype(str)
    .str.replace(',', '', regex=True)
    .astype(float)
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
)

# Plot
# plt.figure(figsize=(12, 6))
# plt.plot(data.index, data['Close'], label='Close Price', color='b')
# plt.title('Gold Futures Daily Prices')
# plt.xlabel('Date')
# plt.ylabel('Close Price')
# plt.legend()
# plt.show()

# Define figure and axis
fig, ax = plt.subplots(figsize=(12, 6))

# Plot with a clean style
ax.plot(data.index, data['Close'], color='blue', alpha=0.7,
        marker='o', markersize=4, markeredgecolor='black',
        markerfacecolor='white', label='Close Price')

# Titles and labels with better font sizes
ax.set_title('Gold Futures Daily Prices', fontsize=14,
             fontweight='bold', color='darkblue')
ax.set_xlabel('Date', fontsize=12, fontweight='bold')
ax.set_ylabel('Close Price (USD)', fontsize=12, fontweight='bold')

# Improve readability of x-axis labels
ax.tick_params(axis='x', rotation=45)  # Rotate dates for better visibility
ax.grid(True, linestyle='--', alpha=0.6)  # Add a subtle grid

# Add legend with better styling
ax.legend(frameon=True, fontsize=11, loc='best')

# Improve layout to prevent label cutoff
plt.tight_layout()

# Show plot
plt.show()
