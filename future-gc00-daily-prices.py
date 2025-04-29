import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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

# Define figure and axis
fig, ax = plt.subplots(figsize=(12, 6))

# Plot with a clean style
ax.plot(data.index, data['Close'], 'o-', ms=4, mfc='w',
        mec='k', color='b', alpha=0.7, label='Close Price')

# Titles and labels with better font sizes
ax.set(title='Gold Futures Daily Prices',
       xlabel='Date', ylabel='Close Price (USD)')

# Format x-axis date ticks to avoid clutter
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

# Grid and legend
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(fontsize=11)

# Improve layout to prevent label cutoff
plt.tight_layout()

# Show plot
plt.show()
