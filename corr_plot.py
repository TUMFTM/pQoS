import pandas as pd
from scipy.stats import pearsonr

import pdb

# Load the CSV file
csv_path = 'data/delay.csv'  # Replace with your file path
df = pd.read_csv(csv_path)


# Extract the two rows as vectors
ul = df["Latency"]
sinr = df ["RSRQ"]

# pdb.set_trace()
# Drop rows where either UL or SINR is NaN
valid = (~ul.isna()) & (~sinr.isna())
ul_valid = ul[valid]
sinr_valid = sinr[valid]

# Calculate Pearson correlation
corr, p_value = pearsonr(ul_valid, sinr_valid)

# Output the result
print(f"Pearson correlation between UL and SINR: {corr:.4f}")
print(f"P-value: {p_value:.4e}")
