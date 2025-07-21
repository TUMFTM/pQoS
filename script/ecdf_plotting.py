import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tum_color import color_pallet

plt.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})
# Optional: define a color palette
# color_pallet = ['#1f77b4', '#ff7f0e']  # Line colors for File 1 and File 2

fig, axes = plt.subplots(2, 1, figsize=(3.5, 3))

# File paths
file_paths = ['data/difference_prediction/route_1_ul.txt', 'data/difference_prediction/route_2_ul.txt']
labels = ['Route 1', 'Route 2']


for idx, file_path in enumerate(file_paths):
    with open(file_path, 'r') as f:
        data = np.array([float(line.strip().split()[0]) for line in f if line.strip()])
    
    # Cap values over 100 for plotting but include them in calculations
    capped_data = data.copy()
    # capped_data[capped_data > 150] = 150
    sorted_data = np.sort(capped_data)
    
    cumulative_percentage = [(i + 1) / len(sorted_data) * 100 for i in range(len(sorted_data))]
    
    axes[0].plot(sorted_data, cumulative_percentage,
             color=color_pallet[idx], linestyle='-', linewidth=1, label=labels[idx])
    axes[0].set_xlim(left =0, right= 50)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend(loc="upper left")
    axes[0].set_xlabel("Uplink data-rate in Mbps")
    axes[0].set_ylabel("Percentage in %")
    axes[0].set_title("Empirical Cumulative Distribution Function")
    
# File paths
file_paths = ['data/difference_prediction/route_1_latency.txt', 'data/difference_prediction/route_2_latency.txt']
labels = ['Route 1', 'Route 2']


for idx, file_path in enumerate(file_paths):
    with open(file_path, 'r') as f:
        data = np.array([float(line.strip().split()[0]) for line in f if line.strip()])
    
    # Cap values over 100 for plotting but include them in calculations
    capped_data = data.copy()
    # capped_data[capped_data > 150] = 150
    sorted_data = np.sort(capped_data)
    
    cumulative_percentage = [(i + 1) / len(sorted_data) * 100 for i in range(len(sorted_data))]
    
    axes[1].plot(sorted_data, cumulative_percentage,
             color=color_pallet[idx], linestyle='-', linewidth=1, label=labels[idx])
    axes[1].set_xlim(left =0, right=140)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend(loc="upper left")
    axes[1].set_xlabel("Latency in ms")
    axes[1].set_ylabel("Percentage in %")

# plt.title("Empirical Cumulative Distribution Function")

# plt.legend()
plt.tight_layout()
plt.savefig("out/final_results/ecdf.pdf")
