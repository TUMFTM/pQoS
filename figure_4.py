import matplotlib.pyplot as plt
import pickle
from tum_color import color_pallet, TUMColor

import pdb

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
# Load results
# UL
with open("out/final_results/ul_unseen_ref.pkl", "rb") as f:
    ul_unseen_ref = pickle.load(f)

with open("out/final_results/ul_unseen_gt.pkl", "rb") as f:
    ul_unseen_gt = pickle.load(f)

with open("out/final_results/ul_unseen_old.pkl", "rb") as f:
    ul_unseen_old = pickle.load(f)

with open("out/final_results/ul_seen_ref.pkl", "rb") as f:
    ul_seen_ref = pickle.load(f)

with open("out/final_results/ul_seen_gt.pkl", "rb") as f:
    ul_seen_gt = pickle.load(f)

with open("out/final_results/ul_seen_old.pkl", "rb") as f:
    ul_seen_old = pickle.load(f)

# Latency
with open("out/final_results/latency_unseen_ref.pkl", "rb") as f:
    latency_unseen_ref = pickle.load(f)

with open("out/final_results/latency_unseen_gt.pkl", "rb") as f:
    latency_unseen_gt = pickle.load(f)

with open("out/final_results/latency_unseen_old.pkl", "rb") as f:
    latency_unseen_old = pickle.load(f)

with open("out/final_results/latency_seen_ref.pkl", "rb") as f:
    latency_seen_ref = pickle.load(f)

with open("out/final_results/latency_seen_gt.pkl", "rb") as f:
    latency_seen_gt = pickle.load(f)

with open("out/final_results/latency_seen_old.pkl", "rb") as f:
    latency_seen_old = pickle.load(f)

# pdb.set_trace()

latency_seen_ref.loc[latency_seen_ref['Latency_step1'] > 175, 'Latency_step1'] = 175
latency_seen_gt.loc[latency_seen_gt['Latency_step1'] > 175, 'Latency_step1'] = 175
latency_seen_old.loc[latency_seen_old['Latency_step1'] > 175, 'Latency_step1'] = 175
latency_seen_ref.loc[latency_seen_ref['Latency_step1'] < 10, 'Latency_step1'] = 10
latency_seen_gt.loc[latency_seen_gt['Latency_step1'] < 10, 'Latency_step1'] = 10
latency_seen_old.loc[latency_seen_old['Latency_step1'] < 10, 'Latency_step1'] = 10

latency_unseen_ref.loc[latency_unseen_ref['Latency_step1'] > 175, 'Latency_step1'] = 175
latency_unseen_gt.loc[latency_unseen_gt['Latency_step1'] > 175, 'Latency_step1'] = 175
latency_unseen_old.loc[latency_unseen_old['Latency_step1'] > 175, 'Latency_step1'] = 175
latency_unseen_ref.loc[latency_unseen_ref['Latency_step1'] < 10, 'Latency_step1'] = 10
latency_unseen_gt.loc[latency_unseen_gt['Latency_step1'] < 10, 'Latency_step1'] = 10
latency_unseen_old.loc[latency_unseen_old['Latency_step1'] < 10, 'Latency_step1'] = 10

latency_seen_ref['Latency_step1_smooth'] = latency_seen_ref['Latency_step1'].rolling(window=5, min_periods=1).mean()
latency_seen_gt['Latency_step1_smooth'] = latency_seen_gt['Latency_step1'].rolling(window=5, min_periods=1).mean()
latency_seen_old['Latency_step1_smooth'] = latency_seen_old['Latency_step1'].rolling(window=5, min_periods=1).mean()
latency_unseen_ref['Latency_step1_smooth'] = latency_unseen_ref['Latency_step1'].rolling(window=5, min_periods=1).mean()
latency_unseen_gt['Latency_step1_smooth'] = latency_unseen_gt['Latency_step1'].rolling(window=5, min_periods=1).mean()
latency_unseen_old['Latency_step1_smooth'] = latency_unseen_old['Latency_step1'].rolling(window=5, min_periods=1).mean()

ul_seen_ref['UL_step1_smooth'] = ul_seen_ref['UL_step1'].rolling(window=5, min_periods=1).mean()
ul_seen_gt['UL_step1_smooth'] = ul_seen_gt['UL_step1'].rolling(window=5, min_periods=1).mean()
ul_seen_old['UL_step1_smooth'] = ul_seen_old['UL_step1'].rolling(window=5, min_periods=1).mean()
ul_unseen_ref['UL_step1_smooth'] = ul_unseen_ref['UL_step1'].rolling(window=5, min_periods=1).mean()
ul_unseen_gt['UL_step1_smooth'] = ul_unseen_gt['UL_step1'].rolling(window=5, min_periods=1).mean()
ul_unseen_old['UL_step1_smooth'] = ul_unseen_old['UL_step1'].rolling(window=5, min_periods=1).mean()

# Plot

fig, axes = plt.subplots(2, 2, figsize=(7.16, 4))

tp_regions = [(130, 145), (280, 290), (380, 395)]
fn_regions = [(525, 535)]
fp_regions = []

# UL seen
axes[0, 0].plot(
    ul_seen_gt.index, ul_seen_gt['UL_step1_smooth'], 
    label='ground truth', color=color_pallet[0], linewidth=0.8
)
axes[0, 0].plot(
    ul_seen_ref.index, ul_seen_ref['UL_step1_smooth'], 
    label='with historic data', color=color_pallet[1], linewidth=0.8
)
axes[0, 0].plot(
    ul_seen_old.index, ul_seen_old['UL_step1_smooth'], 
    label='without historic data', color=color_pallet[2], linewidth=0.8
)
for start, end in tp_regions:
    axes[0, 0].axvspan(start, end, color=TUMColor.TUM_GREEN, alpha=0.4)
    axes[0, 0].text((start + end) / 2, axes[0, 0].get_ylim()[0] + 1, 'TP',
                    ha='center', va='bottom', color=TUMColor.TUM_BLUE)

for start, end in fp_regions:
    axes[0, 0].axvspan(start, end, color=TUMColor.TUM_ORANGE, alpha=0.4)
    axes[0, 0].text((start + end) / 2, axes[0, 0].get_ylim()[0] + 1, 'FP',
                    ha='center', va='bottom', color=TUMColor.TUM_RED)

for start, end in fn_regions:
    axes[0, 0].axvspan(start, end, color=TUMColor.TUM_BLUE, alpha=0.4)
    axes[0, 0].text((start + end) / 2, axes[0, 0].get_ylim()[0] + 1, 'FN',
                    ha='center', va='bottom', color=TUMColor.TUM_RED)
# axes[0].legend(loc='lower right')
axes[0, 0].set_title('Route 1')
axes[0, 0].set_ylabel("Uplink data-rate in Mbps")
# axes[0, 0].set_xlimit
axes[0, 0].grid(True, linestyle='--', alpha=0.6)




tp_regions = [(80,85)]
fn_regions = []
fp_regions = []
# UL unseen
axes[0, 1].plot(
    ul_unseen_gt.index, ul_unseen_gt['UL_step1_smooth'], 
    label='ground truth', color=color_pallet[0], linewidth=1
)
axes[0, 1].plot(
    ul_unseen_ref.index, ul_unseen_ref['UL_step1_smooth'], 
    label='with historic data', color=color_pallet[1], linewidth=1
)
axes[0, 1].plot(
    ul_unseen_old.index, ul_unseen_old['UL_step1_smooth'], 
    label='without historic data', color=color_pallet[2], linewidth=1
)
for start, end in tp_regions:
    axes[0, 1].axvspan(start, end, color=TUMColor.TUM_GREEN, alpha=0.4)
    axes[0, 1].text((start + end) / 2, axes[0, 1].get_ylim()[1] * 0.9, 'TP',
                    ha='center', va='bottom', color=TUMColor.TUM_BLUE)

for start, end in fp_regions:
    axes[0, 1].axvspan(start, end, color=TUMColor.TUM_ORANGE, alpha=0.4)
    axes[0, 1].text((start + end) / 2, axes[0, 1].get_ylim()[1] * 0.9, 'FP',
                    ha='center', va='bottom', color=TUMColor.TUM_RED)

for start, end in fn_regions:
    axes[0, 1].axvspan(start, end, color=TUMColor.TUM_BLUE, alpha=0.4)
    axes[0, 1].text((start + end) / 2, axes[0, 1].get_ylim()[1] * 0.9, 'FN',
                    ha='center', va='bottom', color=TUMColor.TUM_RED)
# axes[1].legend(loc='upper right')
axes[0, 1].set_title('Route 2')
# axes[1, 0].set_ylabel("in Mbps")
axes[0, 1].grid(True, linestyle='--', alpha=0.6)


tp_regions = [(700, 725)]
fn_regions = []
fp_regions = [(50,75)]
# Latency seen
axes[1, 0].plot(
    latency_seen_gt.index, latency_seen_gt['Latency_step1_smooth'], 
    label='ground truth', color=color_pallet[0], linewidth=0.8
)
axes[1, 0].plot(
    latency_seen_ref.index, latency_seen_ref['Latency_step1_smooth'], 
    label='with historic data', color=color_pallet[1], linewidth=0.8
)
axes[1, 0].plot(
    latency_seen_old.index, latency_seen_old['Latency_step1_smooth'], 
    label='without historic data', color=color_pallet[2], linewidth=0.8
)
for start, end in tp_regions:
    axes[1, 0].axvspan(start, end, color=TUMColor.TUM_GREEN, alpha=0.4)
    axes[1, 0].text((start + end) / 2, axes[1, 0].get_ylim()[1] * 0.9, 'TP',
                    ha='center', va='bottom', color=TUMColor.TUM_BLUE)

for start, end in fp_regions:
    axes[1, 0].axvspan(start, end, color=TUMColor.TUM_ORANGE, alpha=0.4)
    axes[1, 0].text((start + end) / 2, axes[1, 0].get_ylim()[1] * 0.9, 'FP',
                    ha='center', va='bottom', color=TUMColor.TUM_RED)

for start, end in fn_regions:
    axes[1, 0].axvspan(start, end, color=TUMColor.TUM_BLUE, alpha=0.4)
    axes[1, 0].text((start + end) / 2, axes[1, 0].get_ylim()[1] * 0.9, 'FN',
                    ha='center', va='bottom', color=TUMColor.TUM_RED)
# axes[1, 0].legend(loc='upper right')
axes[1, 0].set_ylabel("Latency in ms")
axes[1, 0].set_xlabel("Data point index")
axes[1, 0].grid(True, linestyle='--', alpha=0.6)


tp_regions = []
fn_regions = [(150, 200), (550, 575)]
fp_regions = []
# Latency unseen
axes[1, 1].plot(
    latency_unseen_gt.index, latency_unseen_gt['Latency_step1_smooth'], 
    label='ground truth', color=color_pallet[0], linewidth=0.8
)
axes[1, 1].plot(
    latency_unseen_ref.index, latency_unseen_ref['Latency_step1_smooth'], 
    label='Prediction', color=color_pallet[1], linewidth=0.8
)
axes[1, 1].plot(
    latency_unseen_old.index, latency_unseen_old['Latency_step1_smooth'], 
    label='Prediction', color=color_pallet[2], linewidth=0.8
)

for start, end in tp_regions:
    axes[1, 1].axvspan(start, end, color=TUMColor.TUM_GREEN, alpha=0.4)
    axes[1, 1].text((start + end) / 2, axes[1, 1].get_ylim()[1] * 0.9, 'TP',
                    ha='center', va='bottom', color=TUMColor.TUM_BLUE)

for start, end in fp_regions:
    axes[1, 1].axvspan(start, end, color=TUMColor.TUM_ORANGE, alpha=0.4)
    axes[1, 1].text((start + end) / 2, axes[1, 1].get_ylim()[1] * 0.9, 'FP',
                    ha='center', va='bottom', color=TUMColor.TUM_RED)

for start, end in fn_regions:
    axes[1, 1].axvspan(start, end, color=TUMColor.TUM_BLUE, alpha=0.4)
    axes[1, 1].text((start + end) / 2, axes[1, 1].get_ylim()[1] * 0.9, 'FN',
                    ha='center', va='bottom', color=TUMColor.TUM_RED)
# axes[1, 1].set_ylabel("in")
axes[1, 1].set_xlabel("Data point index")
axes[1, 1].grid(True, linestyle='--', alpha=0.6)

# Create shared legend
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc='lower center',
    ncol=3,
    fontsize='small'
)

# pdb.set_trace()

plt.tight_layout()
fig.subplots_adjust(bottom=0.15)
# fig.set_title('Uplink data-rate prediction for Step 1')
plt.savefig("out/final_results/uplink_prediction_comparison.pdf")
# plt.show()