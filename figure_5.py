import matplotlib.pyplot as plt
import pickle
import numpy as np

from tum_color import color_pallet, TUMColor

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import f1_score, precision_score, recall_score


def binary_classification_metrics(gt_series, pred_series, threshold, less_than=True):
    if less_than:
        gt_binary = (gt_series < threshold).astype(int)
        pred_binary = (pred_series < threshold).astype(int)
    else:
        gt_binary = (gt_series > threshold).astype(int)
        pred_binary = (pred_series > threshold).astype(int)

    f1 = f1_score(gt_binary, pred_binary)
    precision = precision_score(gt_binary, pred_binary)
    recall = recall_score(gt_binary, pred_binary)
    return f1, precision, recall

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

ul_unseen_ref_mae = []
ul_unseen_old_mae = []
ul_seen_ref_mae = []
ul_seen_old_mae = []

latency_unseen_ref_mae = []
latency_unseen_old_mae = []
latency_seen_ref_mae = []
latency_seen_old_mae = []

ul_unseen_ref_std = []
ul_unseen_old_std = []
ul_seen_ref_std = []
ul_seen_old_std = []

latency_unseen_ref_std = []
latency_unseen_old_std = []
latency_seen_ref_std = []
latency_seen_old_std = []

steps = 60

for s in range(1, steps + 1):
    ul_unseen_ref_mae.append(mean_absolute_error(ul_unseen_ref[f'UL_step{s}'], ul_unseen_gt[f'UL_step{s}']))
    ul_unseen_old_mae.append(mean_absolute_error(ul_unseen_old[f'UL_step{s}'], ul_unseen_gt[f'UL_step{s}']))
    ul_seen_ref_mae.append(mean_absolute_error(ul_seen_ref[f'UL_step{s}'], ul_seen_gt[f'UL_step{s}']))
    ul_seen_old_mae.append(mean_absolute_error(ul_seen_old[f'UL_step{s}'], ul_seen_gt[f'UL_step{s}']))

    latency_unseen_ref_mae.append(mean_absolute_error(latency_unseen_ref[f'Latency_step{s}'], latency_unseen_gt[f'Latency_step{s}']))
    latency_unseen_old_mae.append(mean_absolute_error(latency_unseen_old[f'Latency_step{s}'], latency_unseen_gt[f'Latency_step{s}']))
    latency_seen_ref_mae.append(mean_absolute_error(latency_seen_ref[f'Latency_step{s}'], latency_seen_gt[f'Latency_step{s}']))
    latency_seen_old_mae.append(mean_absolute_error(latency_seen_old[f'Latency_step{s}'], latency_seen_gt[f'Latency_step{s}']))

    ul_unseen_ref_std.append(np.std(np.abs(ul_unseen_ref[f'UL_step{s}'].values - ul_unseen_gt[f'UL_step{s}'])))
    ul_unseen_old_std.append(np.std(np.abs(ul_unseen_old[f'UL_step{s}'].values - ul_unseen_gt[f'UL_step{s}'])))
    ul_seen_ref_std.append(np.std(np.abs(ul_seen_ref[f'UL_step{s}'].values - ul_seen_gt[f'UL_step{s}'])))
    ul_seen_old_std.append(np.std(np.abs(ul_seen_old[f'UL_step{s}'].values - ul_seen_gt[f'UL_step{s}'])))
 
    latency_unseen_ref_std.append(np.std(np.abs(latency_unseen_ref[f'Latency_step{s}'].values - latency_unseen_gt[f'Latency_step{s}'])))
    latency_unseen_old_std.append(np.std(np.abs(latency_unseen_old[f'Latency_step{s}'].values - latency_unseen_gt[f'Latency_step{s}'])))
    latency_seen_ref_std.append(np.std(np.abs(latency_seen_ref[f'Latency_step{s}'].values - latency_seen_gt[f'Latency_step{s}'])))
    latency_seen_old_std.append(np.std(np.abs(latency_seen_old[f'Latency_step{s}'].values - latency_seen_gt[f'Latency_step{s}'])))

fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.5))
steps_range = range(1, steps + 1)

# UL unseen ref
# plt.plot(steps_range, ul_unseen_ref_mae, label='Ref Unseen', color=color_pallet[0], linewidth=0.8)
# plt.fill_between(steps_range,
#                  np.array(ul_unseen_ref_mae) - np.array(ul_unseen_ref_std),
#                  np.array(ul_unseen_ref_mae) + np.array(ul_unseen_ref_std),
#                  color=color_pallet[0], alpha=0.2)

# # UL unseen old
# plt.plot(steps_range, ul_unseen_old_mae, label='Old Unseen', color=color_pallet[1], linewidth=0.8)
# plt.fill_between(steps_range,
#                  np.array(ul_unseen_old_mae) - np.array(ul_unseen_old_std),
#                  np.array(ul_unseen_old_mae) + np.array(ul_unseen_old_std),
#                  color=color_pallet[1], alpha=0.2)

# UL seen ref
axes[0].plot(steps_range, ul_seen_ref_mae, label='with historic data', color=color_pallet[0], linewidth=1)
axes[0].fill_between(steps_range,
                 np.array(ul_seen_ref_mae) - np.array(ul_seen_ref_std),
                 np.array(ul_seen_ref_mae) + np.array(ul_seen_ref_std),
                 color=color_pallet[0], alpha=0.2)

# UL seen old
axes[0].plot(steps_range, ul_seen_old_mae, label='without historic data', color=color_pallet[1], linewidth=1)
axes[0].fill_between(steps_range,
                 np.array(ul_seen_old_mae) - np.array(ul_seen_old_std),
                 np.array(ul_seen_old_mae) + np.array(ul_seen_old_std),
                 color=color_pallet[1], alpha=0.2)
axes[0].set_ylabel("UL data-rate MAE in Mbps")
# axes[0].legend(loc='upper right')

axes[0].set_title('MAE over prediction horizons')

# Latency seen ref
axes[1].plot(steps_range, latency_seen_ref_mae, label='with historic data', color=color_pallet[0], linewidth=1)
axes[1].fill_between(steps_range,
                 np.array(latency_seen_ref_mae) - np.array(latency_seen_ref_std),
                 np.array(latency_seen_ref_mae) + np.array(latency_seen_ref_std),
                 color=color_pallet[0], alpha=0.2)

# Latency seen old
axes[1].plot(steps_range, latency_seen_old_mae, label='without historic data', color=color_pallet[1], linewidth=1)
axes[1].fill_between(steps_range,
                 np.array(latency_seen_old_mae) - np.array(latency_seen_old_std),
                 np.array(latency_seen_old_mae) + np.array(latency_seen_old_std),
                 color=color_pallet[1], alpha=0.2)
axes[1].set_ylabel("Latency MAE in ms")
axes[1].set_xlabel("Prediction horizon (step) in second")
# axes[0].legend(loc='upper right')

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc='lower center',
    ncol=2,
    fontsize='small'
)

plt.ylim(bottom=0)
plt.grid(True)
plt.tight_layout()
fig.subplots_adjust(bottom=0.18)
plt.savefig("out/final_results/mae_over_pred_horizons.pdf")

# UL seen
ul_seen_ref_f1, ul_seen_ref_precision, ul_seen_ref_recall = binary_classification_metrics(
    ul_seen_gt['UL_step1'], ul_seen_ref['UL_step1'], threshold=14, less_than=True)

ul_seen_old_f1, ul_seen_old_precision, ul_seen_old_recall = binary_classification_metrics(
    ul_seen_gt['UL_step1'], ul_seen_old['UL_step1'], threshold=14, less_than=True)

ul_unseen_ref_f1, ul_unseen_ref_precision, ul_unseen_ref_recall = binary_classification_metrics(
    ul_unseen_gt['UL_step1'], ul_unseen_ref['UL_step1'], threshold=14, less_than=True)

ul_unseen_old_f1, ul_unseen_old_precision, ul_unseen_old_recall = binary_classification_metrics(
    ul_unseen_gt['UL_step1'], ul_unseen_old['UL_step1'], threshold=14, less_than=True)

# Latency seen
latency_seen_ref_f1, latency_seen_ref_precision, latency_seen_ref_recall = binary_classification_metrics(
    latency_seen_gt['Latency_step1'], latency_seen_ref['Latency_step1'], threshold=100, less_than=False)

latency_seen_old_f1, latency_seen_old_precision, latency_seen_old_recall = binary_classification_metrics(
    latency_seen_gt['Latency_step1'], latency_seen_old['Latency_step1'], threshold=100, less_than=False)

latency_unseen_ref_f1, latency_unseen_ref_precision, latency_unseen_ref_recall = binary_classification_metrics(
    latency_unseen_gt['Latency_step1'], latency_unseen_ref['Latency_step1'], threshold=100, less_than=False)

latency_unseen_old_f1, latency_unseen_old_precision, latency_unseen_old_recall = binary_classification_metrics(
    latency_unseen_gt['Latency_step1'], latency_unseen_old['Latency_step1'], threshold=100, less_than=False)

print("UL Seen Ref (<12 Mbps): F1 = {:.3f}, Precision = {:.3f}, Recall = {:.3f}".format(
    ul_seen_ref_f1, ul_seen_ref_precision, ul_seen_ref_recall))
print("UL Seen Old (<12 Mbps): F1 = {:.3f}, Precision = {:.3f}, Recall = {:.3f}".format(
    ul_seen_old_f1, ul_seen_old_precision, ul_seen_old_recall))
print("UL Unseen Ref (<12 Mbps): F1 = {:.3f}, Precision = {:.3f}, Recall = {:.3f}".format(
    ul_unseen_ref_f1, ul_unseen_ref_precision, ul_unseen_ref_recall))
print("UL Unseen Old (<12 Mbps): F1 = {:.3f}, Precision = {:.3f}, Recall = {:.3f}".format(
    ul_unseen_old_f1, ul_unseen_old_precision, ul_unseen_old_recall))

print("Latency Seen Ref (>100 ms): F1 = {:.3f}, Precision = {:.3f}, Recall = {:.3f}".format(
    latency_seen_ref_f1, latency_seen_ref_precision, latency_seen_ref_recall))
print("Latency Seen Old (>100 ms): F1 = {:.3f}, Precision = {:.3f}, Recall = {:.3f}".format(
    latency_seen_old_f1, latency_seen_old_precision, latency_seen_old_recall))
print("Latency Unseen Ref (>100 ms): F1 = {:.3f}, Precision = {:.3f}, Recall = {:.3f}".format(
    latency_unseen_ref_f1, latency_unseen_ref_precision, latency_unseen_ref_recall))
print("Latency Unseen Old (>100 ms): F1 = {:.3f}, Precision = {:.3f}, Recall = {:.3f}".format(
    latency_unseen_old_f1, latency_unseen_old_precision, latency_unseen_old_recall))
# print("Latency Seen (>100 ms): F1 = {:.3f}, Precision = {:.3f}, Recall = {:.3f}".format(
#     latency_f1_seen, latency_precision_seen, latency_recall_seen))



