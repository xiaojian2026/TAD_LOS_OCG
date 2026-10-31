# -*- coding: utf-8 -*-
"""
Plot batch grid simulation results:
1. Heatmaps: 3 slices for Vd=400/500/600, x=Va, y=dist, color=miss distance
2. Grouped ECDF: miss distance ECDF grouped by Vd
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Plot style - match TAD_Comp.py
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 12
rcParams['axes.labelsize'] = 15
rcParams['axes.titlesize'] = 16
rcParams['legend.fontsize'] = 15
rcParams['xtick.labelsize'] = 15
rcParams['ytick.labelsize'] = 15
rcParams['grid.linestyle'] = '--'
rcParams['grid.alpha'] = 0.3

# Load data
HERE = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(HERE, 'Batch_data_figure', 'batch_results.csv')

Va_list = []
dist_list = []
alt_list = []
Vd_list = []
miss_list = []

with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        Va_list.append(float(row['Va']))
        dist_list.append(float(row['dist']))
        alt_list.append(float(row['alt']))
        Vd_list.append(float(row['Vd']))
        miss_list.append(float(row['miss']))

Va_arr = np.array(Va_list)
dist_arr = np.array(dist_list)
alt_arr = np.array(alt_list)
Vd_arr = np.array(Vd_list)
miss_arr = np.array(miss_list)

print(f"Loaded {len(miss_arr)} rows")
print(f"Va range: {Va_arr.min()} - {Va_arr.max()}")
print(f"dist range: {dist_arr.min()} - {dist_arr.max()}")
print(f"Vd values: {sorted(np.unique(Vd_arr))}")
print(f"alt values: {sorted(np.unique(alt_arr))}")

# ============================================================
# 1. Heatmaps: 2 rows x 3 cols
#    Row: R0 = 5000m (near) / 10000m (far)
#    Col: Vd = 400 (low) / 500 (mid) / 600 (high)
#    Inside each: x=Va, y=alt
# ============================================================
R0_list = [5000.0, 10000.0]
Vd_list = [400.0, 500.0, 600.0]

fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharey=True)

for row_idx, R0 in enumerate(R0_list):
    for col_idx, Vd in enumerate(Vd_list):
        ax = axes[row_idx, col_idx]
        
        mask = (dist_arr == R0) & (Vd_arr == Vd)
        Va_sub = Va_arr[mask]
        alt_sub = alt_arr[mask]
        miss_sub = miss_arr[mask]
        
        Va_vals = sorted(np.unique(Va_sub))
        alt_vals = sorted(np.unique(alt_sub))
        
        grid = np.zeros((len(alt_vals), len(Va_vals)))
        for i, a in enumerate(alt_vals):
            for j, v in enumerate(Va_vals):
                m = (Va_sub == v) & (alt_sub == a)
                grid[i, j] = np.mean(miss_sub[m])
        
        # Compute cell boundaries for proper alignment
        Va_step = Va_vals[1] - Va_vals[0]
        alt_step = alt_vals[1] - alt_vals[0]
        Va_extent = [Va_vals[0] - Va_step/2, Va_vals[-1] + Va_step/2,
                     alt_vals[0] - alt_step/2, alt_vals[-1] + alt_step/2]
        
        im = ax.imshow(grid, aspect='auto', origin='lower',
                       extent=Va_extent,
                       cmap='YlOrRd', vmin=0, vmax=1.2)
        
        ax.set_title(f'$R_0$ = {int(R0)} m, $V_d$ = {int(Vd)} m/s')
        ax.set_xlabel('$V_a$ (m/s)')
        if col_idx == 0:
            ax.set_ylabel('Altitude offset (m)')
        
        # Annotate values at cell centers
        for i in range(len(alt_vals)):
            for j in range(len(Va_vals)):
                x_center = Va_vals[j]
                y_center = alt_vals[i]
                ax.text(x_center, y_center, f'{grid[i,j]:.2f}',
                        ha='center', va='center', fontsize=12, color='black')

# Horizontal colorbar at bottom
fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.12, wspace=0.025, hspace=0.25)
cbar_ax = fig.add_axes([0.3, 0.04, 0.4, 0.02])
cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
cbar.set_label('Miss distance (m)', labelpad=5)

fig.suptitle('Miss distance heatmaps across engagement conditions', y=0.98, fontsize=16)
heatmap_path = os.path.join(HERE, 'Batch_data_figure', 'Heatmap_miss_2x3.png')
plt.savefig(heatmap_path, dpi=200, bbox_inches='tight')
print(f"Saved: {heatmap_path}")
plt.close()

# ============================================================
# 2. Grouped ECDF: miss distance by Vd
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))

colors = ['k', '0.4', '0.7']
linestyles = ['-', '--', '-.']

Vd_plot = [400.0, 500.0, 600.0]
for idx, Vd in enumerate(Vd_plot):
    sub = miss_arr[Vd_arr == Vd]
    sub_sorted = np.sort(sub)
    ecdf = np.arange(1, len(sub_sorted) + 1) / len(sub_sorted)
    
    ax.plot(sub_sorted, ecdf, color=colors[idx], linestyle=linestyles[idx],
            linewidth=1.5, label=f'$V_d$ = {int(Vd)} m/s')

ax.axvline(x=1.0, color='k', linestyle=':', linewidth=1.0, label='1 m threshold')
ax.set_xlabel('Miss distance (m)')
ax.set_ylabel('Cumulative probability')
ax.set_title('Terminal miss distance ECDF by defender speed')
ax.legend(loc='lower right')
ax.grid(True)
ax.set_xlim(0, 1.5)

ecdf_path = os.path.join(HERE, 'Batch_data_figure', 'ECDF_miss_by_Vd.png')
plt.savefig(ecdf_path, dpi=200, bbox_inches='tight')
print(f"Saved: {ecdf_path}")
plt.close()

# ============================================================
# 3. Grouped ECDF: miss distance by Va
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))

Va_unique = sorted(np.unique(Va_arr))
Va_colors = plt.cm.Greys(np.linspace(0.3, 0.9, len(Va_unique)))

for idx, Va in enumerate(Va_unique):
    sub = miss_arr[Va_arr == Va]
    sub_sorted = np.sort(sub)
    ecdf = np.arange(1, len(sub_sorted) + 1) / len(sub_sorted)
    
    ax.plot(sub_sorted, ecdf, color=Va_colors[idx], linewidth=1.2,
            label=f'$V_a$ = {int(Va)} m/s')

ax.axvline(x=1.0, color='k', linestyle=':', linewidth=1.0)
ax.set_xlabel('Miss distance (m)')
ax.set_ylabel('Cumulative probability')
ax.set_title('Terminal miss distance ECDF by attacker speed')
ax.legend(loc='lower right', ncol=2, fontsize=9)
ax.grid(True)
ax.set_xlim(0, 1.5)

ecdf_va_path = os.path.join(HERE, 'Batch_data_figure', 'ECDF_miss_by_Va.png')
plt.savefig(ecdf_va_path, dpi=200, bbox_inches='tight')
print(f"Saved: {ecdf_va_path}")
plt.close()

print("\nDone!")
