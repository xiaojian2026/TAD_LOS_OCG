# -*- coding: utf-8 -*-
"""
Title: Terminal Cooperative Guidance Based on LOS Geometry and Optimal Control for TAD Engagement

Section 4.5: Batch Comparative Simulation and Verification under Diverse Trajectory Conditions

This script performs large-scale batch simulations (6435 trajectories) to verify
the robustness and performance of the proposed cooperative guidance law (Case 4)
across a wide range of initial conditions, including variations in attacker
velocity, engagement distance, altitude offset, and defender velocity.
"""

import numpy as np
import sys
import site
user_site = site.getusersitepackages()
if user_site not in sys.path:
    sys.path.append(user_site)

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for multiprocessing
import matplotlib.pyplot as plt
import os
from math import sqrt
import warnings
warnings.filterwarnings("ignore")
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, Border, Side

# Import core simulation components from TAD_Comp
import TAD_Comp
from TAD_Comp import (
    g, deg2rad, rad2deg, dt,
    wrap_pi, pos_A0, pos_T0, Va,
    a_D_max, run_single_simulation
)

# Unified plotting style
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.titlesize'] = 21
plt.rcParams['axes.labelsize'] = 19.5
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['legend.fontsize'] = 16.5

def set_ylim_tight(ax, data, low_pct=0, high_pct=100, margin=0.05, include_zero=False):
    """Set y-axis limits to show all data points.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to modify.
    data : array-like
        Data values for computing limits.
    low_pct, high_pct : float
        Percentile bounds for the data range (default: 0-100, show all data).
    margin : float
        Fractional padding added to the data range.
    include_zero : bool
        If True, ensure zero is included in the y-range.
    """
    d = np.asarray(data, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return
    dmin = np.percentile(d, low_pct)
    dmax = np.percentile(d, high_pct)
    if include_zero:
        dmin = min(dmin, 0.0)
    if dmax - dmin < 1e-10:
        dmax = dmin + 1.0
    pad = (dmax - dmin) * margin
    ax.set_ylim(dmin - pad, dmax + pad)


# ===================== 1. Batch simulation =====================
def _worker_single_run(args):
    """Worker function for parallel batch simulation. Runs a single trajectory."""
    Va_val, dist, alt, Vd_val = args

    # Set global parameters for this run (modify TAD_Comp module globals directly)
    TAD_Comp.Va = Va_val
    TAD_Comp.pos_A0[0] = dist
    TAD_Comp.pos_A0[1] = TAD_Comp.pos_T0[1] + alt
    TAD_Comp.a_D_max = 15.0 * g

    try:
        res = run_single_simulation(CASE=4, Vd=Vd_val, save_traj=False)
        traj = res['traj']
        # Heuristic check statistics (CORRECT direction)
        # Condition 1: K < 0 -> check MAX (closest to 0)
        heur_K_max = np.max(traj['heur_K'])
        # Condition 2: RI3 < 0 -> check MAX (closest to 0)
        heur_RI3_max = np.max(traj['heur_RI3'])
        # Condition 3: I1 > 0 -> check MIN (closest to 0)
        heur_I1_min = np.min(traj['heur_I1'])
        # Condition 4: Phi_c < 0 -> check MAX (closest to 0)
        heur_Phi_c_max = np.max(traj['heur_Phi_c'])
        # Condition 5: |sigma|tgo^2 < 2|ZEM| -> check ratio when ZEM > 10m
        ZEM_arr = traj['ZEM']
        sigma_ratio_arr = traj['heur_sigma_tgo2_over_2ZEM']
        mask_ZEM = np.abs(ZEM_arr) > 10.0  # Only when ZEM > 10m
        if np.any(mask_ZEM):
            heur_sigma_tgo2_max = np.max(sigma_ratio_arr[mask_ZEM])
        else:
            heur_sigma_tgo2_max = np.max(sigma_ratio_arr)
        # Condition 6: tgoB exists throughout
        heur_tgoB_exists_all = np.all(traj['heur_tgoB_exists'] > 0.5)
        return {
            'Va': Va_val,
            'dist': dist,
            'alt': alt,
            'Vd': Vd_val,
            'miss': res['miss_final_m'],
            'ZEM': traj['ZEM'][-1],
            'ZEAE1': traj['ZEAE1_deg'][-1],
            'ZEAE2': traj['ZEAE2_deg'][-1],
            'a_D': res['a_D_sf_peak_g'],
            'a_A': res['a_A_sf_peak_g'],
            'a_T': res['a_T_sf_peak_g'],
            'J_D': res['J_D'],
            'J_T': res['J_T'],
            'J_A': res['J_A'],
            'termination_mode': res['termination_mode'],
            # Convergence condition statistics (CORRECT direction)
            'heur_K_max': heur_K_max,          # Condition 1: K < 0
            'heur_RI3_max': heur_RI3_max,      # Condition 2: RI3 < 0
            'heur_I1_min': heur_I1_min,        # Condition 3: I1 > 0
            'heur_Phi_c_max': heur_Phi_c_max,  # Condition 4: Phi_c < 0
            'heur_sigma_tgo2_max': heur_sigma_tgo2_max,  # Condition 5: |sigma|tgo^2 < 2|ZEM|
            'heur_tgoB_exists_all': heur_tgoB_exists_all,  # Condition 6: tgoB exists
            # Backward compatibility (old keys, same values as new correct ones)
            'heur_K_min': heur_K_max,
            'heur_RI3_min': heur_RI3_max,
            'heur_Phi_c_min': heur_Phi_c_max,
            'heur_sign_u1_sign_ZEM_min': 0.0,  # Not used anymore
            'heur_dDdot_over_Rdot_ad_max': 0.0,  # Not used anymore
            'heur_ZEMdot_g_over_ZEMdot_u1_max': 0.0,  # Not used anymore
        }
    except Exception as e:
        print(f"  Error at Va={Va_val}, dist={dist}, alt={alt}, Vd={Vd_val}: {e}")
        return None


def batch_simulate_case4(folder='Batch_data_figure', n_workers=14):
    """Run batch simulations for Case 4 across a grid of initial conditions.

    The parameter grid covers:
    - Attacker velocity Va: 600 to 1000 m/s (9 values)
    - Engagement distance: 5000 to 10000 m (11 values)
    - Altitude offset: -3000 to +3000 m relative to target (13 values)
    - Defender velocity Vd: 400 to 600 m/s (5 values)
    Total: 9 x 11 x 13 x 5 = 6435 trajectories.

    Returns
    -------
    results : list of dict
        Each dict contains simulation parameters and performance metrics.
    """
    import multiprocessing as mp

    os.makedirs(folder, exist_ok=True)

    # Parameter grids (full resolution)
    Va_range = np.arange(600, 1050, 50)      # 9 values: 600, 650, ..., 1000
    dist_range = np.arange(5000, 10500, 500)  # 11 values: 5000, 5500, ..., 10000
    alt_range = np.arange(-3000, 3500, 500)   # 13 values: -3000, -2500, ..., +3000
    Vd_range = np.arange(400, 650, 50)        # 5 values: 400, 450, ..., 600

    # Generate all parameter combinations
    all_args = []
    for Va_val in Va_range:
        for dist in dist_range:
            for alt in alt_range:
                for Vd_val in Vd_range:
                    all_args.append((Va_val, dist, alt, Vd_val))

    total = len(all_args)
    print(f"  Total trajectories: {total}, workers: {n_workers}")

    # Run in parallel
    results = []
    with mp.Pool(processes=n_workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_worker_single_run, all_args)):
            if res is not None:
                results.append(res)
            if (i + 1) % 500 == 0:
                print(f"  Batch progress: {i+1}/{total}")

    # Save raw batch data to compressed NPZ file for post-processing
    np.savez(os.path.join(folder, 'batch_results.npz'),
             Va=np.array([r['Va'] for r in results]),
             dist=np.array([r['dist'] for r in results]),
             alt=np.array([r['alt'] for r in results]),
             Vd=np.array([r['Vd'] for r in results]),
             miss=np.array([r['miss'] for r in results]),
             ZEM=np.array([r['ZEM'] for r in results]),
             ZEAE1=np.array([r['ZEAE1'] for r in results]),
             ZEAE2=np.array([r['ZEAE2'] for r in results]),
             a_D=np.array([r['a_D'] for r in results]),
             a_A=np.array([r['a_A'] for r in results]),
             a_T=np.array([r['a_T'] for r in results]),
             J_D=np.array([r['J_D'] for r in results]),
             J_T=np.array([r['J_T'] for r in results]),
             J_A=np.array([r['J_A'] for r in results]),
             termination_mode=np.array([r['termination_mode'] for r in results]),
             heur_K_min=np.array([r['heur_K_min'] for r in results]),
             heur_RI3_min=np.array([r['heur_RI3_min'] for r in results]),
             heur_I1_min=np.array([r['heur_I1_min'] for r in results]),
             heur_Phi_c_min=np.array([r['heur_Phi_c_min'] for r in results]),
             heur_sigma_tgo2_max=np.array([r['heur_sigma_tgo2_max'] for r in results]),
             heur_sign_u1_sign_ZEM_min=np.array([r['heur_sign_u1_sign_ZEM_min'] for r in results]),
             heur_tgoB_exists_all=np.array([r['heur_tgoB_exists_all'] for r in results]),
             heur_dDdot_over_Rdot_ad_max=np.array([r['heur_dDdot_over_Rdot_ad_max'] for r in results]),
             heur_ZEMdot_g_over_ZEMdot_u1_max=np.array([r['heur_ZEMdot_g_over_ZEMdot_u1_max'] for r in results]))
    print(f"\n  Raw data saved: {folder}/batch_results.npz ({len(results)} runs)")

    return results


# ===================== 2. Plotting functions =====================
def plot_terminal_scatter(results, folder='Batch_data_figure'):
    """Plot terminal error distributions as scatter plots.

    Generates four separate figures for miss distance, ZEM, ZEAE1, and ZEAE2,
    each showing the value across all trajectory indices with enlarged markers.
    Vertical dividers and labels show Va grouping (outer loop).
    """
    n = len(results)
    x = np.arange(1, n + 1)
    miss = np.array([r['miss'] for r in results])
    zem = np.array([r['ZEM'] for r in results])
    zeae1 = np.array([r['ZEAE1'] for r in results])
    zeae2 = np.array([r['ZEAE2'] for r in results])

    # Va grouping info (outer loop: Va -> dist -> alt -> Vd)
    Va_range = np.arange(600, 1050, 50)      # 9 values
    n_per_Va = n // len(Va_range)            # trajectories per Va group

    # Helper: add Va group dividers and labels
    def add_va_groups(ax, y_max):
        for i, va in enumerate(Va_range):
            # Vertical divider at group boundary
            if i > 0:
                boundary = i * n_per_Va
                ax.axvline(x=boundary, color='gray', linestyle=':', alpha=0.6)
            # Label at group center, placed above the axes so the 9 groups never collide
            center = i * n_per_Va + n_per_Va / 2
            ax.text(center, 1.006, f'Va={va:.0f}\nm/s', ha='center', va='bottom',
                    transform=ax.get_xaxis_transform(),
                    fontsize=15, color='black', linespacing=0.95,
                    bbox=dict(boxstyle='round,pad=0.12', facecolor='white', alpha=0.85, edgecolor='none'))

    # ---------- 1. Miss distance ----------
    fig1, ax1 = plt.subplots(figsize=(14, 7))
    ax1.plot(x, miss, 'k.', markersize=3)
    ax1.set_xlabel('Trajectory index', fontsize=15)
    ax1.set_ylabel('Miss distance (m)', fontsize=15)
    ax1.tick_params(axis='both', labelsize=18)
    ax1.set_title('Terminal miss distance distribution', fontsize=16, pad=38)
    ax1.grid(True, linestyle='--', alpha=0.5)
    set_ylim_tight(ax1, miss, include_zero=True)
    # Add Va group labels at top
    y_max = ax1.get_ylim()[1]
    add_va_groups(ax1, y_max)
    plt.tight_layout()
    plt.savefig(f'{folder}/Terminal_Miss.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved Terminal_Miss")

    # ---------- 2. ZEM ----------
    fig2, ax2 = plt.subplots(figsize=(14, 7))
    ax2.plot(x, zem, 'k.', markersize=3)
    ax2.set_xlabel('Trajectory index', fontsize=15)
    ax2.set_ylabel('ZEM (m)', fontsize=15)
    ax2.tick_params(axis='both', labelsize=18)
    ax2.set_title('Terminal ZEM distribution', fontsize=16, pad=38)
    ax2.grid(True, linestyle='--', alpha=0.5)
    set_ylim_tight(ax2, zem)
    y_max = ax2.get_ylim()[1]
    add_va_groups(ax2, y_max)
    plt.tight_layout()
    plt.savefig(f'{folder}/Terminal_ZEM.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved Terminal_ZEM")

    # ---------- 3. ZEAE1 ----------
    fig3, ax3 = plt.subplots(figsize=(14, 7))
    ax3.plot(x, zeae1, 'k.', markersize=3)
    ax3.set_xlabel('Trajectory index', fontsize=15)
    ax3.set_ylabel('ZEAE1 (°)', fontsize=15)
    ax3.tick_params(axis='both', labelsize=18)
    ax3.set_title('Terminal ZEAE1 distribution', fontsize=16, pad=38)
    ax3.grid(True, linestyle='--', alpha=0.5)
    set_ylim_tight(ax3, zeae1)
    y_max = ax3.get_ylim()[1]
    add_va_groups(ax3, y_max)
    plt.tight_layout()
    plt.savefig(f'{folder}/Terminal_ZEAE1.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved Terminal_ZEAE1")

    # ---------- 4. ZEAE2 ----------
    fig4, ax4 = plt.subplots(figsize=(14, 7))
    ax4.plot(x, zeae2, 'k.', markersize=3)
    ax4.set_xlabel('Trajectory index', fontsize=15)
    ax4.set_ylabel('ZEAE2 (°)', fontsize=15)
    ax4.tick_params(axis='both', labelsize=18)
    ax4.set_title('Terminal ZEAE2 distribution', fontsize=16, pad=38)
    ax4.grid(True, linestyle='--', alpha=0.5)
    set_ylim_tight(ax4, zeae2)
    y_max = ax4.get_ylim()[1]
    add_va_groups(ax4, y_max)
    plt.tight_layout()
    plt.savefig(f'{folder}/Terminal_ZEAE2.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved Terminal_ZEAE2")


def plot_ecdf(results, folder='Batch_data_figure'):
    """Plot ECDF (Empirical Cumulative Distribution Function) for miss distance and overload.

    Replaces arbitrary trajectory-index scatter plots with ECDF plots,
    which are more informative for identifying operating limits.
    """
    miss = np.array([r['miss'] for r in results])
    a_D = np.array([r['a_D'] for r in results])
    a_A = np.array([r['a_A'] for r in results])
    a_T = np.array([r['a_T'] for r in results])

    # ---------- 1. Miss distance ECDF ----------
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    miss_sorted = np.sort(miss)
    ecdf = np.arange(1, len(miss_sorted) + 1) / len(miss_sorted)
    ax1.plot(miss_sorted, ecdf, 'k-', linewidth=2)
    ax1.axvline(x=1.0, color='r', linestyle='--', label='1 m threshold')
    ax1.set_xlabel('Miss distance (m)', fontsize=15)
    ax1.set_ylabel('Cumulative probability', fontsize=15)
    ax1.tick_params(axis='both', labelsize=18)
    ax1.set_title('ECDF of terminal miss distance', fontsize=16, pad=15)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='lower right', fontsize=14)
    ax1.set_xlim(left=0)
    ax1.set_ylim([0, 1.02])
    plt.tight_layout()
    plt.savefig(f'{folder}/ECDF_Miss.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved ECDF_Miss")

    # ---------- 2. Overload ECDF ----------
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    for name, arr, style in [('Defender', a_D, 'k-'), ('Attacker', a_A, 'b--'), ('Target', a_T, 'r:')]:
        arr_sorted = np.sort(arr)
        ecdf = np.arange(1, len(arr_sorted) + 1) / len(arr_sorted)
        ax2.plot(arr_sorted, ecdf, style, linewidth=2, label=name)
    ax2.set_xlabel('Peak overload (g)', fontsize=15)
    ax2.set_ylabel('Cumulative probability', fontsize=15)
    ax2.tick_params(axis='both', labelsize=18)
    ax2.set_title('ECDF of peak overload', fontsize=16, pad=15)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='lower right', fontsize=14)
    ax2.set_xlim(left=0)
    ax2.set_ylim([0, 1.02])
    plt.tight_layout()
    plt.savefig(f'{folder}/ECDF_Overload.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved ECDF_Overload")


def plot_overload_distribution(results, folder='Batch_data_figure'):
    """Plot peak overload distributions as histograms for all three bodies.

    Shows the frequency distribution of defender, attacker, and target peak
    overloads across all batch simulation trajectories.
    """
    a_D = np.array([r['a_D'] for r in results])
    a_A = np.array([r['a_A'] for r in results])
    a_T = np.array([r['a_T'] for r in results])

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    axes[0].hist(a_D, bins=40, edgecolor='black', color='lightgray')
    axes[0].set_xlabel('Peak overload (g)', fontsize=14)
    axes[0].set_ylabel('Frequency', fontsize=14)
    axes[0].set_title('Defender peak overload', fontsize=15, pad=10)
    axes[0].tick_params(axis='both', labelsize=15)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].set_xlim(left=0)

    axes[1].hist(a_A, bins=40, edgecolor='black', color='lightgray')
    axes[1].set_xlabel('Peak overload (g)', fontsize=14)
    axes[1].set_ylabel('Frequency', fontsize=14)
    axes[1].set_title('Attacker peak overload', fontsize=15, pad=10)
    axes[1].tick_params(axis='both', labelsize=15)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].set_xlim(left=0)

    axes[2].hist(a_T, bins=40, edgecolor='black', color='lightgray')
    axes[2].set_xlabel('Peak overload (g)', fontsize=14)
    axes[2].set_ylabel('Frequency', fontsize=14)
    axes[2].set_title('Target peak overload', fontsize=15, pad=10)
    axes[2].tick_params(axis='both', labelsize=15)
    axes[2].grid(True, linestyle='--', alpha=0.5)
    axes[2].set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(f'{folder}/Overload_Distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved Overload_Distribution")


def plot_maneuver_effort(results, folder='Batch_data_figure'):
    """Plot maneuver effort ratio scatter plots.

    Compares the maneuver effort (J = ∫a² dt) of the target and attacker
    relative to the defender, showing the cooperative guidance advantage.
    """
    J_D = np.array([r['J_D'] for r in results])
    J_T = np.array([r['J_T'] for r in results])
    J_A = np.array([r['J_A'] for r in results])

    ratio_TD = J_T / (J_D + 1e-10)  # Target-to-defender ratio
    ratio_AD = J_A / (J_D + 1e-10)  # Attacker-to-defender ratio

    fig, axes = plt.subplots(1, 2, figsize=(12, 7.2))  # 5:3 aspect ratio

    axes[0].plot(J_D, ratio_TD, 'k.', markersize=4)
    axes[0].set_xlabel('Defender maneuver effort (g²·s)')
    axes[0].set_ylabel('Target / Defender ratio')
    axes[0].set_title('Target-to-defender\nmaneuver effort ratio', fontsize=14)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    set_ylim_tight(axes[0], ratio_TD, include_zero=True)

    axes[1].plot(J_D, ratio_AD, 'k.', markersize=4)
    axes[1].set_xlabel('Defender maneuver effort (g²·s)')
    axes[1].set_ylabel('Attacker / Defender ratio')
    axes[1].set_title('Attacker-to-defender\nmaneuver effort ratio', fontsize=14)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    set_ylim_tight(axes[1], ratio_AD, include_zero=True)

    plt.tight_layout()
    plt.savefig(f'{folder}/Maneuver_Effort_Ratio.png', dpi=150)
    plt.close()
    print("Saved Maneuver_Effort_Ratio")


def plot_representative_trajectories(folder='Batch_data_figure'):
    """Plot representative trajectories for upward and dive interception scenarios.

    Generates two separate figures:
    - Upward interception: attacker altitude above target (0 to +3000 m)
    - Dive interception: attacker altitude below target (-500 to -3000 m)
    Both use Vd=400 m/s, Va=600 m/s, distance=6000 m.
    """
    global Va, pos_A0
    orig_Va = Va
    orig_pos = pos_A0.copy()

    Va = 600
    pos_A0[0] = 6000

    # Altitude offsets relative to target aircraft altitude (5000 m)
    alt_list_up = [0, 500, 1000, 1500, 2000, 2500, 3000]       # Upward interception
    alt_list_down = [-500, -1000, -1500, -2000, -2500, -3000]  # Dive interception

    # ========== Upward interception (separate figure) ==========
    fig1, ax1 = plt.subplots(figsize=(10, 7))
    for i, alt in enumerate(alt_list_up):
        pos_A0[1] = pos_T0[1] + alt
        res = run_single_simulation(CASE=4, Vd=400, save_traj=True)
        traj = res['traj']
        lab_T = 'Target' if i == 0 else ""
        lab_A = 'Attacker' if i == 0 else ""
        lab_D = 'Defender' if i == 0 else ""
        ax1.plot(traj['x_T'], traj['y_T'], ':', color='black', label=lab_T, linewidth=1.2)
        ax1.plot(traj['x_A'], traj['y_A'], '--', color='black', label=lab_A, linewidth=1.2)
        ax1.plot(traj['x_D'], traj['y_D'], '-', color='black', label=lab_D, linewidth=1.8)
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_title('Representative trajectories during upward interception')
    ax1.legend(loc='best')
    ax1.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{folder}/Trajectory_Upward.png', dpi=150)
    plt.close()
    print("Saved Trajectory_Upward")

    # ========== Dive interception (separate figure) ==========
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    for i, alt in enumerate(alt_list_down):
        pos_A0[1] = pos_T0[1] + alt
        res = run_single_simulation(CASE=4, Vd=400, save_traj=True)
        traj = res['traj']
        lab_T = 'Target' if i == 0 else ""
        lab_A = 'Attacker' if i == 0 else ""
        lab_D = 'Defender' if i == 0 else ""
        ax2.plot(traj['x_T'], traj['y_T'], ':', color='black', label=lab_T, linewidth=1.2)
        ax2.plot(traj['x_A'], traj['y_A'], '--', color='black', label=lab_A, linewidth=1.2)
        ax2.plot(traj['x_D'], traj['y_D'], '-', color='black', label=lab_D, linewidth=1.8)
    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('y (m)')
    ax2.set_title('Representative trajectories during dive interception')
    ax2.legend(loc='best')
    ax2.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{folder}/Trajectory_Dive.png', dpi=150)
    plt.close()
    print("Saved Trajectory_Dive")

    # Restore original global parameters
    Va = orig_Va
    pos_A0[:] = orig_pos


def plot_heatmaps(results, folder='Batch_data_figure'):
    """Generate parameter-mapped heatmaps for operating limit identification."""
    miss = np.array([r['miss'] for r in results])
    Va = np.array([r['Va'] for r in results])
    dist = np.array([r['dist'] for r in results])
    alt = np.array([r['alt'] for r in results])
    Vd = np.array([r['Vd'] for r in results])

    Va_unique = np.unique(Va)
    dist_unique = np.unique(dist)
    alt_unique = np.unique(alt)
    Vd_unique = np.unique(Vd)

    # ---------- 1. Va x dist heatmaps for 3 alt slices (Vd=500) ----------
    alt_slices = [-3000, 0, 3000]
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    for idx, alt_val in enumerate(alt_slices):
        mask = (alt == alt_val) & (Vd == 500)
        miss_sub = miss[mask]
        Va_sub = Va[mask]
        dist_sub = dist[mask]

        heatmap_data = np.zeros((len(Va_unique), len(dist_unique)))
        for i, v in enumerate(Va_unique):
            for j, d in enumerate(dist_unique):
                m = (Va_sub == v) & (dist_sub == d)
                if np.any(m):
                    heatmap_data[i, j] = np.mean(miss_sub[m])
                else:
                    heatmap_data[i, j] = np.nan

        ax = axes[idx]
        im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', origin='lower', vmin=0, vmax=1.5)
        ax.set_xticks(range(0, len(dist_unique), 2))
        ax.set_xticklabels(dist_unique[::2], rotation=45)
        ax.set_yticks(range(len(Va_unique)))
        ax.set_yticklabels(Va_unique)
        ax.tick_params(axis='both', labelsize=13)
        ax.set_xlabel('Engagement distance (m)', fontsize=16)
        if idx == 0:
            ax.set_ylabel('Attacker velocity Va (m/s)', fontsize=16)
        ax.set_title(f'alt = {alt_val} m, Vd = 500 m/s', fontsize=15)

        for i in range(len(Va_unique)):
            for j in range(len(dist_unique)):
                if not np.isnan(heatmap_data[i, j]):
                    color = 'white' if heatmap_data[i, j] > 0.8 else 'black'
                    ax.text(j, i, f'{heatmap_data[i, j]:.2f}', ha='center', va='center', fontsize=8, color=color)

    fig.subplots_adjust(left=0.08, right=0.90, bottom=0.17, top=0.85, wspace=0.32)
    cbar_ax = fig.add_axes([0.915, 0.17, 0.013, 0.68])
    cbar = fig.colorbar(im, cax=cbar_ax, label='Mean miss distance (m)')
    cbar.set_label('Mean miss distance (m)', fontsize=16)
    cbar.ax.tick_params(labelsize=13)
    plt.suptitle('Mean miss distance heatmaps across altitude slices', y=1.02, fontsize=15)
    plt.savefig(f'{folder}/Heatmap_Va_dist_3slices.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved Heatmap_Va_dist_3slices.png")

    # ---------- 2. alt x Vd heatmaps for 3 dist slices (Va=800) ----------
    dist_slices = [5000, 7500, 10000]
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    for idx, dist_val in enumerate(dist_slices):
        mask = (Va == 800) & (dist == dist_val)
        miss_sub = miss[mask]
        alt_sub = alt[mask]
        Vd_sub = Vd[mask]

        heatmap_data = np.zeros((len(alt_unique), len(Vd_unique)))
        for i, a in enumerate(alt_unique):
            for j, v in enumerate(Vd_unique):
                m = (alt_sub == a) & (Vd_sub == v)
                if np.any(m):
                    heatmap_data[i, j] = np.mean(miss_sub[m])
                else:
                    heatmap_data[i, j] = np.nan

        ax = axes[idx]
        im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', origin='lower', vmin=0, vmax=1.5)
        ax.set_xticks(range(len(Vd_unique)))
        ax.set_xticklabels(Vd_unique)
        ax.set_yticks(range(len(alt_unique)))
        ax.set_yticklabels(alt_unique)
        ax.tick_params(axis='both', labelsize=13)
        ax.set_xlabel('Defender velocity Vd (m/s)', fontsize=16)
        if idx == 0:
            ax.set_ylabel('Altitude offset (m)', fontsize=16)
        ax.set_title(f'dist = {dist_val} m, Va = 800 m/s', fontsize=15)

        for i in range(len(alt_unique)):
            for j in range(len(Vd_unique)):
                if not np.isnan(heatmap_data[i, j]):
                    color = 'white' if heatmap_data[i, j] > 0.8 else 'black'
                    ax.text(j, i, f'{heatmap_data[i, j]:.2f}', ha='center', va='center', fontsize=8, color=color)

    fig.subplots_adjust(left=0.08, right=0.90, bottom=0.17, top=0.85, wspace=0.32)
    cbar_ax = fig.add_axes([0.915, 0.17, 0.013, 0.68])
    cbar = fig.colorbar(im, cax=cbar_ax, label='Mean miss distance (m)')
    cbar.set_label('Mean miss distance (m)', fontsize=16)
    cbar.ax.tick_params(labelsize=13)
    plt.suptitle('Mean miss distance heatmaps across distance slices', y=1.02, fontsize=15)
    plt.savefig(f'{folder}/Heatmap_alt_Vd_3slices.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved Heatmap_alt_Vd_3slices.png")


# ===================== 3. Statistics and table output =====================
def print_statistics(results):
    """Print summary statistics for peak overload and miss distance."""
    n = len(results)
    miss = np.array([r['miss'] for r in results])
    a_D = np.array([r['a_D'] for r in results])
    a_A = np.array([r['a_A'] for r in results])
    a_T = np.array([r['a_T'] for r in results])
    term_modes = np.array([r['termination_mode'] for r in results])

    print("\n" + "=" * 75)
    print("Table 2. Peak overload distribution (all trajectories)")
    print("=" * 75)
    print(f"{'':>15} {'Mean':>8} {'RMS':>8} {'Max':>8} {'95%':>8} {'99%':>8}")
    for name, arr in [('Defender', a_D), ('Attacker', a_A), ('Target', a_T)]:
        print(f"{name:>15} {np.mean(arr):8.2f} {np.sqrt(np.mean(arr**2)):8.2f} "
              f"{np.max(arr):8.2f} {np.percentile(arr, 95):8.2f} {np.percentile(arr, 99):8.2f}")

    print("\n" + "=" * 75)
    print("Table 3. Terminal miss distance statistics")
    print("=" * 75)
    print(f"  Total runs : {n}")
    print(f"  Mean       : {np.mean(miss):.3f} m")
    print(f"  RMS        : {np.sqrt(np.mean(miss**2)):.3f} m")
    print(f"  Max        : {np.max(miss):.3f} m")
    print(f"  Median     : {np.median(miss):.3f} m")
    print(f"  95%        : {np.percentile(miss, 95):.3f} m")
    print(f"  99%        : {np.percentile(miss, 99):.3f} m")

    # Success/failure separation
    success_mask = miss <= 1.0
    n_success = np.sum(success_mask)
    n_fail = n - n_success
    print(f"\n  Success (miss <= 1m): {n_success} ({n_success/n*100:.2f}%)")
    print(f"  Failure (miss > 1m):  {n_fail} ({n_fail/n*100:.2f}%)")

    if n_success > 0:
        print(f"\n  --- Success cases ---")
        print(f"  Mean miss: {np.mean(miss[success_mask]):.3f} m")
        print(f"  Mean defender overload: {np.mean(a_D[success_mask]):.2f} g")
    if n_fail > 0:
        print(f"\n  --- Failure cases ---")
        print(f"  Mean miss: {np.mean(miss[~success_mask]):.3f} m")
        print(f"  Mean defender overload: {np.mean(a_D[~success_mask]):.2f} g")

    # Worst parameter tuples
    print(f"\n  --- Worst 10 parameter combinations (largest miss) ---")
    sorted_idx = np.argsort(miss)[::-1]
    for i in range(min(10, n)):
        idx = sorted_idx[i]
        r = results[idx]
        print(f"  {i+1}. miss={r['miss']:.3f} m, Va={r['Va']:.0f}, dist={r['dist']:.0f}, alt={r['alt']:.0f}, Vd={r['Vd']:.0f}, term={r['termination_mode']}")

    # Termination mode statistics
    print(f"\n  --- Termination mode statistics ---")
    for mode in ['intercept', 'target_hit', 'pass_cross', 'timeout']:
        cnt = np.sum(term_modes == mode)
        print(f"  {mode:>12}: {cnt:>5} ({cnt/n*100:.2f}%)")
    print("=" * 75)


def print_heuristic_checks(results):
    """Print heuristic check statistics for Section 3.4 verification."""
    n = len(results)

    # Extract heuristic check arrays
    heur_K_min = np.array([r['heur_K_min'] for r in results])
    heur_RI3_min = np.array([r['heur_RI3_min'] for r in results])
    heur_I1_min = np.array([r['heur_I1_min'] for r in results])
    heur_Phi_c_min = np.array([r['heur_Phi_c_min'] for r in results])
    heur_sigma_tgo2_max = np.array([r['heur_sigma_tgo2_max'] for r in results])
    heur_sign_u1_sign_ZEM_min = np.array([r['heur_sign_u1_sign_ZEM_min'] for r in results])
    heur_tgoB_exists_all = np.array([r['heur_tgoB_exists_all'] for r in results])
    heur_dDdot_over_Rdot_ad_max = np.array([r['heur_dDdot_over_Rdot_ad_max'] for r in results])
    heur_ZEMdot_g_over_ZEMdot_u1_max = np.array([r['heur_ZEMdot_g_over_ZEMdot_u1_max'] for r in results])

    print("\n" + "=" * 75)
    print("Table 4. Heuristic condition verification (Section 3.4)")
    print("=" * 75)
    print(f"  Total runs: {n}")
    print("")

    # 1. K < 0
    pct_K_neg = np.sum(heur_K_min < 0) / n * 100
    print(f"  1. K < 0:")
    print(f"     min(K) = {np.min(heur_K_min):.4f}, max(min(K)) = {np.max(heur_K_min):.4f}")
    print(f"     Runs with K < 0 throughout: {pct_K_neg:.1f}%")
    print("")

    # 2. R_at * I3 < 0
    pct_RI3_neg = np.sum(heur_RI3_min < 0) / n * 100
    print(f"  2. R_at * I3 < 0:")
    print(f"     min(R_at*I3) = {np.min(heur_RI3_min):.4f}, max(min) = {np.max(heur_RI3_min):.4f}")
    print(f"     Runs with R_at*I3 < 0 throughout: {pct_RI3_neg:.1f}%")
    print("")

    # 3. I1 > 0
    pct_I1_pos = np.sum(heur_I1_min > 0) / n * 100
    print(f"  3. I1 > 0:")
    print(f"     min(I1) = {np.min(heur_I1_min):.4f}, max(min) = {np.max(heur_I1_min):.4f}")
    print(f"     Runs with I1 > 0 throughout: {pct_I1_pos:.1f}%")
    print("")

    # 4. Phi_c < 0
    pct_Phi_c_neg = np.sum(heur_Phi_c_min < 0) / n * 100
    print(f"  4. Phi_c < 0:")
    print(f"     min(Phi_c) = {np.min(heur_Phi_c_min):.4f}, max(min) = {np.max(heur_Phi_c_min):.4f}")
    print(f"     Runs with Phi_c < 0 throughout: {pct_Phi_c_neg:.1f}%")
    print("")

    # 5. |sigma| * tgo^2 / (2*|ZEM|) < 1
    pct_sigma_tgo2 = np.sum(heur_sigma_tgo2_max < 1.0) / n * 100
    print(f"  5. |sigma| * tgo^2 / (2*|ZEM|) < 1:")
    print(f"     max = {np.max(heur_sigma_tgo2_max):.4f}, min(max) = {np.min(heur_sigma_tgo2_max):.4f}")
    print(f"     Runs with condition satisfied: {pct_sigma_tgo2:.1f}%")
    print("")

    # 6. sign(u1) * sign(ZEM) > 0
    pct_sign_match = np.sum(heur_sign_u1_sign_ZEM_min > 0) / n * 100
    print(f"  6. sign(u1) * sign(ZEM) > 0:")
    print(f"     min = {np.min(heur_sign_u1_sign_ZEM_min):.4f}, max(min) = {np.max(heur_sign_u1_sign_ZEM_min):.4f}")
    print(f"     Runs with condition satisfied: {pct_sign_match:.1f}%")
    print("")

    # 7. tgo_B exists
    pct_tgoB_exists = np.sum(heur_tgoB_exists_all) / n * 100
    print(f"  7. tgo_B exists (I3 != 0 and I1 > 0):")
    print(f"     Runs with tgo_B exists throughout: {pct_tgoB_exists:.1f}%")
    print("")

    # 8. |delta_d_dot| / |R_ad_dot| << 1
    print(f"  8. |delta_d_dot| / |R_ad_dot| << 1:")
    print(f"     max = {np.max(heur_dDdot_over_Rdot_ad_max):.4f}, mean(max) = {np.mean(heur_dDdot_over_Rdot_ad_max):.4f}")
    print("")

    # 9. |ZEMdot_g| / |ZEMdot_u1| << 1
    print(f"  9. |ZEMdot_g| / |ZEMdot_u1| << 1:")
    print(f"     max = {np.max(heur_ZEMdot_g_over_ZEMdot_u1_max):.4f}, mean(max) = {np.mean(heur_ZEMdot_g_over_ZEMdot_u1_max):.4f}")
    print("=" * 75)


def export_tables_excel(results, folder='Batch_data_figure'):
    """Export all statistical tables to Excel.

    Sheets:
    - Summary: Overall statistics, success/failure separation
    - Worst10: Worst 10 parameter combinations
    - HeuristicChecks: Heuristic condition verification (Section 3.4)
    - Table2: Overload interval statistics
    - Table3: Maneuver effort ratio
    """
    n = len(results)
    miss = np.array([r['miss'] for r in results])
    a_T = np.array([r['a_T'] for r in results])
    a_A = np.array([r['a_A'] for r in results])
    a_D = np.array([r['a_D'] for r in results])
    J_D = np.array([r['J_D'] for r in results])
    J_A = np.array([r['J_A'] for r in results])
    Va = np.array([r['Va'] for r in results])
    dist = np.array([r['dist'] for r in results])
    alt = np.array([r['alt'] for r in results])
    Vd = np.array([r['Vd'] for r in results])

    success_mask = miss <= 1.0
    n_success = int(np.sum(success_mask))
    n_fail = n - n_success

    # ---- Write to Excel ----
    wb = Workbook()
    thin = Side(border_style="thin", color="000000")
    thick = Side(border_style="thick", color="000000")
    border_header = Border(top=thick, bottom=thin)
    border_bottom = Border(bottom=thick)
    center = Alignment(horizontal="center", vertical="center", wrap_text=True)
    bold = Font(bold=True)

    def fill_sheet(ws, title, headers, data_rows):
        """Helper to fill an Excel sheet with formatted table."""
        ws.append([title])
        ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(headers))
        c = ws.cell(row=1, column=1)
        c.font = bold
        c.alignment = center
        ws.append(headers)
        for j in range(1, len(headers) + 1):
            c = ws.cell(row=2, column=j)
            c.font = bold
            c.alignment = center
            c.border = border_header
        for row in data_rows:
            ws.append(row)
        nd = len(data_rows)
        for r in range(3, 3 + nd):
            for j in range(1, len(headers) + 1):
                c = ws.cell(row=r, column=j)
                c.alignment = center
                if r == 2 + nd:
                    c.border = border_bottom
        for j, w in enumerate([24, 18, 18, 18][:len(headers)], 1):
            ws.column_dimensions[chr(64 + j)].width = w

    # ---- Sheet 1: Summary ----
    ws_summary = wb.active
    ws_summary.title = "Summary"
    summary_headers = ["Metric", "Value", "Unit"]
    summary_data = [
        ["Total trajectories", n, "-"],
        ["Success (miss <= 1 m)", f"{n_success} ({n_success/n*100:.2f}%)", "-"],
        ["Failure (miss > 1 m)", f"{n_fail} ({n_fail/n*100:.2f}%)", "-"],
        ["", "", ""],
        ["Miss distance (all)", "", ""],
        ["  Mean", f"{np.mean(miss):.3f}", "m"],
        ["  Median", f"{np.median(miss):.3f}", "m"],
        ["  RMS", f"{np.sqrt(np.mean(miss**2)):.3f}", "m"],
        ["  Max", f"{np.max(miss):.3f}", "m"],
        ["  95th percentile", f"{np.percentile(miss, 95):.3f}", "m"],
        ["  99th percentile", f"{np.percentile(miss, 99):.3f}", "m"],
        ["", "", ""],
        ["Miss distance (success only)", "", ""],
        ["  Mean", f"{np.mean(miss[success_mask]):.3f}", "m"],
        ["  Mean defender overload", f"{np.mean(a_D[success_mask]):.2f}", "g"],
        ["", "", ""],
        ["Miss distance (failure only)", "", ""],
        ["  Mean", f"{np.mean(miss[~success_mask]):.3f}", "m"],
        ["  Mean defender overload", f"{np.mean(a_D[~success_mask]):.2f}", "g"],
    ]
    fill_sheet(ws_summary, "Batch Simulation Summary", summary_headers, summary_data)

    # ---- Sheet 2: Worst10 ----
    ws_worst = wb.create_sheet("Worst10")
    worst_headers = ["Rank", "Miss (m)", "Va (m/s)", "dist (m)", "alt (m)", "Vd (m/s)", "a_D (g)"]
    sorted_idx = np.argsort(miss)[::-1]
    worst_data = []
    for i in range(min(10, n)):
        idx = sorted_idx[i]
        worst_data.append([
            i + 1,
            f"{miss[idx]:.3f}",
            f"{Va[idx]:.0f}",
            f"{dist[idx]:.0f}",
            f"{alt[idx]:.0f}",
            f"{Vd[idx]:.0f}",
            f"{a_D[idx]:.2f}"
        ])
    fill_sheet(ws_worst, "Worst 10 parameter combinations (largest miss)", worst_headers, worst_data)

    # ---- Sheet 2b: MaxOverload10 ----
    ws_overload = wb.create_sheet("MaxOverload10")
    overload_headers = ["Rank", "a_D (g)", "Miss (m)", "Va (m/s)", "dist (m)", "alt (m)", "Vd (m/s)"]
    sorted_idx_od = np.argsort(a_D)[::-1]
    overload_data = []
    for i in range(min(10, n)):
        idx = sorted_idx_od[i]
        overload_data.append([
            i + 1,
            f"{a_D[idx]:.2f}",
            f"{miss[idx]:.3f}",
            f"{Va[idx]:.0f}",
            f"{dist[idx]:.0f}",
            f"{alt[idx]:.0f}",
            f"{Vd[idx]:.0f}"
        ])
    fill_sheet(ws_overload, "Max 10 defender overload combinations", overload_headers, overload_data)

    # ---- Sheet 3: HeuristicChecks ----
    heur_K_min = np.array([r['heur_K_min'] for r in results])
    heur_RI3_min = np.array([r['heur_RI3_min'] for r in results])
    heur_I1_min = np.array([r['heur_I1_min'] for r in results])
    heur_Phi_c_min = np.array([r['heur_Phi_c_min'] for r in results])
    heur_sigma_tgo2_max = np.array([r['heur_sigma_tgo2_max'] for r in results])
    heur_sign_u1_sign_ZEM_min = np.array([r['heur_sign_u1_sign_ZEM_min'] for r in results])
    heur_tgoB_exists_all = np.array([r['heur_tgoB_exists_all'] for r in results])
    heur_dDdot_over_Rdot_ad_max = np.array([r['heur_dDdot_over_Rdot_ad_max'] for r in results])
    heur_ZEMdot_g_over_ZEMdot_u1_max = np.array([r['heur_ZEMdot_g_over_ZEMdot_u1_max'] for r in results])

    ws_heur = wb.create_sheet("HeuristicChecks")
    heur_headers = ["Condition", "Min", "Max", "% Satisfied"]
    heur_data = [
        ["1. K < 0", f"{np.min(heur_K_min):.4f}", f"{np.max(heur_K_min):.4f}",
         f"{np.sum(heur_K_min < 0)/n*100:.1f}%"],
        ["2. R_at * I3 < 0", f"{np.min(heur_RI3_min):.4f}", f"{np.max(heur_RI3_min):.4f}",
         f"{np.sum(heur_RI3_min < 0)/n*100:.1f}%"],
        ["3. I1 > 0", f"{np.min(heur_I1_min):.4f}", f"{np.max(heur_I1_min):.4f}",
         f"{np.sum(heur_I1_min > 0)/n*100:.1f}%"],
        ["4. Phi_c < 0", f"{np.min(heur_Phi_c_min):.4f}", f"{np.max(heur_Phi_c_min):.4f}",
         f"{np.sum(heur_Phi_c_min < 0)/n*100:.1f}%"],
        ["5. |sigma|*tgo^2/(2*|ZEM|) < 1", f"{np.min(heur_sigma_tgo2_max):.4f}", f"{np.max(heur_sigma_tgo2_max):.4f}",
         f"{np.sum(heur_sigma_tgo2_max < 1.0)/n*100:.1f}%"],
        ["6. sign(u1)*sign(ZEM) > 0", f"{np.min(heur_sign_u1_sign_ZEM_min):.4f}", f"{np.max(heur_sign_u1_sign_ZEM_min):.4f}",
         f"{np.sum(heur_sign_u1_sign_ZEM_min > 0)/n*100:.1f}%"],
        ["7. tgo_B exists", "-", "-",
         f"{np.sum(heur_tgoB_exists_all)/n*100:.1f}%"],
        ["8. |delta_d_dot|/|R_ad_dot| << 1", f"{np.min(heur_dDdot_over_Rdot_ad_max):.4f}", f"{np.max(heur_dDdot_over_Rdot_ad_max):.4f}",
         "-"],
        ["9. |ZEMdot_g|/|ZEMdot_u1| << 1", f"{np.min(heur_ZEMdot_g_over_ZEMdot_u1_max):.4f}", f"{np.max(heur_ZEMdot_g_over_ZEMdot_u1_max):.4f}",
         "-"],
    ]
    fill_sheet(ws_heur, "Heuristic condition verification (Section 3.4)", heur_headers, heur_data)

    # ---- Sheet 4: TerminationModes ----
    termination_modes = np.array([r['termination_mode'] for r in results])
    ws_term = wb.create_sheet("TerminationModes")
    term_headers = ["Termination Mode", "Count", "Proportion"]
    term_data = []
    for mode in ['intercept', 'target_hit', 'pass_cross', 'timeout']:
        cnt = int(np.sum(termination_modes == mode))
        term_data.append([mode, cnt, f"{cnt/n*100:.2f}%"])
    fill_sheet(ws_term, "Termination mode statistics", term_headers, term_data)

    # ---- Table 2: Overload interval statistics (separated by success/failure) ----
    bins = [0, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60]
    labels = ["<=6", "(6,7]", "(7,8]", "(8,9]", "(9,10]",
              "(10,20]", "(20,30]", "(30,40]", "(40,50]", "(50,60]"]

    def count_in_bins(arr):
        """Count number of elements falling into each bin interval."""
        counts = []
        for i in range(len(bins) - 1):
            if i == 0:
                cnt = np.sum(arr <= bins[i + 1])
            else:
                cnt = np.sum((arr > bins[i]) & (arr <= bins[i + 1]))
            counts.append(cnt)
        return counts

    # Success/failure mask
    success_mask = miss <= 1.0
    fail_mask = ~success_mask
    n_success = int(np.sum(success_mask))
    n_fail = int(np.sum(fail_mask))

    cnt_T = count_in_bins(a_T)
    cnt_A = count_in_bins(a_A)
    cnt_D_success = count_in_bins(a_D[success_mask])
    cnt_D_fail = count_in_bins(a_D[fail_mask])

    # ---- Table 3: Maneuver effort ratio Jd/Ja (separated by success/failure) ----
    ratio = J_D / (J_A + 1e-10)
    thresholds = [0.8, 0.5, 0.4, 0.3, 0.2, 0.1]
    t3_counts_success = [int(np.sum(ratio[success_mask] <= t)) for t in thresholds]
    t3_counts_fail = [int(np.sum(ratio[fail_mask] <= t)) for t in thresholds]

    # Table 2 sheet
    ws2 = wb.create_sheet("Table2")
    headers2 = ["Maximum Overload (g)", "Attacker", "Defender\n(R_ad,min <= 1m)", "Defender\n(R_ad,min > 1m)"]
    data2 = []
    for i, lab in enumerate(labels):
        data2.append([lab,
                      f"{cnt_A[i]/n*100:.2f}%",
                      f"{cnt_D_success[i]/n_success*100:.2f}%",
                      f"{cnt_D_fail[i]/n_fail*100:.2f}%" if n_fail > 0 else "-"])
    fill_sheet(ws2,
               "Table 5. Statistical proportions of trajectories under maximum overload.",
               headers2, data2)

    # Table 3 sheet
    ws3 = wb.create_sheet("Table3")
    headers3 = ["Maneuver Effort Ratio (Jd/Ja)",
                "R_ad,min <= 1m\nNo. of Trajectories", "R_ad,min <= 1m\nProportion",
                "R_ad,min > 1m\nNo. of Trajectories", "R_ad,min > 1m\nProportion"]
    data3 = []
    for i, t in enumerate(thresholds):
        data3.append([f"<={t}",
                      t3_counts_success[i], f"{t3_counts_success[i]/n_success*100:.2f}%",
                      t3_counts_fail[i], f"{t3_counts_fail[i]/n_fail*100:.2f}%" if n_fail > 0 else "-"])
    fill_sheet(ws3,
               "Table 8. Maneuver effort ratio of the defender relative to the incoming attacker.",
               headers3, data3)

    out_path = os.path.join(folder, "Tables_T2_T3.xlsx")
    wb.save(out_path)
    print(f"  Excel tables saved: {out_path}")


# ===================== Main program =====================
if __name__ == "__main__":
    # ====== 测试开关：先跑 5 个工况验证 Va 变化是否生效 ======
    TEST_MODE = False
    
    if TEST_MODE:
        print("=" * 60)
        print("TEST MODE: 测试 Va 变化是否生效")
        print("=" * 60)
        
        dist = 5000
        alt = 1000
        Vd = 400
        
        print(f"固定参数: R0={dist} m, Δh={alt} m, Vd={Vd} m/s")
        print()
        print(f"{'Va (m/s)':<12} {'Miss (m)':<12} {'a_D (g)':<10} {'a_A (g)':<10}")
        print("-" * 50)
        
        for Va_test in [600, 700, 800, 900, 1000]:
            res = _worker_single_run((Va_test, dist, alt, Vd))
            if res is not None:
                print(f"{Va_test:<12.0f} {res['miss']:<12.4f} {res['a_D']:<10.2f} {res['a_A']:<10.2f}")
            else:
                print(f"{Va_test:<12.0f} ERROR")
        
        print()
        print("=" * 60)
        print("如果不同 Va 的结果不一样，说明修复成功！")
        print("把 TEST_MODE 改成 False 即可跑全部 6435 条轨迹")
        print("=" * 60)
        sys.exit(0)
    
    folder = 'Batch_data_figure'
    os.makedirs(folder, exist_ok=True)

    print("=" * 60)
    print("Batch simulation (Case 4, 6435 trajectories)")
    print("=" * 60)

    # Step 1: Batch simulation + save raw data
    results = batch_simulate_case4(folder)

    # Step 2: Print statistics
    print_statistics(results)

    # Step 2b: Print heuristic check statistics
    print_heuristic_checks(results)

    # Step 3: Export Excel tables
    export_tables_excel(results, folder)

    # Step 3b: Export CSV file
    import csv
    csv_path = os.path.join(folder, 'batch_results.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Va', 'dist', 'alt', 'Vd', 'miss', 'ZEM', 'ZEAE1', 'ZEAE2',
                         'a_D', 'a_A', 'a_T', 'J_D', 'J_T', 'J_A', 'termination_mode'])
        for r in results:
            writer.writerow([
                r['Va'], r['dist'], r['alt'], r['Vd'],
                r['miss'], r['ZEM'], r['ZEAE1'], r['ZEAE2'],
                r['a_D'], r['a_A'], r['a_T'],
                r['J_D'], r['J_T'], r['J_A'],
                r['termination_mode']
            ])
    print(f"  CSV saved: {csv_path}")

    # Step 4: Generate all plots
    plot_terminal_scatter(results, folder)
    plot_ecdf(results, folder)
    plot_overload_distribution(results, folder)
    plot_maneuver_effort(results, folder)
    plot_representative_trajectories(folder)
    plot_heatmaps(results, folder)

    print(f"\nAll results saved to '{folder}' directory.")
