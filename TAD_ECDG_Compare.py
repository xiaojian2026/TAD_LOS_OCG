# -*- coding: utf-8 -*-
"""
TAD_ECDG_Compare.py
====================
Side-by-side comparison of published ECDG (Ref.[21] printed Eq.34) vs corrected ECDG.

Published variant uses Ref.[21] Eq.(34) literally:
    φ_pub = (V_DM·Δt_f/2)·[(t+Δt_f)·(e^{2t_gDM/(t+Δt_f)}−1) − 2t_gDM]
where t is the current absolute simulation time and Δt_f = t_fTM − t_fDM (constant).

Corrected variant uses Appendix B Eq.(B.13):
    φ_corr = (V_DM·Δt_f/2)·[2t_gDM − t_gTM·(e^{2t_gDM/t_gTM}−1)]

The key difference: the printed Eq.(34) uses (t+Δt_f) as the exponent time constant
instead of t_gTM, causing exponential blow-up at the initial instant when t≈0 and
Δt_f ≈ 1 s (instead of t_gTM ≈ 16 s).

Same simulation conditions as TAD_Comp.py Section 4.2.
"""

import os
import sys
import numpy as np
from math import atan2, cos, sin, exp, pi

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

import TAD_Comp as TC
from TAD_Comp import (
    g, deg2rad, dt, t_max, eps, TGO_MAX, R_PROTECT,
    Va, pos_A0, theta_A_sf0, a_A_max,
    pos_D0, a_D_max,
    Vt, pos_T0, a_T_max,
    tau_A, tau_D, tau_T,
    wrap_pi,
)


def phi_published(V_DM, dt_f, t_gDM, t_sim):
    """Ref.[21] Eq.(34) printed form.

    φ = (V_DM·Δt_f/2)·[(t+Δt_f)·(e^{2t_gDM/(t+Δt_f)}−1) − 2t_gDM]
    t = current absolute simulation time.
    """
    t_plus_dt = t_sim + dt_f
    if t_plus_dt <= 1e-6:
        return 0.0
    exponent = 2.0 * t_gDM / t_plus_dt
    if abs(exponent) > 25.0:
        return 0.0
    return (V_DM * dt_f / 2.0) * (t_plus_dt * (exp(exponent) - 1.0) - 2.0 * t_gDM)


def phi_corrected(V_DM, dt_f, t_gDM, t_gTM):
    """Appendix B Eq.(B.13) corrected form.

    φ = (V_DM·Δt_f/2)·[2t_gDM − t_gTM·(e^{2t_gDM/t_gTM}−1)]
    """
    if dt_f <= 0.0:
        return 0.0
    exponent = 2.0 * t_gDM / t_gTM
    if abs(exponent) > 25.0:
        return 0.0
    return (V_DM * dt_f / 2.0) * (2.0 * t_gDM - t_gTM * (exp(exponent) - 1.0))


def _zero_effort_integrals(geom, u_M, th_A, th_D, t_gDM, t_sim, version):
    """Zero-effort nominal integration.

    version: 'published' or 'corrected'
    """
    xA, yA = geom['x_A'], geom['y_A']
    xD, yD = geom['x_D'], geom['y_D']
    xT, yT = geom['x_T'], geom['y_T']
    thT = geom['th_T']
    Vd, Vt = geom['Vd'], geom['Vt']
    gA = th_A
    I_D = 0.0
    I_T = 0.0
    I_M = 0.0
    prev = None
    tau = 0.0
    dti = 0.05
    while tau <= t_gDM + dti:
        dx_ad = xA - xD
        dy_ad = yA - yD
        dx_at = xA - xT
        dy_at = yA - yT
        R_DMn = np.hypot(dx_ad, dy_ad)
        R_TMn = np.hypot(dx_at, dy_at)
        vx_ad = Va * cos(gA) - Vd * cos(th_D)
        vy_ad = Va * sin(gA) - Vd * sin(th_D)
        R_DM_dotn = (dx_ad * vx_ad + dy_ad * vy_ad) / max(R_DMn, 1e-9)
        vx_at = Va * cos(gA) - Vt * cos(thT)
        vy_at = Va * sin(gA) - Vt * sin(thT)
        R_TM_dotn = (dx_at * vx_at + dy_at * vy_at) / max(R_TMn, 1e-9)
        V_DMn = -R_DM_dotn
        V_TMn = -R_TM_dotn
        if V_DMn <= 1.0 or V_TMn <= 1.0 or R_DMn < 1.0:
            break
        t_gDMn = R_DMn / V_DMn
        t_gTMn = R_TMn / V_TMn
        dt_fn = t_gTMn - t_gDMn

        if version == 'published':
            phi_n = phi_published(V_DMn, dt_fn, t_gDMn, t_sim + tau)
        else:
            phi_n = phi_corrected(V_DMn, dt_fn, t_gDMn, t_gTMn)

        lDMn = atan2(dy_ad, dx_ad)
        lTMn = atan2(dy_at, dx_at)
        cD_n = cos(th_D - lDMn)
        cT_n = cos(thT - lTMn)
        cM_n = cos(gA - lTMn)
        cM2n = cos(gA - lDMn)
        bD_n = cD_n * t_gDMn
        bT_n = -(R_DMn / R_TMn) * cT_n * t_gDMn - (cT_n / R_TMn) * phi_n
        bM_n = ((R_DMn / R_TMn) * cM_n - cM2n) * t_gDMn + (cM_n / R_TMn) * phi_n
        cur = (bD_n ** 2, bT_n ** 2, bM_n * u_M)
        if prev is not None:
            I_D += 0.5 * (prev[0] + cur[0]) * dti
            I_T += 0.5 * (prev[1] + cur[1]) * dti
            I_M += 0.5 * (prev[2] + cur[2]) * dti
        prev = cur
        gA += u_M / Va * dti
        xA += Va * cos(gA) * dti
        yA += Va * sin(gA) * dti
        xD += Vd * cos(th_D) * dti
        yD += Vd * sin(th_D) * dti
        xT += Vt * cos(thT) * dti
        yT += Vt * sin(thT) * dti
        tau += dti
    return I_D, I_T, I_M


def guidance_step(params, geom, coeff, t_sim, version):
    """ECDG guidance command at one evaluation point.

    version: 'published' or 'corrected'
    """
    R_DM = geom['R_ad']
    R_TM = geom['R_at']
    l_DM = geom['q_ad']
    l_TM = geom['q_at']
    ld_TM = geom['q_at_dot']
    ld_DM = geom['q_ad_dot']
    th_D = geom['th_D']
    th_T = geom['th_T']
    th_A = geom['th_A']
    u_M = coeff['a_A']

    R_DM_dot = geom['R_ad_dot']
    V_DM = -R_DM_dot
    V_TM = -geom['R_at_dot']
    if V_DM <= 1.0 or V_TM <= 1.0 or R_DM < 1.0:
        return 0.0, 0.0
    t_gDM = R_DM / V_DM
    t_gTM = R_TM / V_TM
    dt_f = t_gTM - t_gDM
    if t_gDM < 0.05:
        return 0.0, 0.0

    if version == 'published':
        phi = phi_published(V_DM, dt_f, t_gDM, t_sim)
    else:
        phi = phi_corrected(V_DM, dt_f, t_gDM, t_gTM)

    phi_los = wrap_pi(l_TM - l_DM)
    z = R_DM * phi_los
    z_dot = R_DM_dot * phi_los + R_DM * (ld_TM - ld_DM)

    cD = cos(th_D - l_DM)
    cT = cos(th_T - l_TM)
    cM = cos(th_A - l_TM)
    cM2 = cos(th_A - l_DM)
    b_D = cD * t_gDM
    b_T = -(R_DM / R_TM) * cT * t_gDM - (cT / R_TM) * phi
    b_M = (R_DM / R_TM * cM - cM2) * t_gDM + cM / R_TM * phi

    I_D, I_T, I_M = _zero_effort_integrals(geom, u_M, th_A, th_D, t_gDM, t_sim, version)

    R_D_w = params.get('R_D_weight', 1.0)
    R_T_w = params.get('R_T_weight', 1.0)

    Z = z + z_dot * t_gDM + phi * ld_TM + I_M
    den = I_D / R_D_w + I_T / R_T_w
    if den < 1e-12:
        return 0.0, 0.0
    u_D = -(b_D / R_D_w) * Z / den
    u_T = -(b_T / R_T_w) * Z / den
    return u_D, u_T


def run_simulation(Vd, version, cfg=None):
    """Run a single ECDG simulation.

    version: 'published' or 'corrected'
    cfg: optional dict overriding default initial conditions.
    """
    from scipy.integrate import solve_ivp

    # Default initial conditions (our case)
    pos_A0_use = pos_A0
    pos_D0_use = pos_D0
    pos_T0_use = pos_T0
    Va_use = Va
    Vt_use = Vt

    if cfg is not None:
        pos_A0_use = cfg.get('pos_A0', pos_A0)
        pos_D0_use = cfg.get('pos_D0', pos_D0)
        pos_T0_use = cfg.get('pos_T0', pos_T0)
        Va_use = cfg.get('Va', Va)
        Vt_use = cfg.get('Vt', Vt)

    params = TC.get_guidance_params(3, Vd)

    # Initial headings: use cfg values if provided, otherwise match TAD_Comp defaults
    if cfg is not None and 'theta_A0' in cfg:
        theta_A0 = cfg['theta_A0'] * deg2rad
    else:
        theta_A0 = np.pi  # 180 deg, attacker flies left (matches TAD_Comp.py)

    if cfg is not None and 'theta_D0' in cfg:
        theta_D0 = cfg['theta_D0'] * deg2rad
    else:
        dx0 = pos_A0_use[0] - pos_D0_use[0]
        dy0 = pos_A0_use[1] - pos_D0_use[1]
        if pos_A0_use[1] >= pos_D0_use[1]:
            theta_D0 = atan2(dy0, dx0) - 6 * deg2rad
        else:
            theta_D0 = atan2(dy0, dx0) + 6 * deg2rad

    if cfg is not None and 'theta_T0' in cfg:
        theta_T0 = cfg['theta_T0'] * deg2rad
    else:
        theta_T0 = theta_D0

    def deriv(t, y):
        x_T, y_T, th_T, a_T_sf = y[0], y[1], y[2], y[3]
        x_A, y_A, th_A, a_A_sf = y[4], y[5], y[6], y[7]
        x_D, y_D, th_D, a_D_sf = y[8], y[9], y[10], y[11]
        th_T, th_A, th_D = wrap_pi(th_T), wrap_pi(th_A), wrap_pi(th_D)

        dx_at, dy_at = x_A - x_T, y_A - y_T
        R_at = np.sqrt(dx_at**2 + dy_at**2)
        q_at = wrap_pi(atan2(dy_at, dx_at))
        vx_at = Va_use*cos(th_A) - Vt_use*cos(th_T)
        vy_at = Va_use*sin(th_A) - Vt_use*sin(th_T)
        R_at_dot = (dx_at*vx_at + dy_at*vy_at) / (R_at + eps)
        q_at_dot = (dx_at*vy_at - dy_at*vx_at) / (R_at**2 + eps)

        dx_ad, dy_ad = x_A - x_D, y_A - y_D
        R_ad = np.sqrt(dx_ad**2 + dy_ad**2)
        q_ad = wrap_pi(atan2(dy_ad, dx_ad))
        vx_ad = Va_use*cos(th_A) - Vd*cos(th_D)
        vy_ad = Va_use*sin(th_A) - Vd*sin(th_D)
        R_ad_dot = (dx_ad*vx_ad + dy_ad*vy_ad) / (R_ad + eps)
        q_ad_dot = (dx_ad*vy_ad - dy_ad*vx_ad) / (R_ad**2 + eps)

        I1 = -2.0 * R_at_dot / R_at
        I2 = cos(th_A - q_at) / R_at
        a_A = a_A_sf - g*cos(th_A)

        geom = {'Vd': Vd, 'Vt': Vt_use,
                'R_ad': R_ad, 'R_ad_dot': R_ad_dot, 'q_ad': q_ad, 'q_ad_dot': q_ad_dot,
                'R_at': R_at, 'R_at_dot': R_at_dot, 'q_at': q_at, 'q_at_dot': q_at_dot,
                'th_A': th_A, 'th_D': th_D, 'th_T': th_T,
                'x_A': x_A, 'y_A': y_A, 'x_D': x_D, 'y_D': y_D,
                'x_T': x_T, 'y_T': y_T}
        coeff = {'a_A': a_A}

        u_D, u_T = guidance_step(params, geom, coeff, t, version)

        u_A = 3.0 * Va_use * q_at_dot

        u_D_cmd = np.clip(u_D + g*cos(th_D), -a_D_max, a_D_max)
        u_T_cmd = np.clip(u_T + g*cos(th_T), -a_T_max, a_T_max)
        u_A_cmd = np.clip(u_A + g*cos(th_A), -a_A_max, a_A_max)

        return [Vt_use*cos(th_T), Vt_use*sin(th_T), (a_T_sf - g*cos(th_T))/Vt_use,
                (u_T_cmd - a_T_sf)/tau_T,
                Va_use*cos(th_A), Va_use*sin(th_A), (a_A_sf - g*cos(th_A))/Va_use,
                (u_A_cmd - a_A_sf)/tau_A,
                Vd*cos(th_D), Vd*sin(th_D), (a_D_sf - g*cos(th_D))/Vd,
                (u_D_cmd - a_D_sf)/tau_D]

    def ev_target(t, y):
        return np.hypot(y[0]-y[4], y[1]-y[5]) - 1.0
    ev_target.terminal = True
    ev_target.direction = -1.0

    def ev_closest(t, y):
        dx = y[4] - y[8]
        dy = y[5] - y[9]
        R = np.hypot(dx, dy) + eps
        vx = Va_use*np.cos(y[6]) - Vd*np.cos(y[10])
        vy = Va_use*np.sin(y[6]) - Vd*np.sin(y[10])
        return (dx*vx + dy*vy) / R
    ev_closest.terminal = True
    ev_closest.direction = 1.0

    y0 = np.array([pos_T0_use[0], pos_T0_use[1], wrap_pi(theta_T0), 0.0,
                   pos_A0_use[0], pos_A0_use[1], wrap_pi(theta_A0), 0.0,
                   pos_D0_use[0], pos_D0_use[1], wrap_pi(theta_D0), 0.0])

    sol = solve_ivp(deriv, (0.0, t_max), y0, method='RK45',
                    rtol=1e-9, atol=1e-11,
                    events=(ev_target, ev_closest), dense_output=True)

    t_th = sol.t_events[0]
    t_cl = sol.t_events[1]

    if len(t_th) > 0 and (len(t_cl) == 0 or t_th[0] <= t_cl[0]):
        t_miss = float(t_th[0])
    elif len(t_cl) > 0:
        t_miss = float(t_cl[0])
    else:
        t_miss = float(sol.t[-1])

    ym = sol.sol(t_miss)
    miss_final = float(np.hypot(ym[4] - ym[8], ym[5] - ym[9]))

    ts = np.linspace(0.0, t_miss, 2000)
    S = sol.sol(ts)
    a_D_g = S[11] / g
    a_A_g = S[7] / g
    a_T_g = S[3] / g
    peak_D = float(np.max(np.abs(a_D_g)))

    traj = {
        't': ts,
        'x_T': S[0], 'y_T': S[1],
        'x_A': S[4], 'y_A': S[5],
        'x_D': S[8], 'y_D': S[9],
        'a_D_g': a_D_g,
        'a_A_g': a_A_g,
        'a_T_g': a_T_g,
    }
    return miss_final, peak_D, traj


def _run_task(args):
    """Worker function for parallel execution (must be at module level for Windows spawn)."""
    case_name, Vd, version, cfg = args
    miss, peak, traj = run_simulation(Vd, version, cfg=cfg)
    return case_name, Vd, version, miss, peak, traj


if __name__ == "__main__":
    # --- Scenario definitions ---
    # Our case (default TAD geometry)
    our_case = {
        'name': 'Our case',
        'Vd_list': [600.0],
        'cfg': None,  # use default initial conditions
    }

    # Ref.[21] case: Table 1 initial conditions
    # (x_M0,y_M0)=(15km,0), (x_D0,y_D0)=(0,6km), (x_T0,y_T0)=(0,6km)
    # V_M=V_D=600 m/s, V_T=300 m/s
    # gamma_M0=135°, gamma_D0=0°, gamma_T0=0°
    ref21_case = {
        'name': 'Ref.[21] case',
        'Vd_list': [600.0],
        'cfg': {
            'pos_A0': [15000.0, 0.0],
            'pos_D0': [0.0, 6000.0],
            'pos_T0': [0.0, 6000.0],
            'Va': 600.0,
            'Vt': 300.0,
            'theta_A0': 135.0,
            'theta_D0': 0.0,
            'theta_T0': 0.0,
        },
    }

    all_cases = [our_case, ref21_case]

    print("=" * 100)
    print("ECDG: ECDG (Ref.[21] Eq.34 printed) vs C-ECDG (Appendix B corrected)")
    print("=" * 100)
    print()

    all_results = []
    all_traj_data = {}

    # Build task list: (case_name, Vd, version, cfg)
    tasks = []
    for case in all_cases:
        case_name = case['name']
        cfg = case['cfg']
        for Vd in case['Vd_list']:
            tasks.append((case_name, Vd, 'published', cfg))
            tasks.append((case_name, Vd, 'corrected', cfg))

    print(f"Running {len(tasks)} simulations in parallel...")
    print()

    # Parallel execution
    from concurrent.futures import ProcessPoolExecutor, as_completed

    results_map = {}
    with ProcessPoolExecutor(max_workers=min(6, len(tasks))) as executor:
        futures = {executor.submit(_run_task, task): task for task in tasks}
        for future in as_completed(futures):
            case_name, Vd, version, miss, peak, traj = future.result()
            key = (case_name, Vd)
            if key not in results_map:
                results_map[key] = {}
            results_map[key][version] = {'miss': miss, 'peak': peak, 'traj': traj}

    # Print results table
    for case in all_cases:
        case_name = case['name']
        Vd_list = case['Vd_list']
        print(f"--- {case_name} ---")
        print(f"{'Vd (m/s)':>10} | {'ECDG miss (m)':>16} | {'C-ECDG miss (m)':>18} | "
              f"{'ECDG peak_D (g)':>18} | {'C-ECDG peak_D (g)':>20}")
        print("-" * 90)

        all_traj_data[case_name] = {}
        for Vd in Vd_list:
            key = (case_name, Vd)
            mp = results_map[key]['published']['miss']
            mc = results_map[key]['corrected']['miss']
            pp = results_map[key]['published']['peak']
            pc = results_map[key]['corrected']['peak']
            print(f"{Vd:10.1f} | {mp:16.4f} | {mc:18.4f} | "
                  f"{pp:18.4f} | {pc:20.4f}")
            all_results.append((case_name, Vd, mp, mc, pp, pc))
            all_traj_data[case_name][Vd] = {
                'published': results_map[key]['published']['traj'],
                'corrected': results_map[key]['corrected']['traj'],
            }

    print("=" * 100)

    out_dir = os.path.join(SRC_DIR, "TAD_ECDG_Compare_data")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "ecdg_published_vs_corrected.csv")

    import csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Case", "Vd (m/s)", "ECDG_miss (m)", "C-ECDG_miss (m)",
                    "ECDG_peak_D (g)", "C-ECDG_peak_D (g)"])
        for case_name, Vd, mp, mc, pp, pc in all_results:
            w.writerow([case_name, f"{Vd:.1f}", f"{mp:.4f}", f"{mc:.4f}", f"{pp:.4f}", f"{pc:.4f}"])

    print(f"\nCSV saved to: {csv_path}")

    # --- Plot trajectories and overload histories (black-white style, match TAD_Comp) ---
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams['lines.linewidth'] = 1.8
    plt.rcParams['axes.labelsize'] = 15
    plt.rcParams['xtick.labelsize'] = 15
    plt.rcParams['ytick.labelsize'] = 15
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['grid.linewidth'] = 0.5

    # Marker style per Vd (match TAD_Comp)
    vd_markers = {400.0: 'D', 500.0: 'o', 600.0: 's'}

    # Plot each case group separately
    for case in all_cases:
        case_name = case['name']
        Vd_list = case['Vd_list']
        traj_data = all_traj_data[case_name]
        safe_name = case_name.replace(' ', '_').replace('.', '').replace('[', '').replace(']', '')

        # --- Figure: Trajectories (single axis, all Vd on one plot) ---
        fig, ax = plt.subplots(figsize=(10, 6))

        # First plot ECDG (published) as dashed lines
        for Vd in Vd_list:
            tr_p = traj_data[Vd]['published']
            ax.plot(tr_p['x_A'], tr_p['y_A'], 'k--', lw=1.5, label='A (ECDG)' if Vd == Vd_list[0] else None)
            ax.plot(tr_p['x_D'], tr_p['y_D'], 'k-.', lw=1.5, label='D (ECDG)' if Vd == Vd_list[0] else None)
            ax.plot(tr_p['x_T'], tr_p['y_T'], 'k:', lw=1.8, marker='o', markevery=1000,
                    markerfacecolor='white', markeredgecolor='black', ms=4,
                    label='T (ECDG)' if Vd == Vd_list[0] else None)

        # Then plot C-ECDG (corrected) as solid lines with markers
        for Vd in Vd_list:
            tr_c = traj_data[Vd]['corrected']
            ax.plot(tr_c['x_A'], tr_c['y_A'], color='0.2', lw=1.2, marker='D', markevery=1000,
                    markerfacecolor='white', markeredgecolor='0.2', ms=5,
                    label=f'A (C-ECDG)' if Vd == Vd_list[0] else None)
            ax.plot(tr_c['x_D'], tr_c['y_D'], 'k-', lw=2.2, marker='s', markevery=1000,
                    markerfacecolor='white', markeredgecolor='black', ms=6,
                    label=f'D (C-ECDG)')
            ax.plot(tr_c['x_T'], tr_c['y_T'], 'k-', lw=1.5, marker='^', markevery=1000,
                    markerfacecolor='white', markeredgecolor='black', ms=5,
                    label=f'T (C-ECDG)' if Vd == Vd_list[0] else None)

        ax.set_title('Trajectory: ECDG vs C-ECDG', fontsize=14)
        ax.set_xlabel('$x$/m', fontsize=15)
        ax.set_ylabel('$y$/m', fontsize=15)
        ax.grid(True, alpha=0.6)
        # Legend position: right for Our case, left for Ref.[21] case
        legend_loc = 'lower right' if 'Our' in case_name else 'lower left'
        ax.legend(fontsize=12, loc=legend_loc, framealpha=0.9)
        plt.tight_layout()
        fig_path = os.path.join(out_dir, f"ecdg_trajectories_{safe_name}.png")
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Trajectory figure saved to: {fig_path}")

        # --- Figure: Overload comparison (single axis, D/T/A curves) ---
        fig, ax = plt.subplots(figsize=(10, 6))
        for Vd in Vd_list:
            tr_p = traj_data[Vd]['published']
            tr_c = traj_data[Vd]['corrected']

            # ECDG (solid lines, gray levels + solid markers)
            ax.plot(tr_p['t'], tr_p['a_D_g'], color='black', lw=2.2, linestyle='-',
                    marker='s', markevery=500, markersize=5, markerfacecolor='black', label='ECDG-D')
            ax.plot(tr_p['t'], tr_p['a_T_g'], color='0.3', lw=2.2, linestyle='-',
                    marker='^', markevery=500, markersize=5, markerfacecolor='0.3', label='ECDG-T')
            ax.plot(tr_p['t'], tr_p['a_A_g'], color='0.55', lw=2.2, linestyle='-',
                    marker='o', markevery=500, markersize=5, markerfacecolor='0.55', label='ECDG-A')

            # C-ECDG (solid lines, gray levels + open markers)
            ax.plot(tr_c['t'], tr_c['a_D_g'], color='black', lw=2.5, linestyle='-',
                    marker='s', markevery=500, markersize=5, markerfacecolor='white', label='C-ECDG-D')
            ax.plot(tr_c['t'], tr_c['a_T_g'], color='0.3', lw=2.5, linestyle='-',
                    marker='^', markevery=500, markersize=5, markerfacecolor='white', label='C-ECDG-T')
            ax.plot(tr_c['t'], tr_c['a_A_g'], color='0.55', lw=2.5, linestyle='-',
                    marker='o', markevery=500, markersize=5, markerfacecolor='white', label='C-ECDG-A')

        ax.set_title(f'Overload: ECDG vs C-ECDG ($V_d$={Vd_list[0]:.0f} m/s)', fontsize=14)
        ax.set_xlabel('Time (s)', fontsize=15)
        ax.set_ylabel('$N_y$ / g', fontsize=15)
        ax.grid(True, alpha=0.6)
        ax.legend(fontsize=11, loc='best', framealpha=0.9, ncol=2)
        plt.tight_layout()
        fig_path = os.path.join(out_dir, f"ecdg_overload_{safe_name}.png")
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Overload figure saved to: {fig_path}")
