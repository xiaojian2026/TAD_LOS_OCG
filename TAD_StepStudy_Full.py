# -*- coding: utf-8 -*-
"""
TAD_StepStudy_Full.py
=====================
Unified step-size / solver study for reviewer comment M1 (Round 3).

One program, one closed-loop Case-4 (LOS-OCG) model, five solvers:
    explicit Euler  dt = 1 ms, 0.5 ms, 0.25 ms, 0.125 ms
    + adaptive RK45 reference (Dormand-Prince 5(4), rtol=1e-9)

Nine configurations (R0 = 6000/8000/10000 m x Vd = 400/500/600 m/s),
Va = 1000 m/s (worst case), deterministic (no noise, no random maneuver).

Outputs two tables under Appendix_data/:
    Table_A1_StepStudy_MissDistance_Full.csv   (R_ad,min, m)
    Table_A2_StepStudy_PeakOverload_Full.csv   (defender peak |a_D,sf|/g, g)

Notes
-----
* u2 (target cooperative command) is ACTIVE in both the Euler and RK45 runs,
  because the production Case-4 closed loop includes it.
* USE_JERK = False, so the finite-difference terms (q_at_ddot, dot_a_A) do not
  enter the guidance command; the closed loop is a standard continuous ODE and
  RK45's fifth-order error control is meaningful.
* RK45 peak overload is evaluated over [0, t_closest] to match the Euler
  metric, which is taken up to the closest-approach index.
"""

import os
import sys
import csv
import time
import numpy as np
from math import atan2, cos, sin, exp, pi
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
from decimal import Decimal, ROUND_HALF_UP

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import TAD_Comp as tc

# ------------------------------------------------------------------ settings
CASE = 4
USE_JERK = False
VA_FIX = 1000.0
MISS_THRESHOLD = 1.0
DT_LIST = [0.001, 0.0005, 0.00025, 0.000125]
DT_LABELS = ['dt=1ms', 'dt=0.5ms', 'dt=0.25ms', 'dt=0.125ms']
CONFIGS = [(d, v) for d in (6000, 8000, 10000) for v in (400, 500, 600)]

g = tc.g
deg2rad = tc.deg2rad
rad2deg = tc.rad2deg
eps = tc.eps
R_PROTECT = tc.R_PROTECT
t_max = tc.t_max
tau_T, tau_A, tau_D = tc.tau_T, tc.tau_A, tc.tau_D
a_T_max, a_A_max = tc.a_T_max, tc.a_A_max
# Appendix A uses the 15 g defender limit of Sections 4 and 5. TAD_Comp's module
# default is 50 g, which belongs to the guidance-law comparison of Section 4.2.
a_D_max = 15.0 * g
tc.a_D_max = a_D_max


def _set_initial_geometry(dist, Vd, Va=None, alt=8000.0):
    """Same initial-condition convention as TAD_Comp.run_single_case."""
    tc.pos_A0[0] = float(dist)
    tc.pos_A0[1] = float(alt)
    tc.pos_T0 = tc.pos_D0.copy()
    tc.Va = (VA_FIX if Va is None else float(Va))
    tc.a_D_max = a_D_max
    return tc.pos_A0, tc.pos_D0, tc.pos_T0


# ================================================================== Euler
def run_euler(dist, Vd, dt, Va=None, alt=8000.0):
    _set_initial_geometry(dist, Vd, Va, alt)
    tc.dt = dt
    tc.USE_RK45_SOLVER = False   # force the legacy Euler kernel for this comparison
    t0 = time.time()
    res = tc.run_single_simulation(CASE, Vd, save_traj=False)
    steps = int(np.ceil(res['t_end'] / dt))   # effective integration steps
    return {
        'miss': res['miss_final_m'],
        'peak': res['a_D_sf_peak_g'],
        'aTpeak': res['a_T_sf_peak_g'],
        'mode': res['termination_mode'],
        't_end': res['t_end'],
        'cost': time.time() - t0,      # wall time (s)
        'steps': steps,                # Euler: one derivative evaluation per step
        'nfev': steps,
    }


# ================================================================== RK45
def make_rhs(Vd):
    params = tc.get_guidance_params(CASE, Vd)
    Vt = tc.Vt
    Va = tc.Va
    pos_A0 = tc.pos_A0
    pos_D0 = tc.pos_D0
    pos_T0 = tc.pos_T0

    dx0 = pos_A0[0] - pos_D0[0]
    dy0 = pos_A0[1] - pos_D0[1]
    th_D0 = (atan2(dy0, dx0) - 6 * deg2rad) if pos_A0[1] >= pos_D0[1] else (atan2(dy0, dx0) + 6 * deg2rad)
    th_T0 = th_D0
    th_A0 = pi

    freeze_zeae = [False, 0.0, 0.0]

    def deriv(t, y):
        x_T, y_T, th_T, a_T_sf, x_A, y_A, th_A, a_A_sf, x_D, y_D, th_D, a_D_sf = y
        th_T = tc.wrap_pi(th_T); th_A = tc.wrap_pi(th_A); th_D = tc.wrap_pi(th_D)

        dx_at = x_A - x_T; dy_at = y_A - y_T
        R_at = np.hypot(dx_at, dy_at)
        q_at = tc.wrap_pi(atan2(dy_at, dx_at))
        vx_at = Va*cos(th_A) - Vt*cos(th_T); vy_at = Va*sin(th_A) - Vt*sin(th_T)
        R_at_dot = (dx_at*vx_at + dy_at*vy_at) / (R_at + eps)
        q_at_dot = (dx_at*vy_at - dy_at*vx_at) / (R_at**2 + eps)

        dx_ad = x_A - x_D; dy_ad = y_A - y_D
        R_ad = np.hypot(dx_ad, dy_ad)
        q_ad = tc.wrap_pi(atan2(dy_ad, dx_ad))
        vx_ad = Va*cos(th_A) - Vd*cos(th_D); vy_ad = Va*sin(th_A) - Vd*sin(th_D)
        R_ad_dot = (dx_ad*vx_ad + dy_ad*vy_ad) / (R_ad + eps)
        q_ad_dot = (dx_ad*vy_ad - dy_ad*vx_ad) / (R_ad**2 + eps)

        dx_td = x_D - x_T; dy_td = y_D - y_T
        q_dt = tc.wrap_pi(atan2(dy_td, dx_td))
        R_td = np.hypot(dx_td, dy_td)
        vx_td = Vd*cos(th_D) - Vt*cos(th_T); vy_td = Vd*sin(th_D) - Vt*sin(th_T)
        q_dt_dot = (dx_td*vy_td - dy_td*vx_td) / (R_td**2 + eps)
        R_td_dot = (dx_td*vx_td + dy_td*vy_td) / (R_td + eps)

        if R_ad_dot < -1e-3:
            tgo = -R_ad / R_ad_dot
        else:
            tgo = 0.01
        tgo = max(tgo, 0.001)

        delta_d = R_ad * sin(th_A - q_ad)
        theta_a_dot = (a_A_sf - g*cos(th_A)) / Va
        delta_d_dot = -Vd*sin(th_A - th_D) + R_ad*cos(th_A - q_ad)*theta_a_dot
        x1, x2, x3, x4, x5 = delta_d, delta_d_dot, q_at, q_at_dot, th_T

        I1 = -2.0*R_at_dot / R_at
        I2 = cos(th_A - q_at) / R_at
        I3 = -cos(th_T - q_at) / R_at
        K = cos(th_A - th_D)
        a_A = a_A_sf - g*cos(th_A)

        if abs(I1) < 1e-6:
            phi = tgo; phi2 = tgo**2/2.0; phi22 = tgo**3/3.0
        else:
            arg = I1 * tgo
            max_arg = 350.0
            arg = max(-max_arg, min(arg, max_arg))
            e1 = exp(arg); e2 = exp(2*arg)
            phi = (e1 - 1.0) / I1
            phi2 = (e1 - I1*tgo - 1.0) / (I1**2)
            phi22 = (e2 - 4.0*e1 + 2.0*I1*tgo + 3.0) / (2.0*I1**3)

        alpha_base = params.get('alpha', 1.0)
        alpha = alpha_base * (20 ** min(max(tgo*0.8, 1.0), 5))
        a_weight, m_weight, n_weight = tc.compute_dynamic_weights(t, params, CASE)

        sigma = (1.0 - 2.0*Vd/Va*cos(th_A - th_D))*a_A \
                - sin(th_A - q_ad)*R_ad*(a_A**2)/(Va**2)

        ZEM = x1 + x2*tgo
        ZEAE1_raw = x3 + x4*phi - q_ad
        theta_s = params['theta_s']
        if pos_A0[1] >= pos_D0[1]:
            ZEAE2_raw = th_A - x5 + theta_s
        else:
            ZEAE2_raw = th_A - x5 + theta_s - pi
        ZEAE2_raw = tc.wrap_pi(ZEAE2_raw)

        if not freeze_zeae[0] and R_ad < R_PROTECT:
            freeze_zeae[0] = True
            freeze_zeae[1] = ZEAE1_raw
            freeze_zeae[2] = ZEAE2_raw
        if freeze_zeae[0]:
            ZEAE1, ZEAE2 = freeze_zeae[1], freeze_zeae[2]
        else:
            ZEAE1, ZEAE2 = ZEAE1_raw, ZEAE2_raw

        geom = {'Vd':Vd, 'Vt':Vt,
                'R_ad':R_ad, 'R_ad_dot':R_ad_dot, 'q_ad':q_ad, 'q_ad_dot':q_ad_dot,
                'R_at':R_at, 'R_at_dot':R_at_dot, 'q_at':q_at, 'q_at_dot':q_at_dot,
                'q_dt':q_dt, 'q_dt_dot':q_dt_dot,
                'R_td':R_td, 'R_td_dot':R_td_dot,
                'th_A':th_A, 'th_D':th_D, 'th_T':th_T,
                'err_los': q_at - q_dt, 'err_los_dot': q_at_dot - q_dt_dot,
                'q_at_ddot': 0.0, 'filt_x': 0.0, 't': t,
                'x_A':x_A, 'y_A':y_A, 'x_D':x_D, 'y_D':y_D, 'x_T':x_T, 'y_T':y_T}
        coeff = {'I1':I1, 'I2':I2, 'I3':I3, 'K':K, 'a_A':a_A}
        state = {'ZEM':ZEM, 'ZEAE1':ZEAE1, 'ZEAE2':ZEAE2,
                 'phi':phi, 'phi2':phi2, 'phi22':phi22}
        u_D, u_T = tc.guidance_case4(params, geom, coeff, state, tgo, sigma, alpha,
                                     a_weight, m_weight, n_weight)
        u_A = 3.0*Va*q_at_dot

        u_D_cmd = np.clip(u_D + g*cos(th_D), -a_D_max, a_D_max)
        u_T_cmd = np.clip(u_T + g*cos(th_T), -a_T_max, a_T_max)
        u_A_cmd = np.clip(u_A + g*cos(th_A), -a_A_max, a_A_max)

        dy = np.zeros(12)
        dy[0] = Vt*cos(th_T);  dy[1] = Vt*sin(th_T)
        dy[2] = (a_T_sf - g*cos(th_T)) / Vt
        dy[3] = (u_T_cmd - a_T_sf) / tau_T
        dy[4] = Va*cos(th_A);  dy[5] = Va*sin(th_A)
        dy[6] = (a_A_sf - g*cos(th_A)) / Va
        dy[7] = (u_A_cmd - a_A_sf) / tau_A
        dy[8] = Vd*cos(th_D);  dy[9] = Vd*sin(th_D)
        dy[10] = (a_D_sf - g*cos(th_D)) / Vd
        dy[11] = (u_D_cmd - a_D_sf) / tau_D
        return dy

    y0 = np.array([pos_T0[0], pos_T0[1], th_T0, 0.0,
                   pos_A0[0], pos_A0[1], th_A0, 0.0,
                   pos_D0[0], pos_D0[1], th_D0, 0.0])
    return deriv, y0


def _intercept_event(t, y):
    return np.hypot(y[8] - y[4], y[9] - y[5]) - MISS_THRESHOLD
_intercept_event.direction = -1.0
_intercept_event.terminal = False

def _target_hit_event(t, y):
    return np.hypot(y[0] - y[4], y[1] - y[5]) - MISS_THRESHOLD
_target_hit_event.direction = -1.0
_target_hit_event.terminal = True

_VD_GLOBAL = [400.0]
def _closest_event(t, y):
    th_A = y[6]; th_D = y[10]
    dx = y[4] - y[8]; dy = y[5] - y[9]
    R = np.hypot(dx, dy) + eps
    vx = tc.Va*cos(th_A) - _VD_GLOBAL[0]*cos(th_D)
    vy = tc.Va*sin(th_A) - _VD_GLOBAL[0]*sin(th_D)
    return (dx*vx + dy*vy) / R
_closest_event.direction = 1.0
_closest_event.terminal = False


def run_rk45(dist, Vd, rtol=1e-9, atol=1e-11, Va=None, alt=8000.0):
    """Production RK45 kernel of TAD_Comp.

    The reference solution is now taken from the same implementation that
    produces every result in Sections 4 and 5, so the comparison validates the
    production solver rather than a private re-implementation.  The legacy
    hand-written RHS above (make_rhs) is kept only for cross-checking and is no
    longer used to generate any table.
    """
    _set_initial_geometry(dist, Vd, Va, alt)
    tc.USE_RK45_SOLVER = True
    tc.RK45_RTOL = rtol
    tc.RK45_ATOL = atol
    t0 = time.time()
    res = tc.run_single_simulation(CASE, Vd, save_traj=False)
    return {
        'miss': res['miss_final_m'],
        'peak': res['a_D_sf_peak_g'],
        'aTpeak': res['a_T_sf_peak_g'],
        'mode': res['termination_mode'],
        't_end': res['t_end'],
        't_miss': res['t_end'],
        'cost': time.time() - t0,
        'steps': int(res.get('nsteps', 0)),
        'nfev': int(res.get('nfev', 0)),
    }


# ================================================================== main
def main():
    outdir = os.path.join(HERE, 'Appendix_data')
    os.makedirs(outdir, exist_ok=True)

    miss_tab = {}   # (dist,Vd) -> {label: value}
    peak_tab = {}
    cost_tab = {}   # (dist,Vd) -> {label: dict(steps, nfev, cost, t_end)}

    print("=" * 104)
    print("Unified study: Euler (1 / 0.5 / 0.25 / 0.125 ms) vs RK45 | Case 4 | Va=1000 m/s")
    print("=" * 104)

    for idx, (dist, Vd) in enumerate(CONFIGS, 1):
        key = (dist, Vd)
        miss_tab[key] = {}
        peak_tab[key] = {}
        cost_tab[key] = {}

        for dt, lab in zip(DT_LIST, DT_LABELS):
            r = run_euler(dist, Vd, dt)
            miss_tab[key][lab] = r['miss']
            peak_tab[key][lab] = r['peak']
            cost_tab[key][lab] = r
            print(f"[{idx}/9] {dist}/{Vd}  {lab:<10} miss={r['miss']:.4f} m  "
                  f"peak={r['peak']:.3f} g  steps={r['steps']}  ({r['cost']:.1f}s)", flush=True)

        r = run_rk45(dist, Vd)
        miss_tab[key]['RK45'] = r['miss']
        peak_tab[key]['RK45'] = r['peak']
        cost_tab[key]['RK45'] = r
        print(f"[{idx}/9] {dist}/{Vd}  {'RK45':<10} miss={r['miss']:.4f} m  "
              f"peak={r['peak']:.3f} g  steps={r['steps']}  nfev={r['nfev']}  "
              f"({r['cost']:.1f}s)", flush=True)

    cols = DT_LABELS + ['RK45']

    def f2(x):
        """Two-decimal, half-up rounding immune to binary floating-point noise.

        Some values land exactly on a .xx5 boundary (4.465, 2.665); plain
        '%.2f' formatting then flips between the two neighbours depending on
        whether the number was just computed or re-read from a CSV.  Snapping
        to six decimals first and quantizing half-up keeps the table values
        stable and identical to the published ones.
        """
        return str(Decimal(repr(round(float(x), 6))).quantize(
            Decimal('0.01'), rounding=ROUND_HALF_UP))

    def _write(fname, tab, fmt, title):
        path = os.path.join(outdir, fname)

        def _dump(target):
            with open(target, 'w', newline='', encoding='utf-8-sig') as f:
                w = csv.writer(f)
                w.writerow(['Configuration'] + cols)
                for (dist, Vd) in CONFIGS:
                    row = [f"R0={dist}m, Vd={Vd}m/s"]
                    row += [fmt(tab[(dist, Vd)][c]) for c in cols]
                    w.writerow(row)

        try:
            _dump(path)
            print(f"\nSaved: {path}")
        except PermissionError:
            alt = path[:-4] + '_v2.csv'
            _dump(alt)
            print(f"\n[WARN] {path} is locked by another process; wrote {alt}")

    _write('Table_A1_StepStudy_MissDistance_Full.csv', miss_tab, f2,
           'R_ad,min (m)')
    _write('Table_A2_StepStudy_PeakOverload_Full.csv', peak_tab, f2,
           'peak |a_D,sf| (g)')

    # ---- Table A3: steps / cost / accuracy for every configuration and solver ----
    dt_map = dict(zip(DT_LABELS, DT_LIST))
    a3_detail = os.path.join(outdir, 'Detail_PerConfiguration_Cost.csv')
    with open(a3_detail, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['Configuration', 'Method', 'dt_ms', 'Steps',
                    'DerivativeEvaluations', 'WallTime_s', 't_miss_s',
                    'MissError_vs_RK45_m', 'PeakError_vs_RK45_g'])
        for (dist, Vd) in CONFIGS:
            bm = miss_tab[(dist, Vd)]['RK45']
            bp = peak_tab[(dist, Vd)]['RK45']
            for lab in cols:
                c = cost_tab[(dist, Vd)][lab]
                dt_s = 'adaptive' if lab == 'RK45' else ('%g' % (dt_map[lab] * 1000))
                w.writerow([f"R0={dist}m, Vd={Vd}m/s", lab, dt_s,
                            c['steps'], c['nfev'], '%.2f' % c['cost'],
                            '%.4f' % c.get('t_miss', c['t_end']),
                            f2(abs(miss_tab[(dist, Vd)][lab] - bm)),
                            f2(abs(peak_tab[(dist, Vd)][lab] - bp))])
    print(f"\nSaved: {a3_detail}")

    # ---- Table A4: aggregate accuracy vs computational cost ----
    summ = {}
    for lab in cols:
        max_m = max(abs(miss_tab[k][lab] - miss_tab[k]['RK45']) for k in CONFIGS)
        max_p = max(abs(peak_tab[k][lab] - peak_tab[k]['RK45']) for k in CONFIGS)
        tot_steps = sum(cost_tab[k][lab]['steps'] for k in CONFIGS)
        tot_nfev = sum(cost_tab[k][lab]['nfev'] for k in CONFIGS)
        tot_wall = sum(cost_tab[k][lab]['cost'] for k in CONFIGS)
        summ[lab] = (max_m, max_p, tot_steps, tot_nfev, tot_wall)

    a3_path = os.path.join(outdir, 'Table_A3_AccuracyVsCost_Summary.csv')
    with open(a3_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['Method', 'dt_ms', 'MaxMissError_m', 'MaxPeakError_g',
                    'TotalSteps_9configs', 'TotalDerivativeEvaluations', 'TotalWallTime_s'])
        for lab in cols:
            dt_s = 'adaptive' if lab == 'RK45' else ('%g' % (dt_map[lab] * 1000))
            mm, mp, ts, tn, tw = summ[lab]
            w.writerow([lab, dt_s, f2(mm), f2(mp), ts, tn, '%.1f' % tw])
    print(f"Saved: {a3_path}")

    # ---- Table A4: eight grid configurations near the 1 m threshold ----
    # Published format (same layout as Table A1/A2): one row per configuration,
    # columns = Configuration | dt=1ms | dt=0.5ms | dt=0.25ms | dt=0.125ms | RK45.
    THRESH = [(600, 500), (600, 1000), (600, 1500), (600, 2000),
              (650, 500), (650, 1000), (650, 1500), (700, 1000)]
    R0_T, VD_T = 5000.0, 400.0
    t4_path = os.path.join(outdir, 'Table_A4_ThresholdCases.csv')
    t4_detail = os.path.join(outdir, 'Table_A4_ThresholdCases_detail.csv')
    a4_cols = DT_LABELS + ['RK45']
    print("\n" + "-" * 104)
    print(f"Threshold cases: R0 = {R0_T:.0f} m, Vd = {VD_T:.0f} m/s, 8 configurations")
    print("-" * 104)
    rows4 = []
    for Va_t, dh in THRESH:
        alt = 5000.0 + dh   # attacker altitude = target altitude (5000 m) + dh
        res = {}
        for dt, lab in zip(DT_LIST, DT_LABELS):
            res[lab] = run_euler(R0_T, VD_T, dt, Va=Va_t, alt=alt)
        res['RK45'] = run_rk45(R0_T, VD_T, Va=Va_t, alt=alt)
        for lab in a4_cols:
            r = res[lab]
            print(f"  Va={Va_t:<4} dh=+{dh:<5} {lab:<10} miss={r['miss']:.4f} m  "
                  f"{r['mode']:<12} aT={r['aTpeak']:.3f} g", flush=True)
        rows4.append((f"V_a={Va_t}m/s, \u0394h=+{dh}m", res))
    with open(t4_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['Configuration'] + a4_cols)
        for label, res in rows4:
            w.writerow([label] + [f2(res[lab]['miss']) for lab in a4_cols])
    with open(t4_detail, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['Configuration', 'Solver', 'Miss_m', 'Termination',
                    'Peak_aT_g', 'Nfev'])
        for label, res in rows4:
            for lab in a4_cols:
                r = res[lab]
                w.writerow([label, lab, f2(r['miss']), r['mode'],
                            f2(r['aTpeak']), r['nfev']])
    print(f"Saved: {t4_path}")
    print(f"Saved: {t4_detail}")

    print("\n" + "=" * 104)
    print("Aggregate: accuracy vs computational cost (9 configurations)")
    print("-" * 104)
    print(f"{'Method':<12}{'Max|d miss|':>14}{'Max|d peak|':>14}"
          f"{'TotalSteps':>13}{'TotalDerivEval':>16}{'Wall(s)':>10}")
    print("-" * 104)
    for lab in cols:
        mm, mp, ts, tn, tw = summ[lab]
        print(f"{lab:<12}{mm:>14.4f}{mp:>14.4f}{ts:>13d}{tn:>16d}{tw:>10.1f}")
    print("=" * 104)
    print("Note: Euler performs one derivative evaluation per step; RK45 reports nfev,")
    print("which counts every RHS evaluation, including rejected steps.")
    print("=" * 104)


if __name__ == '__main__':
    main()
