# -*- coding: utf-8 -*-
"""
Title: Terminal Cooperative Guidance Based on LOS Geometry and Optimal Control for TAD Engagement

Section 4.2: Comparative Simulation of Four Guidance Laws

This script implements and compares four guidance laws for the target-attacker-defender
(TAD) engagement scenario:
  Case 1 (APN): Augmented proportional navigation - traditional baseline
  Case 2 (LOSIG): Line-of-Sight Interceptor Guidance - cooperative baseline (Ratnoo & Shima, 2012)
  Case 3 (C-ECDG): Explicit Cooperative Defense Guidance (Li et al., 2022) - CORRECTED VERSION
                 Three corrections to the published paper (see Appendix B):
                   (1) Missing lambda_dot_ij factor in diagonal elements A33/A44
                   (2) Wrong sign and denominator in coupling coefficient A23
                   (3) Wrong time constant and sign in Eq.(34) closed-form phi
  Case 4 (LOS-OCG): Proposed LOS-based Optimal Cooperative Guidance - this paper

For each case, simulations are run at three defender velocities (400, 500, 600 m/s).
Performance metrics include terminal miss distance (computed via cubic spline
interpolation with sub-step precision), peak overload, and maneuver effort.
Trajectory plots, overload time histories, and summary tables are generated.
"""

import numpy as np
import matplotlib.pyplot as plt

from math import atan2, cos, sin, exp, pi
import os
import csv
import time
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, Border, Side
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar

# Guidance law abbreviations
CASE_ABBR = {
    1: 'APN',      # Augmented Proportional Navigation
    2: 'LOSIG',    # Line-of-Sight Interceptor Guidance (Ratnoo & Shima, 2012)
    3: 'C-ECDG',   # Corrected Explicit Cooperative Defense Guidance (corrected from Li et al., 2022)
    4: 'LOS-OCG'   # Proposed: LOS-based Optimal Cooperative Guidance
}

# Global switch: include jerk term (dot_a_A_sf) in sigma prediction
USE_JERK = False


# ===================== Common utility: Sub-step miss distance via cubic spline =====================
def calc_interpolated_miss(t_arr, xA_arr, yA_arr, xD_arr, yD_arr, min_idx, half_span=4):
    """
    Compute high-precision minimum miss distance using cubic spline interpolation
    and bounded Brent optimization around the discrete minimum point.

    Parameters
    ----------
    t_arr : array-like
        Time array.
    xA_arr, yA_arr : array-like
        Attacker (incoming missile) position history.
    xD_arr, yD_arr : array-like
        Defender missile position history.
    min_idx : int
        Index of the discrete minimum distance point.
    half_span : int
        Number of points to take on each side of min_idx for interpolation.

    Returns
    -------
    miss_final : float
        High-precision minimum miss distance.
    """
    # Discrete miss distance at the minimum index (fallback value)
    miss_disc = np.sqrt((xD_arr[min_idx] - xA_arr[min_idx]) ** 2 + (yD_arr[min_idx] - yA_arr[min_idx]) ** 2)
    N = len(t_arr)

    # Check if the interpolation window is within valid bounds
    if half_span <= min_idx <= N - half_span - 1:
        idxs = range(min_idx - half_span, min_idx + half_span + 1)
        ts = t_arr[idxs]
        xA_vals = xA_arr[idxs]
        yA_vals = yA_arr[idxs]
        xD_vals = xD_arr[idxs]
        yD_vals = yD_arr[idxs]

        try:
            # Fit natural cubic splines for each trajectory component
            spline_xA = CubicSpline(ts, xA_vals, bc_type='natural')
            spline_yA = CubicSpline(ts, yA_vals, bc_type='natural')
            spline_xD = CubicSpline(ts, xD_vals, bc_type='natural')
            spline_yD = CubicSpline(ts, yD_vals, bc_type='natural')

            def dist_func(t):
                """Relative distance between attacker and defender at time t."""
                xA = spline_xA(t)
                yA = spline_yA(t)
                xD = spline_xD(t)
                yD = spline_yD(t)
                return np.sqrt((xD - xA) ** 2 + (yD - yA) ** 2)

            # Bounded Brent optimization to find sub-step minimum
            res = minimize_scalar(dist_func, bounds=(ts[0], ts[-1]), method='bounded')
            if res.success:
                return res.fun
        except Exception:
            pass

    # Fallback 1: Three-point quadratic polynomial fit if spline window invalid
    if 1 <= min_idx <= N - 2:
        idxs2 = [min_idx - 1, min_idx, min_idx + 1]
        ts2 = t_arr[idxs2]
        xA2 = xA_arr[idxs2]
        yA2 = yA_arr[idxs2]
        xD2 = xD_arr[idxs2]
        yD2 = yD_arr[idxs2]
        try:
            p_xA2 = np.polyfit(ts2, xA2, 2)
            p_yA2 = np.polyfit(ts2, yA2, 2)
            p_xD2 = np.polyfit(ts2, xD2, 2)
            p_yD2 = np.polyfit(ts2, yD2, 2)
            t_fine2 = np.linspace(ts2[0], ts2[-1], 5000)
            xA_f2 = np.polyval(p_xA2, t_fine2)
            yA_f2 = np.polyval(p_yA2, t_fine2)
            xD_f2 = np.polyval(p_xD2, t_fine2)
            yD_f2 = np.polyval(p_yD2, t_fine2)
            dist_f2 = np.sqrt((xD_f2 - xA_f2) ** 2 + (yD_f2 - yA_f2) ** 2)
            return np.min(dist_f2)
        except Exception:
            pass

    # Fallback 2: Return discrete minimum if all interpolation methods fail
    return miss_disc


# ====================== 1. Global parameters ======================
g0 = 9.81           # Standard gravitational acceleration at sea level (m/s^2)
Re = 6371000.0      # Earth radius (m)
H0 = 0.0            # Reference altitude (m)
# Gravity correction for altitude (flat-Earth approximation)
g = g0 * np.sqrt(Re / (Re + H0))
deg2rad = pi / 180  # Degrees to radians conversion
rad2deg = 180 / pi  # Radians to degrees conversion
eps = 1e-6          # Small epsilon to avoid division by zero
R_PROTECT = 100     # Protection radius for freezing ZEAE values (m)
TGO_MAX = 20.0      # Upper bound on time-to-go to suppress blow-up near the closing-rate root (s)
dt = 0.00025        # Simulation time step (s) - 0.25 ms for numerical accuracy
t_max = 15.0        # Maximum simulation time (s)


def wrap_pi(angle):
    """Wrap an angle in radians to the interval [-pi, pi].
    Positive direction is counterclockwise from the +X axis."""
    return angle - 2.0 * np.pi * np.round(angle / (2.0 * np.pi))


# Attacker (incoming missile) A parameters (fixed)
Va = 600.0                          # Attacker velocity (m/s)
pos_A0 = np.array([6000.0, 8000.0]) # Initial position [x, y] (m)
theta_A_sf0 = -180.0 * deg2rad         # Initial flight path angle (rad), heading toward -X
a_A_max = 50.0 * g                  # Maximum attacker acceleration (m/s^2)
tau_A = 0.2                         # Attacker acceleration first-order lag time constant (s)

# Defender missile D parameters (velocity is variable across test cases)
Vd_list = [400.0, 500.0, 600.0]     # Defender velocities to test (m/s)
pos_D0 = np.array([0.0, 5000.0])    # Initial defender position [x, y] (m)
a_D_max = 50.0 * g                  # Maximum defender acceleration (m/s^2)
tau_D = 0.2                         # Defender acceleration lag time constant (s)

# Target aircraft T parameters (fixed, co-located with defender at launch)
Vt = 270.0                          # Target aircraft velocity (m/s)
pos_T0 = pos_D0.copy()              # Target starts at same position as defender
a_T_max = 6.0 * g                   # Maximum target acceleration (m/s^2), limited by aircraft maneuverability
tau_T = 0.5                         # Target acceleration lag time constant (s)


# ====================== 2. Guidance law parameters and functions ======================
def get_guidance_params(CASE, Vd):
    """Return guidance law parameters for the specified case.

    Case 1: Augmented proportional navigation (APN)
    Case 2: LOS Defender guidance
    Case 3: Li et al. (2022) Optimal Cooperative LOS Guidance
    Case 4: Proposed cooperative guidance law based on LOS geometry and optimal control
    """
    params = {}
    if CASE == 1:
        params['N_PN'] = 4  # PN navigation constant
    elif CASE == 2:
        params['K_clos'] = 80      # Closest approach guidance gain
        params['a_lead'] = 40.0    # Lead angle filter parameter a
        params['b_lead'] = 75.0    # Lead angle filter parameter b
    elif CASE == 3:
        # Li et al. (2022) Optimal Cooperative LOS Guidance
        params['R_D_weight'] = 10.0
        params['R_T_weight'] = 1.0
    elif CASE == 4:
        # Proposed cooperative guidance weights (dynamic, ramp up via tanh)
        params['a_max'] = 3e4; params['m_max'] = 1e5; params['n_max'] = 1e5
        params['alpha'] = 1; params['beta'] = 1; params['theta_s'] = pi/2  # Desired intercept angle
        params['m1'] = 2; params['m1_a'] = 4  # Time constants for weight ramping
    else:
        raise ValueError("CASE must be 1, 2, 3, or 4")
    return params


def compute_dynamic_weights(t, params, CASE):
    """Compute time-varying guidance weights using tanh ramping functions.

    For Case 4 (proposed method), weights ramp from 0 to their maximum values
    as time progresses, ensuring smooth engagement initiation and full authority
    near intercept.
    """
    if CASE == 4:
        m1 = params['m1']; m1_a = params['m1_a']
        # tanh(x) = (e^x - e^-x) / (e^x + e^-x), smooth sigmoid from 0 to 1
        tanh_val_mn = (exp(t/m1) - exp(-t/m1)) / (exp(t/m1) + exp(-t/m1))
        tanh_val_a = (exp(t/m1_a) - exp(-t/m1_a)) / (exp(t/m1_a) + exp(-t/m1_a))
        a_weight = params['a_max'] * tanh_val_a
        m_weight = params['m_max'] * tanh_val_mn
        n_weight = params['n_max'] * tanh_val_mn
    else:
        # Fixed weights for other cases
        a_weight = 5e3; m_weight = 1e9; n_weight = 2e7
    return a_weight, m_weight, n_weight


def guidance_case1(params, geom, coeff):
    """Case 1: Augmented Proportional Navigation (APN) guidance.

    u1 = N * |R_dot| * q_dot + (N/2) * a_A_sf_perp
    The first term is standard PN; the second term compensates for the
    component of attacker acceleration perpendicular to the LOS, making it
    augmented PN (APN) rather than pure PN.
    u2 = -6g  (target performs constant evasive maneuver)
    """
    N_PN = params['N_PN']
    a_A = coeff['a_A']
    # Component of attacker acceleration perpendicular to LOS (for APN augmentation)
    a_A_perp = a_A * np.cos(geom['th_A'] - geom['q_ad'])
    # APN command: standard PN + attacker acceleration feedforward
    u_D = N_PN * abs(geom['R_ad_dot']) * geom['q_ad_dot'] + (N_PN / 2.0) * a_A_perp
    # Target evasive maneuver command (constant -6g)
    u_T = -a_T_max
    return u_D, u_T


def guidance_case2(params, geom, coeff, state, tgo, sigma, alpha):
    """Case 2: LOSIG - Line-of-Sight Interceptor Guidance (Ratnoo & Shima, 2012).

    Uses a second-order filter to estimate LOS rate derivative for improved
    intercept accuracy against maneuvering targets.
    """
    K_clos = params['K_clos']
    a_lead = params['a_lead']
    b_lead = params['b_lead']
    err_los = geom['err_los']
    err_los_dot = geom['err_los_dot']
    R_td = geom['R_td']
    R_td_dot = geom['R_td_dot']
    q_at_ddot = geom['q_at_ddot']
    th_T = geom['th_T']
    q_at = geom['q_at']
    filt_x = geom['filt_x']
    # Second-order lead filter state update
    dxdt_filt = -b_lead * filt_x + (b_lead - a_lead)*err_los_dot + a_lead * err_los
    filt_x_new = filt_x + dxdt_filt * dt
    filter_out = filt_x_new
    # Determine target maneuver direction based on angle between target velocity and LOS
    ang_diff = th_T - q_at
    if (-np.pi/2 >= ang_diff) or (ang_diff >= np.pi/2):
        u_T = 2.0 * R_td_dot * geom['q_at_dot']
    else:
        u_T = -2.0 * R_td_dot * geom['q_at_dot']
    u_T = np.clip(u_T, -a_T_max, a_T_max)
    # Inertial-frame acceleration command
    a_d_inertial = K_clos * R_td * filter_out + u_T + R_td * q_at_ddot + 2.0 * R_td_dot * geom['q_at_dot']
    # Transform to defender body-frame (perpendicular to velocity)
    e_perp_x = -np.sin(q_at)
    e_perp_y = np.cos(q_at)
    e_nD_x = -np.sin(geom['th_D'])
    e_nD_y = np.cos(geom['th_D'])
    u_D = a_d_inertial * (e_perp_x * e_nD_x + e_perp_y * e_nD_y)
    return u_D, u_T, filt_x_new


def _ecdg_zero_effort_terms(geom, u_M, th_A, th_D, t_gDM, dti=0.05):
    """Trapezoidal numerical integration along the zero-effort nominal trajectory:
    I_D = INT b_d^2 dtau, I_T = INT b_t^2 dtau, I_M = INT b_a*u_M dtau.

    Nominal: D/T fly straight, A turns with current lateral acceleration u_M frozen
    (Remark 3 of the paper); integration horizon is t_fDM = t + t_gDM, stops when
    closure velocity is exhausted or range becomes too small.
    At each prediction point: phi is recomputed from the corrected closed-form
    Eq.(34) (Correction 3), b_d/b_t/b_a follow Eqs.(35)(37), and t_go is derived
    from the instantaneous geometry r/V per Eq.(7). Matches zeroEffortTerms in
    icdg_vs_ecdg.m.
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
    while tau <= t_gDM + dti:
        tg = max(t_gDM - tau, 1e-6)
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
            break                                   # Closure velocity exhausted / too close: stop accumulation
        t_gDMn = R_DMn / V_DMn                       # Eq.(7)
        t_gTMn = R_TMn / V_TMn
        dt_fn = t_gTMn - t_gDMn
        # ---- Correction 3: Closed-form of Eq.(34) (pointwise recomputation) ----
        if dt_fn > 0.0:
            x = 2.0 * t_gDMn / t_gTMn
            if x > 25.0:
                phi_n = 0.0
            else:
                phi_n = (V_DMn * dt_fn / 2.0) * (2.0 * t_gDMn - t_gTMn * (exp(x) - 1.0))
        else:
            phi_n = 0.0
        lDMn = atan2(dy_ad, dx_ad)
        lTMn = atan2(dy_at, dx_at)
        # ---- Eq.(35)(37) strictly per paper ----
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


def guidance_case3(params, geom, coeff, state, tgo, sigma):
    """Case 3: C-ECDG — Corrected ECDG (Li et al., 2022), with three corrections to the published paper.

    Three corrections (see derivation notes):
      Correction 1: Eqs.(5)(6) — missing λ̇_ij factor in the first term
                    (linearized diagonal elements A33=A44=+2/t_g)
      Correction 2: Eqs.(10)(12) — coupling coefficient A23 = -2·V_DM·Δt_f/t_gTM
                    (paper erroneously printed +2·V_DM·Δt_f/t_gDM, both sign and
                    denominator are wrong)
      Correction 3: Eq.(34) — φ = (V_DM·Δt_f/2)·[2·t_gDM − t_gTM·(e^{2t_gDM/t_gTM}−1)]
                    (time constant should be t_gTM, overall sign negative;
                    paper erroneously printed t+Δt_f and positive sign)

    Strictly follows the printed paper formulas (no additional corrections):
      Eq.(8)   z  = R_DM·(q_at − q_ad)                          [small-angle form]
      Eq.(9)   ż  = Ṙ_DM·(q_at − q_ad) + R_DM·(q̇_at − q̇_ad)
      Eq.(16)  t2 = −(R_DM/R_TM)·cos(θ_t−q_at),  t3 = −cos(θ_t−q_at)/R_TM
               m2 = (R_DM/R_TM)·cos(θ_a−q_at) − cos(θ_a−q_ad), m3 = cos(θ_a−q_at)/R_TM
      Eq.(35)  b_d = cos(θ_d − q_ad)·t_gDM
      Eq.(37)  b_t = t2·t_gDM + t3·φ,   b_a = m2·t_gDM + m3·φ
      Eq.(47)  Z(t) = z + ż·t_gDM + φ·q̇_at + ∫ b_a·a_a dτ   (u_M frozen at current value)
      Eq.(46)  den = ∫(b_d²/R_D + b_t²/R_T)dτ
               u1 = −(b_d/R_D)·Z(t)/den,   u2 = −(b_t/R_T)·Z(t)/den
    The paper does not specify the integration method; we use trapezoidal integration
    along the zero-effort nominal trajectory (_ecdg_zero_effort_terms).
    """
    R_DM = geom['R_ad']                 # r_DM
    R_TM = geom['R_at']                 # r_TM
    l_DM = geom['q_ad']                 # lam_DM
    l_TM = geom['q_at']                 # lam_TM
    ld_DM = geom['q_ad_dot']            # lam_DM_dot
    ld_TM = geom['q_at_dot']            # lam_TM_dot
    th_D = geom['th_D']                 # gamma_D
    th_T = geom['th_T']                 # gamma_T
    th_A = geom['th_A']                 # gamma_M
    u_M = coeff['a_A']                  # Attacker current lateral acceleration (frozen, Remark 3)

    R_DM_dot = geom['R_ad_dot']
    V_DM = -R_DM_dot                    # Paper definition: V_DM = -rdot_DM > 0
    V_TM = -geom['R_at_dot']
    if V_DM <= 1.0 or V_TM <= 1.0 or R_DM < 1.0:
        return 0.0, 0.0
    t_gDM = R_DM / V_DM                 # Eq.(7)
    t_gTM = R_TM / V_TM
    dt_f = t_gTM - t_gDM                # Δt_f = t_fTM - t_fDM
    if t_gDM < 0.05:
        return 0.0, 0.0

    # ---------- Correction 3: Closed-form of Eq.(34) ----------
    if dt_f > 0.0:
        x = 2.0 * t_gDM / t_gTM
        if x > 25.0:
            phi = 0.0
        else:
            phi = (V_DM * dt_f / 2.0) * (2.0 * t_gDM - t_gTM * (exp(x) - 1.0))
    else:
        phi = 0.0

    # ---------- Eqs.(8)(9) small-angle form (paper printed form) ----------
    phi_los = wrap_pi(l_TM - l_DM)
    z = R_DM * phi_los                                      # Eq.(8)
    z_dot = R_DM_dot * phi_los + R_DM * (ld_TM - ld_DM)     # Eq.(9)

    # ---------- Eqs.(16)(35)(37) strictly per paper ----------
    cD = cos(th_D - l_DM)
    cT = cos(th_T - l_TM)
    cM = cos(th_A - l_TM)
    cM2 = cos(th_A - l_DM)
    b_D = cD * t_gDM                                       # Eq.(35)
    t2 = -(R_DM / R_TM) * cT                               # Eq.(16)
    t3 = -cT / R_TM
    m2 = (R_DM / R_TM) * cM - cM2
    m3 = cM / R_TM
    b_T = t2 * t_gDM + t3 * phi                            # Eq.(37)
    b_M = m2 * t_gDM + m3 * phi                            # Eq.(37) (diagnostic)

    # ---------- Zero-effort nominal numerical integration (paper does not specify method) ----------
    I_D, I_T, I_M = _ecdg_zero_effort_terms(geom, u_M, th_A, th_D, t_gDM)

    R_D_w = params.get('R_D_weight', 1.0)
    R_T_w = params.get('R_T_weight', 1.0)

    # ---------- Eqs.(47)(46) ----------
    Z = z + z_dot * t_gDM + phi * ld_TM + I_M              # Eq.(47)
    den = I_D / R_D_w + I_T / R_T_w
    if den < 1e-12:
        return 0.0, 0.0
    u_D = -(b_D / R_D_w) * Z / den                 # Eq.(46)
    u_T = -(b_T / R_T_w) * Z / den
    return u_D, u_T


def guidance_case4(params, geom, coeff, state, tgo, sigma, alpha, a_weight, m_weight, n_weight):
    """Case 4: LOS-OCG - Proposed cooperative guidance law based on LOS geometry and optimal control.

    This is the proposed guidance law that coordinates defender (u1) and target (u2)
    maneuvers to simultaneously minimize terminal miss distance (ZEM) and achieve
    desired intercept angles (ZEAE1, ZEAE2), while accounting for attacker acceleration.

    Key differences from Case 4 (Li et al.):
    - Three soft terminal constraints: ZEM, ZEAE1, ZEAE2
    - Cross-coupling between ZEM and ZEAE channels
    - Time-varying weights with tanh ramping
    """
    beta = params['beta']; Vt = geom['Vt']
    I3 = coeff['I3']; K = coeff['K']; I2 = coeff['I2']; a_A = coeff['a_A']
    phi = state['phi']; phi2 = state['phi2']; phi22 = state['phi22']
    ZEM = state['ZEM']; ZEAE1 = state['ZEAE1']; ZEAE2 = state['ZEAE2']
    # Denominator s (positive definite by construction)
    s = 12*beta**2*Vt**2 + 12*beta*n_weight*tgo + 12*beta*m_weight*I3**2*Vt**2*phi22 + 12*m_weight*n_weight*I3**2*(phi22*tgo - phi2**2)
    # Gains A and B for ZEAE feedback channels
    A = -12*I3*m_weight*(beta*Vt**2*phi + n_weight*(phi*tgo - phi2)) / s
    B = 12*n_weight*Vt*(beta + m_weight*I3**2*(phi22 - phi*phi2)) / s
    # Attacker acceleration feedforward coupling term
    C = I2 * a_A * phi2 * A
    # Defender command: ZEM + sigma optimal control
    u_D = - a_weight * K * tgo**2 * (2.0*sigma*tgo**2 + 4.0*ZEM) / (a_weight * K**2 * tgo**4 + 4.0*alpha)
    # Target command: ZEAE1 + ZEAE2 + attacker coupling
    u_T = A * ZEAE1 + B * ZEAE2 + C
    u_T = np.clip(u_T, -a_T_max, a_T_max)
    return u_D, u_T
# ===================== RK45 production kernel (CASE 1-4) =====================
# Switch: True  -> all four guidance laws are integrated by the adaptive
#                   Dormand-Prince 5(4) solver below
#         False -> legacy explicit Euler with fixed dt
USE_RK45_SOLVER = True
RK45_RTOL = 1e-9
RK45_ATOL = 1e-11


def run_single_simulation_rk45(CASE, Vd, save_traj=False, params_override=None,
                               rtol=None, atol=None):
    """Adaptive Dormand-Prince 5(4) kernel for the CASE 1-4 closed loop.

    Integrates the same model, guidance law, gravity compensation, saturation
    and first-order actuator lag as run_single_simulation(), but replaces the
    fixed-step Euler march by solve_ivp(method='RK45'). CASE 2 carries the
    lead-filter state as a 13th state variable and evaluates the LOS angular
    acceleration analytically, so no step-wise finite difference remains.
    The closest approach is located by a dR_ad/dt = 0 event inside the
    integrator; no spline and no external minimisation are used.
    """
    from scipy.integrate import solve_ivp

    rtol = RK45_RTOL if rtol is None else rtol
    atol = RK45_ATOL if atol is None else atol

    params = get_guidance_params(CASE, Vd)
    if params_override is not None:
        params.update(params_override)

    # Initial geometry (identical to run_single_simulation)
    dx0 = pos_A0[0] - pos_D0[0]
    dy0 = pos_A0[1] - pos_D0[1]
    if pos_A0[1] >= pos_D0[1]:
        theta_D0 = atan2(dy0, dx0) - 6 * deg2rad
    else:
        theta_D0 = atan2(dy0, dx0) + 6 * deg2rad
    theta_T0 = theta_D0
    theta_A0 = np.pi

    # ZEAE freeze state (mutable across RHS calls, as in the Euler loop)
    freeze = {'on': False, 'z1': 0.0, 'z2': 0.0}

    def deriv(t, y):
        x_T, y_T, th_T, a_T_sf = y[0], y[1], y[2], y[3]
        x_A, y_A, th_A, a_A_sf = y[4], y[5], y[6], y[7]
        x_D, y_D, th_D, a_D_sf = y[8], y[9], y[10], y[11]
        filt_x = y[12]
        th_T, th_A, th_D = wrap_pi(th_T), wrap_pi(th_A), wrap_pi(th_D)

        # --- Attacker-Target geometry ---
        dx_at, dy_at = x_A - x_T, y_A - y_T
        R_at = np.sqrt(dx_at**2 + dy_at**2)
        q_at = wrap_pi(atan2(dy_at, dx_at))
        vx_at = Va*cos(th_A) - Vt*cos(th_T)
        vy_at = Va*sin(th_A) - Vt*sin(th_T)
        R_at_dot = (dx_at*vx_at + dy_at*vy_at) / (R_at + eps)
        q_at_dot = (dx_at*vy_at - dy_at*vx_at) / (R_at**2 + eps)

        # --- Attacker-Defender geometry ---
        dx_ad, dy_ad = x_A - x_D, y_A - y_D
        R_ad = np.sqrt(dx_ad**2 + dy_ad**2)
        q_ad = wrap_pi(atan2(dy_ad, dx_ad))
        vx_ad = Va*cos(th_A) - Vd*cos(th_D)
        vy_ad = Va*sin(th_A) - Vd*sin(th_D)
        R_ad_dot = (dx_ad*vx_ad + dy_ad*vy_ad) / (R_ad + eps)
        q_ad_dot = (dx_ad*vy_ad - dy_ad*vx_ad) / (R_ad**2 + eps)

        # --- Target-Defender geometry ---
        dx_td, dy_td = x_D - x_T, y_D - y_T
        q_dt = wrap_pi(atan2(dy_td, dx_td))
        R_td = np.sqrt(dx_td**2 + dy_td**2)
        vx_td = Vd*cos(th_D) - Vt*cos(th_T)
        vy_td = Vd*sin(th_D) - Vt*sin(th_T)
        q_dt_dot = (dx_td*vy_td - dy_td*vx_td) / (R_td**2 + eps)
        R_td_dot = (dx_td*vx_td + dy_td*vy_td) / (R_td + eps)

        # Analytic LOS angular acceleration of the A-T line (used by CASE 2).
        # In the Euler kernel this was a backward difference of q_at_dot;
        # here it follows from the relative kinematics, so it is consistent
        # with the integrated state rather than with a step size.
        a_A_net = a_A_sf - g*cos(th_A)
        a_T_net = a_T_sf - g*cos(th_T)
        ax_rel = -a_A_net*sin(th_A) + a_T_net*sin(th_T)
        ay_rel = a_A_net*cos(th_A) - a_T_net*cos(th_T)
        q_at_ddot = (dx_at*ay_rel - dy_at*ax_rel) / (R_at**2 + eps) \
                    - 2.0*R_at_dot*q_at_dot / (R_at + eps)
        q_at_ddot = float(np.clip(q_at_ddot, -2.5, 2.5))

        # --- Time-to-go and ZEM state ---
        if R_ad_dot < -1e-3:
            tgo = -R_ad / R_ad_dot
        else:
            tgo = 0.01
        tgo = min(max(tgo, 0.001), TGO_MAX)

        delta_d = R_ad * sin(th_A - q_ad)
        theta_a_dot = (a_A_sf - g*cos(th_A)) / Va
        delta_d_dot = -Vd*sin(th_A - th_D) + R_ad*cos(th_A - q_ad)*theta_a_dot
        x1, x2, x3, x4, x5 = delta_d, delta_d_dot, q_at, q_at_dot, th_T

        # --- Guidance coefficients ---
        I1 = -2.0 * R_at_dot / R_at
        I2 = cos(th_A - q_at) / R_at
        I3 = -cos(th_T - q_at) / R_at
        K = cos(th_A - th_D)
        a_A = a_A_sf - g*cos(th_A)
        sigma = (1.0 - 2.0*Vd/Va*cos(th_A - th_D)) * a_A \
                - sin(th_A - q_ad) * R_ad * (a_A**2) / (Va**2)

        # --- phi integrals ---
        if abs(I1) < 1e-6:
            phi = tgo
            phi2 = tgo**2/2.0
            phi22 = tgo**3/3.0
        else:
            arg = I1 * tgo
            arg = min(max(arg, -350.0), 350.0)
            e1 = exp(arg)
            e2 = exp(2*arg)
            phi = (e1 - 1.0) / I1
            phi2 = (e1 - I1*tgo - 1.0) / (I1**2)
            phi22 = (e2 - 4.0*e1 + 2.0*I1*tgo + 3.0) / (2.0 * I1**3)
        # guard against extreme transient states visited by adaptive trial steps
        phi2 = float(np.clip(phi2, -1e6, 1e6))
        phi22 = float(np.clip(phi22, -1e8, 1e8))

        alpha = params.get('alpha', 1.0) * (20 ** min(max(tgo * 0.8, 1.0), 5))
        a_weight, m_weight, n_weight = compute_dynamic_weights(t, params, CASE)

        # --- ZEM / ZEAE ---
        ZEM = x1 + x2 * tgo
        ZEAE1_raw = x3 + x4 * phi - q_ad
        theta_s = params['theta_s'] if CASE == 4 else 0.0
        if pos_A0[1] >= pos_D0[1]:
            ZEAE2_raw = th_A - x5 + theta_s
        else:
            ZEAE2_raw = th_A - x5 + theta_s - np.pi
        ZEAE2_raw = wrap_pi(ZEAE2_raw)

        if R_ad < R_PROTECT:
            if not freeze['on']:
                freeze['on'] = True
                freeze['z1'] = ZEAE1_raw
                freeze['z2'] = ZEAE2_raw
            ZEAE1, ZEAE2 = freeze['z1'], freeze['z2']
        else:
            ZEAE1, ZEAE2 = ZEAE1_raw, ZEAE2_raw

        geom = {'Vd': Vd, 'Vt': Vt,
                'R_ad': R_ad, 'R_ad_dot': R_ad_dot, 'q_ad': q_ad, 'q_ad_dot': q_ad_dot,
                'R_at': R_at, 'R_at_dot': R_at_dot, 'q_at': q_at, 'q_at_dot': q_at_dot,
                'q_dt': q_dt, 'q_dt_dot': q_dt_dot,
                'R_td': R_td, 'R_td_dot': R_td_dot,
                'th_A': th_A, 'th_D': th_D, 'th_T': th_T,
                'err_los': q_at - q_dt, 'err_los_dot': q_at_dot - q_dt_dot,
                'q_at_ddot': q_at_ddot, 'filt_x': filt_x, 't': t,
                'x_A': x_A, 'y_A': y_A, 'x_D': x_D, 'y_D': y_D,
                'x_T': x_T, 'y_T': y_T}
        coeff = {'I1': I1, 'I2': I2, 'I3': I3, 'K': K, 'a_A': a_A}
        state = {'ZEM': ZEM, 'ZEAE1': ZEAE1, 'ZEAE2': ZEAE2,
                 'phi': phi, 'phi2': phi2, 'phi22': phi22}

        dfilt_x = 0.0
        if CASE == 1:
            u_D, u_T = guidance_case1(params, geom, coeff)
        elif CASE == 2:
            # LOSIG law inlined so that the filter output is the current state
            # (no +dt update inside the guidance call)
            K_clos = params['K_clos']
            a_lead = params['a_lead']
            b_lead = params['b_lead']
            dfilt_x = (-b_lead*filt_x + (b_lead - a_lead)*geom['err_los_dot']
                       + a_lead*geom['err_los'])
            ang_diff = th_T - q_at
            if (-np.pi/2 >= ang_diff) or (ang_diff >= np.pi/2):
                u_T = 2.0*R_td_dot*q_at_dot
            else:
                u_T = -2.0*R_td_dot*q_at_dot
            u_T = float(np.clip(u_T, -a_T_max, a_T_max))
            a_d_inertial = (K_clos*R_td*filt_x + u_T
                            + R_td*q_at_ddot + 2.0*R_td_dot*q_at_dot)
            u_D = a_d_inertial * cos(q_at - th_D)
        elif CASE == 3:
            u_D, u_T = guidance_case3(params, geom, coeff, state, tgo, sigma)
        elif CASE == 4:
            u_D, u_T = guidance_case4(params, geom, coeff, state, tgo, sigma,
                                      alpha, a_weight, m_weight, n_weight)
        else:
            u_D, u_T = 0.0, 0.0

        u_A = 3.0 * Va * q_at_dot

        # Gravity compensation, saturation, first-order lag
        u_D_cmd = np.clip(u_D + g*cos(th_D), -a_D_max, a_D_max)
        u_T_cmd = np.clip(u_T + g*cos(th_T), -a_T_max, a_T_max)
        u_A_cmd = np.clip(u_A + g*cos(th_A), -a_A_max, a_A_max)

        return [Vt*cos(th_T), Vt*sin(th_T), (a_T_sf - g*cos(th_T))/Vt,
                (u_T_cmd - a_T_sf)/tau_T,
                Va*cos(th_A), Va*sin(th_A), (a_A_sf - g*cos(th_A))/Va,
                (u_A_cmd - a_A_sf)/tau_A,
                Vd*cos(th_D), Vd*sin(th_D), (a_D_sf - g*cos(th_D))/Vd,
                (u_D_cmd - a_D_sf)/tau_D, dfilt_x]

    # --- Events: attacker reaches the target, or A-D closest approach ---
    def ev_target(t, y):
        return np.hypot(y[0]-y[4], y[1]-y[5]) - 1.0
    ev_target.terminal = True
    ev_target.direction = -1.0

    def ev_closest(t, y):
        dx = y[4] - y[8]
        dy = y[5] - y[9]
        R = np.hypot(dx, dy) + eps
        vx = Va*np.cos(y[6]) - Vd*np.cos(y[10])
        vy = Va*np.sin(y[6]) - Vd*np.sin(y[10])
        return (dx*vx + dy*vy) / R
    ev_closest.terminal = True
    ev_closest.direction = 1.0

    y0 = np.array([pos_T0[0], pos_T0[1], wrap_pi(theta_T0), 0.0,
                   pos_A0[0], pos_A0[1], wrap_pi(theta_A0), 0.0,
                   pos_D0[0], pos_D0[1], wrap_pi(theta_D0), 0.0, 0.0])

    sol = solve_ivp(deriv, (0.0, t_max), y0, method='RK45',
                    rtol=rtol, atol=atol,
                    events=(ev_target, ev_closest), dense_output=True)

    t_th = sol.t_events[0]
    t_cl = sol.t_events[1]

    if len(t_th) > 0 and (len(t_cl) == 0 or t_th[0] <= t_cl[0]):
        t_miss = float(t_th[0])
        termination_mode = 'target_hit'
    elif len(t_cl) > 0:
        t_miss = float(t_cl[0])
        termination_mode = None          # decided from the refined miss below
    else:
        t_miss = float(sol.t[-1])
        termination_mode = 'timeout'

    ym = sol.sol(t_miss)
    miss_final = float(np.hypot(ym[4] - ym[8], ym[5] - ym[9]))
    if termination_mode is None:
        termination_mode = 'intercept' if miss_final <= 1.0 else 'pass_cross'

    # --- Sampling for overload / effort statistics ---
    if save_traj:
        n_s = max(2, int(np.ceil(t_miss / dt)) + 1)
        ts = np.linspace(0.0, t_miss, n_s)
    else:
        ts = np.linspace(0.0, t_miss, 2000)
    S = sol.sol(ts)

    a_D_g = S[11] / g
    a_T_g = S[3] / g
    a_A_g = S[7] / g

    idx_peak_D = int(np.argmax(np.abs(a_D_g)))
    t_peak_D = float(ts[idx_peak_D])

    J_D = float(np.trapezoid(a_D_g**2, ts))
    J_T = float(np.trapezoid(a_T_g**2, ts))
    J_A = float(np.trapezoid(a_A_g**2, ts))

    xT_s, yT_s, thT_s = S[0], S[1], S[2]
    xA_s, yA_s, thA_s = S[4], S[5], S[6]
    thD_s = S[10]
    Rat_s = np.hypot(xA_s - xT_s, yA_s - yT_s)
    qat_s = np.arctan2(yA_s - yT_s, xA_s - xT_s)
    I3_s = -np.cos(thT_s - qat_s) / (Rat_s + eps)
    K_min = float(np.min(np.abs(np.cos(thA_s - thD_s))))
    RI3_min = float(np.min(Rat_s * np.abs(I3_s)))

    # Always provide the time/acceleration columns consumed by the plotting and
    # weight-sweep utilities; full state columns and ZEM/ZEAE only when asked.
    traj_data = {
        't': ts,
        'a_D_sf_g': a_D_g, 'a_T_sf_g': a_T_g, 'a_A_sf_g': a_A_g,
    }

    # ---- Unconditional heuristic block (consumed by TAD_Batch.py) ----------
    # The batch driver reads ZEM and six heur_* columns from res['traj'] even
    # when save_traj=False.  To keep the statistics comparable with the fixed-
    # step kernel, they are evaluated on the same 0.25 ms equispaced grid that
    # the Euler branch integrates on, instead of the coarse 2000-point grid
    # that is only meant for plotting.  Definitions mirror the Euler kernel
    # one-to-one (see "Heuristic check variables" in the fixed-step branch).
    n_h = max(2, int(np.ceil(t_miss / dt)) + 1)
    t_h = np.linspace(0.0, t_miss, n_h)
    S_h = sol.sol(t_h)

    xT_h, yT_h, thT_h = S_h[0], S_h[1], S_h[2]
    xA_h, yA_h, thA_h = S_h[4], S_h[5], S_h[6]
    a_A_sf_h = S_h[7]
    xD_h, yD_h, thD_h = S_h[8], S_h[9], S_h[10]

    dx_ad_h, dy_ad_h = xA_h - xD_h, yA_h - yD_h
    R_ad_h = np.hypot(dx_ad_h, dy_ad_h)
    q_ad_h = np.arctan2(dy_ad_h, dx_ad_h)
    vx_ad_h = Va*np.cos(thA_h) - Vd*np.cos(thD_h)
    vy_ad_h = Va*np.sin(thA_h) - Vd*np.sin(thD_h)
    R_ad_dot_h = (dx_ad_h*vx_ad_h + dy_ad_h*vy_ad_h) / (R_ad_h + eps)
    q_ad_dot_h = (dx_ad_h*vy_ad_h - dy_ad_h*vx_ad_h) / (R_ad_h**2 + eps)

    dx_at_h, dy_at_h = xA_h - xT_h, yA_h - yT_h
    Rat_h = np.hypot(dx_at_h, dy_at_h)
    qat_h = np.arctan2(dy_at_h, dx_at_h)
    vx_at_h = Va*np.cos(thA_h) - Vt*np.cos(thT_h)
    vy_at_h = Va*np.sin(thA_h) - Vt*np.sin(thT_h)
    R_at_dot_h = (dx_at_h*vx_at_h + dy_at_h*vy_at_h) / (Rat_h + eps)
    q_at_dot_h = (dx_at_h*vy_at_h - dy_at_h*vx_at_h) / (Rat_h**2 + eps)

    rdot_safe_h = np.where(np.abs(R_ad_dot_h) < 1e-12, -1.0, R_ad_dot_h)
    tgo_h = np.where(R_ad_dot_h < -1e-3, -R_ad_h / rdot_safe_h, 0.01)
    tgo_h = np.maximum(tgo_h, 0.001)

    delta_h = R_ad_h * np.sin(thA_h - q_ad_h)
    th_a_dot_h = (a_A_sf_h - g*np.cos(thA_h)) / Va
    delta_dot_h = (-Vd*np.sin(thA_h - thD_h)
                   + R_ad_h*np.cos(thA_h - q_ad_h)*th_a_dot_h)

    I1_h = -2.0 * R_at_dot_h / Rat_h
    I3_h = -np.cos(thT_h - qat_h) / (Rat_h + eps)
    K_h = np.cos(thA_h - thD_h)

    safe_I1_h = np.where(np.abs(I1_h) < 1e-6, 1.0, I1_h)
    e1_h = np.exp(np.clip(safe_I1_h*tgo_h, -350.0, 350.0))
    phi_h = np.where(np.abs(I1_h) < 1e-6, tgo_h, (e1_h - 1.0) / safe_I1_h)
    ZEM_h = delta_h + delta_dot_h*tgo_h

    # sigma: predicted attacker acceleration at intercept (same closed form
    # as in deriv()), independent of the commanded defender acceleration.
    a_A_h = a_A_sf_h - g*np.cos(thA_h)
    sigma_h = ((1.0 - 2.0*Vd/Va*np.cos(thA_h - thD_h)) * a_A_h
               - np.sin(thA_h - q_ad_h) * R_ad_h * a_A_h**2 / (Va**2))

    rdot_ok_h = np.abs(R_ad_dot_h) > 1e-6
    rdot2_h = np.where(rdot_ok_h, R_ad_dot_h**2, 1.0)
    Phi_c_h = K_h*tgo_h + np.where(
        rdot_ok_h,
        (delta_dot_h * R_ad_h / rdot2_h) * np.sin(thD_h - q_ad_h),
        0.0)

    zem_ok_h = np.abs(ZEM_h) > 1e-6
    zem_den_h = np.where(zem_ok_h, 2.0*np.abs(ZEM_h), 1.0)
    sigma_tgo2_h = np.where(
        zem_ok_h, np.abs(sigma_h) * tgo_h**2 / zem_den_h, 0.0)

    tgoB_exists_h = np.where(
        (I1_h > 1e-6) & (np.abs(I3_h) > 1e-6), 1.0, 0.0)

    # ZEAE columns are also read by the batch driver at the final node.
    # Freeze both once the defender enters the protection radius, exactly as
    # the fixed-step kernel does, so the LOS singularity cannot leak into the
    # reported values.
    ZEAE1_raw_h = qat_h + q_at_dot_h*phi_h - q_ad_h
    theta_s_off_h = params['theta_s'] if CASE == 4 else 0.0
    if pos_A0[1] >= pos_D0[1]:
        ZEAE2_raw_h = thA_h - thT_h + theta_s_off_h
    else:
        ZEAE2_raw_h = thA_h - thT_h + theta_s_off_h - np.pi
    ZEAE2_raw_h = (ZEAE2_raw_h + np.pi) % (2*np.pi) - np.pi

    prot_h = R_ad_h < R_PROTECT
    if np.any(prot_h):
        i0_h = int(np.argmax(prot_h))
        ZEAE1_h = np.where(prot_h, ZEAE1_raw_h[i0_h], ZEAE1_raw_h)
        ZEAE2_h = np.where(prot_h, ZEAE2_raw_h[i0_h], ZEAE2_raw_h)
        # ZEM is frozen at the same epoch as ZEAE so that all three terminal
        # indices are sampled at the protection-radius entry.
        ZEM_h = np.where(prot_h, ZEM_h[i0_h], ZEM_h)
    else:
        ZEAE1_h, ZEAE2_h = ZEAE1_raw_h, ZEAE2_raw_h

    traj_data.update({
        'heur_t': t_h,
        'ZEM': ZEM_h,
        'ZEAE1_deg': ZEAE1_h*rad2deg,
        'ZEAE2_deg': ZEAE2_h*rad2deg,
        'heur_K': K_h,
        'heur_RI3': Rat_h * I3_h,
        'heur_I1': I1_h,
        'heur_Phi_c': Phi_c_h,
        'heur_sigma_tgo2_over_2ZEM': sigma_tgo2_h,
        'heur_tgoB_exists': tgoB_exists_h,
    })

    if save_traj:
        xA_s, yA_s = S[4], S[5]
        xD_s, yD_s = S[8], S[9]
        dx_ad = xA_s - xD_s
        dy_ad = yA_s - yD_s
        R_ad_s = np.hypot(dx_ad, dy_ad)
        q_ad_s = np.arctan2(dy_ad, dx_ad)
        vx_ad = Va*np.cos(thA_s) - Vd*np.cos(thD_s)
        vy_ad = Va*np.sin(thA_s) - Vd*np.sin(thD_s)
        R_ad_dot_s = (dx_ad*vx_ad + dy_ad*vy_ad) / (R_ad_s + eps)
        q_ad_dot_s = (dx_ad*vy_ad - dy_ad*vx_ad) / (R_ad_s**2 + eps)
        Rat_s2 = np.hypot(xA_s - xT_s, yA_s - yT_s)
        qat_s2 = np.arctan2(yA_s - yT_s, xA_s - xT_s)
        vx_at = Va*np.cos(thA_s) - Vt*np.cos(thT_s)
        vy_at = Va*np.sin(thA_s) - Vt*np.sin(thT_s)
        R_at_dot_s = ((xA_s - xT_s)*vx_at + (yA_s - yT_s)*vy_at) / (Rat_s2 + eps)
        q_at_dot_s = ((xA_s - xT_s)*vy_at - (yA_s - yT_s)*vx_at) / (Rat_s2**2 + eps)

        tgo_s = np.where(R_ad_dot_s < -1e-3, -R_ad_s / R_ad_dot_s, 0.01)
        tgo_s = np.maximum(tgo_s, 0.001)
        delta_s = R_ad_s * np.sin(thA_s - q_ad_s)
        th_a_dot_s = (S[7] - g*np.cos(thA_s)) / Va
        delta_dot_s = (-Vd*np.sin(thA_s - thD_s)
                       + R_ad_s*np.cos(thA_s - q_ad_s)*th_a_dot_s)
        I1_s = -2.0 * R_at_dot_s / Rat_s2
        safe_I1 = np.where(np.abs(I1_s) < 1e-6, 1.0, I1_s)
        e1 = np.exp(np.clip(safe_I1*tgo_s, -350.0, 350.0))
        phi_s = np.where(np.abs(I1_s) < 1e-6, tgo_s, (e1 - 1.0) / safe_I1)
        ZEM_s = delta_s + delta_dot_s*tgo_s
        ZEAE1_s = qat_s2 + q_at_dot_s*phi_s - q_ad_s
        theta_s_off = params['theta_s'] if CASE == 4 else 0.0
        if pos_A0[1] >= pos_D0[1]:
            ZEAE2_s = S[6] - S[2] + theta_s_off
        else:
            ZEAE2_s = S[6] - S[2] + theta_s_off - np.pi
        ZEAE2_s = (ZEAE2_s + np.pi) % (2*np.pi) - np.pi

        traj_data.update({
            'x_T': S[0], 'y_T': S[1], 'x_A': S[4], 'y_A': S[5],
            'x_D': S[8], 'y_D': S[9],
            'th_T_deg': thT_s*rad2deg, 'th_A_deg': thA_s*rad2deg, 'th_D_deg': thD_s*rad2deg,
            'q_at_deg': qat_s2*rad2deg, 'q_ad_deg': q_ad_s*rad2deg,
            'miss': np.hypot(S[8]-S[4], S[9]-S[5]),
            # ZEAE must keep the protection-radius freeze already applied
            # above; the raw values diverge once q_ad flips at closest
            # approach, which is exactly what the freeze exists to prevent.
            'ZEM': ZEM_h,
            'ZEAE1_deg': ZEAE1_h*rad2deg,
            'ZEAE2_deg': ZEAE2_h*rad2deg,
        })

    return {
        'CASE': CASE, 'Vd': Vd,
        'miss_final_m': miss_final,
        'a_D_sf_peak_g': float(np.max(np.abs(a_D_g))),
        'a_T_sf_peak_g': float(np.max(np.abs(a_T_g))),
        'a_A_sf_peak_g': float(np.max(np.abs(a_A_g))),
        't_peak_D': t_peak_D,
        'J_D': J_D, 'J_T': J_T, 'J_A': J_A,
        'K_min': K_min, 'RI3_min': RI3_min,
        't_end': t_miss,
        'termination_mode': termination_mode,
        'solver': 'RK45', 'nfev': int(sol.nfev), 'nsteps': int(len(sol.t) - 1),
        'traj': traj_data,
    }


def run_single_simulation(CASE, Vd, save_traj=False, params_override=None):
    """Run a single engagement simulation for the specified guidance case and defender velocity.

    Simulates three bodies (target T, attacker A, defender D) in a 2D plane using
    first-order acceleration dynamics and the selected guidance law.

    Returns
    -------
    result : dict
        Contains miss distance, peak overloads, maneuver effort, and optional trajectory data.
    """
    if USE_RK45_SOLVER:
        return run_single_simulation_rk45(CASE, Vd, save_traj, params_override)

    # ---------- Helper functions for termination conditions ----------
    def is_intercept_hit(miss_DA, x_D, x_A):
        """Intercept occurs when miss < 1m.
        Note: x_D >= x_A removed per reviewer comment;
        closest approach is now detected via R_ad_dot sign change.
        """
        return miss_DA <= 1.0

    def is_target_hit(miss_AT):
        """Target is hit when attacker-target miss < 1m."""
        return miss_AT <= 1.0

    def extra_flight_exceeded(hit_triggered, t_hit, Vd, current_t, extra_dist=30.0):
        """After intercept trigger, continue flying for extra_dist meters to capture closest point."""
        if not hit_triggered:
            return False
        return Vd * (current_t - t_hit) > extra_dist
    # ---------------------------------------------------------------

    # Initial relative geometry
    dx0 = pos_A0[0] - pos_D0[0]
    dy0 = pos_A0[1] - pos_D0[1]

    # Initial defender flight path angle: lead the attacker by 6 degrees
    if pos_A0[1] >= pos_D0[1]:
        theta_D0 = atan2(dy0, dx0) - 6 * deg2rad
    else:
        theta_D0 = atan2(dy0, dx0) + 6 * deg2rad

    # Target shares initial heading with defender (co-launch)
    theta_T0 = theta_D0

    # Attacker initial heading: toward the origin (pointing left)
    theta_A0 = np.pi

    # Initialize state variables: position [x,y], flight path angle theta, acceleration a
    x_T, y_T, th_T, a_T_sf = pos_T0[0], pos_T0[1], theta_T0, 0.0
    x_A, y_A, th_A, a_A_sf = pos_A0[0], pos_A0[1], theta_A0, 0.0
    x_D, y_D, th_D, a_D_sf = pos_D0[0], pos_D0[1], theta_D0, 0.0

    # Wrap all initial angles to [-pi, pi]
    th_T = wrap_pi(th_T)
    th_A = wrap_pi(th_A)
    th_D = wrap_pi(th_D)

    # Previous-step state for finite differences
    a_A_sf_prev = 0.0
    th_A_prev = th_A
    q_at_dot_prev = 0.0
    filt_x = 0.0  # Lead filter state for Case 3
    R_m0 = np.sqrt((pos_A0[0]-pos_T0[0])**2 + (pos_A0[1]-pos_T0[1])**2)

    # ZEAE freezing logic: when defender enters protection radius, freeze ZEAE values
    # to prevent LOS rotation near intercept from corrupting the guidance command
    freeze_zeae = False
    zeae1_freeze_val = 0.0
    zeae2_freeze_val = 0.0
    zem_freeze_val = 0.0

    # Pre-allocate trajectory storage arrays
    N_step = int(t_max / dt) + 1
    t_array = np.linspace(0, t_max, N_step)
    save_T = np.zeros((N_step, 4))    # [x, y, theta_deg, a_g]
    save_A = np.zeros((N_step, 4))
    save_D = np.zeros((N_step, 4))
    save_geo = np.zeros((N_step, 6))  # [R_at, q_at_deg, R_ad, q_ad_deg, tgo, miss_DA]
    save_zem = np.zeros((N_step, 3))  # [ZEM, ZEAE1_deg, ZEAE2_deg]
    # Heuristic checks: [K, R_at*I3, I1, Phi_c, sigma_tgo2_over_2ZEM, sign_u1_sign_ZEM,
    #                    tgo_over_tgoB, tgoB_exists, dDdot_over_Rdot_ad, ZEMdot_g_over_ZEMdot_u1]
    save_heur = np.zeros((N_step, 10))

    params = get_guidance_params(CASE, Vd)
    if params_override is not None:
        params.update(params_override)

    # Intercept post-flight parameters
    extra_dist = 30.0
    hit_triggered = False
    t_hit = 0.0
    min_miss = np.inf
    min_idx = 0

    # Closest approach detection (d(R_ad^2)/dt sign change)
    R_ad_dot_prev = 0.0
    closest_approach_reached = False
    termination_mode = 'timeout'  # 'intercept', 'target_hit', 'pass_cross', 'timeout'

    # Gain-sign monitoring
    K_min = float('inf')
    RI3_min = float('inf')

    # ===================== Main simulation loop =====================
    for k in range(N_step):
        t = t_array[k]
        a_weight, m_weight, n_weight = compute_dynamic_weights(t, params, CASE)

        # --- Attacker-Target (A-T) geometry ---
        dx_at = x_A - x_T
        dy_at = y_A - y_T
        R_at = np.sqrt(dx_at**2 + dy_at**2)
        q_at = atan2(dy_at, dx_at)
        q_at = wrap_pi(q_at)
        vx_at = Va*cos(th_A) - Vt*cos(th_T)
        vy_at = Va*sin(th_A) - Vt*sin(th_T)
        R_at_dot = (dx_at*vx_at + dy_at*vy_at) / (R_at + eps)
        q_at_dot = (dx_at*vy_at - dy_at*vx_at) / (R_at**2 + eps)

        # --- Attacker-Defender (A-D) geometry ---
        dx_ad = x_A - x_D
        dy_ad = y_A - y_D
        R_ad = np.sqrt(dx_ad**2 + dy_ad**2)
        q_ad = atan2(dy_ad, dx_ad)
        q_ad = wrap_pi(q_ad)
        vx_ad = Va*cos(th_A) - Vd*cos(th_D)
        vy_ad = Va*sin(th_A) - Vd*sin(th_D)
        R_ad_dot = (dx_ad*vx_ad + dy_ad*vy_ad) / (R_ad + eps)
        q_ad_dot = (dx_ad*vy_ad - dy_ad*vx_ad) / (R_ad**2 + eps)

        # Detect closest approach: R_ad_dot changes from negative to non-negative
        if not closest_approach_reached and k > 0:
            if R_ad_dot_prev < -1e-6 and R_ad_dot >= 0:
                closest_approach_reached = True
        R_ad_dot_prev = R_ad_dot

        # --- Target-Defender (T-D) geometry ---
        dx_td = x_D - x_T
        dy_td = y_D - y_T
        q_dt = atan2(dy_td, dx_td)
        q_dt = wrap_pi(q_dt)
        R_td = np.sqrt(dx_td**2 + dy_td**2)
        vx_td = Vd*cos(th_D) - Vt*cos(th_T)
        vy_td = Vd*sin(th_D) - Vt*sin(th_T)
        q_dt_dot = (dx_td * vy_td - dy_td * vx_td) / (R_td**2 + eps)
        R_td_dot = (dx_td*vx_td + dy_td*vy_td) / (R_td + eps)

        # LOS angular acceleration (finite difference, clipped for stability)
        if k == 0:
            q_at_ddot = 0.0
        else:
            q_at_ddot = (q_at_dot - q_at_dot_prev)/dt
        q_at_ddot = np.clip(q_at_ddot, -2.5, 2.5)
        q_at_dot_prev = q_at_dot

        # Time-to-go estimate based on A-D closing rate
        if R_ad_dot < -1e-3:
            tgo = -R_ad / R_ad_dot
        else:
            tgo = 0.01
        tgo = min(max(tgo, 0.001), TGO_MAX)

        # --- ZEM state variables (delta_d and its derivative) ---
        delta_d = R_ad * sin(th_A - q_ad)
        theta_a_dot = (a_A_sf - g*cos(th_A)) / Va
        delta_d_dot = -Vd*sin(th_A - th_D) + R_ad*cos(th_A - q_ad)*theta_a_dot
        x1, x2, x3, x4, x5 = delta_d, delta_d_dot, q_at, q_at_dot, th_T

        # --- Guidance coefficients ---
        I1 = -2.0 * R_at_dot / R_at
        I2 = cos(th_A - q_at) / R_at
        I3 = -cos(th_T - q_at) / R_at
        K = cos(th_A - th_D)
        K_min = min(K_min, abs(K))
        RI3_min = min(RI3_min, R_at * abs(I3))
        # Net kinematic acceleration of attacker (for guidance law / sigma)
        a_A = a_A_sf - g*cos(th_A)
        a_A_sf_prev_available = a_A_sf_prev - g*cos(th_A_prev)
        dot_a_A = (a_A - a_A_sf_prev_available) / dt
        # sigma: predicted attacker acceleration at intercept (including jerk coupling)
        global USE_JERK
        # Only apply jerk term for Case 4; other cases use constant-acceleration assumption
        use_jerk_this_step = USE_JERK and (CASE == 4)
        if use_jerk_this_step:
            sigma = (1.0 - 2.0*Vd/Va*cos(th_A - th_D)) * a_A \
                    - sin(th_A - q_ad) * R_ad * (a_A**2) / (Va**2) \
                    + cos(th_A - q_ad) * R_ad / Va * dot_a_A
        else:
            sigma = (1.0 - 2.0*Vd/Va*cos(th_A - th_D)) * a_A \
                    - sin(th_A - q_ad) * R_ad * (a_A**2) / (Va**2)

        # --- phi integrals (transition matrix elements for ZEAE dynamics) ---
        if abs(I1) < 1e-6:
            # Limit case: I1 -> 0, phi integrals reduce to polynomial forms
            phi = tgo
            phi2 = tgo**2/2.0
            phi22 = tgo**3/3.0
        else:
            arg = I1 * tgo
            max_arg = 350.0  # Prevent exp overflow
            if arg > max_arg: arg = max_arg
            elif arg < -max_arg: arg = -max_arg
            e1 = exp(arg)
            e2 = exp(2*arg)
            phi = (e1 - 1.0) / I1
            phi2 = (e1 - I1*tgo - 1.0) / (I1**2)
            phi22 = (e2 - 4.0*e1 + 2.0*I1*tgo + 3.0) / (2.0 * I1**3)

        # alpha: regularization parameter for defender control (prevents excessive commands)
        alpha_base = params.get('alpha', 1.0)
        alpha = alpha_base * (20 ** min(max(tgo * 0.8, 1.0), 5))

        # --- Zero-Effort Miss (ZEM) and Zero-Effort Angle Error (ZEAE) ---
        ZEM = x1 + x2 * tgo
        ZEAE1_raw = x3 + x4 * phi - q_ad
        if CASE == 4:
            theta_s = params['theta_s']
        else:
            theta_s = 0.0

        # ZEAE2: intercept angle error accounting for attacker above/below defender
        if pos_A0[1] >= pos_D0[1]:
            ZEAE2_raw = th_A - x5 + theta_s
        else:
            ZEAE2_raw = th_A - x5 + theta_s - np.pi
        ZEAE2_raw = wrap_pi(ZEAE2_raw)

        # Freeze ZEAE when defender enters protection radius to avoid LOS singularity
        if R_ad < R_PROTECT:
            if not freeze_zeae:
                freeze_zeae = True
                zeae1_freeze_val = ZEAE1_raw
                zeae2_freeze_val = ZEAE2_raw
                zem_freeze_val = ZEM
            ZEAE1 = zeae1_freeze_val
            ZEAE2 = zeae2_freeze_val
            # Recorded ZEM is frozen at the same instant as ZEAE, so the two
            # terminal indices refer to one common epoch.  The live ZEM used by
            # the guidance law is left untouched.
            ZEM_saved = zem_freeze_val
        else:
            ZEAE1 = ZEAE1_raw
            ZEAE2 = ZEAE2_raw
            ZEM_saved = ZEM

        save_zem[k] = [ZEM_saved, ZEAE1*rad2deg, ZEAE2*rad2deg]

        # Package geometry, coefficients, and state for guidance functions
        geom = {'Vd':Vd, 'Vt':Vt,
                'R_ad':R_ad, 'R_ad_dot':R_ad_dot, 'q_ad':q_ad, 'q_ad_dot':q_ad_dot,
                'R_at':R_at, 'R_at_dot':R_at_dot, 'q_at':q_at, 'q_at_dot':q_at_dot,
                'q_dt':q_dt, 'q_dt_dot':q_dt_dot,
                'R_td':R_td, 'R_td_dot':R_td_dot,
                'th_A':th_A, 'th_D':th_D, 'th_T':th_T,
                'err_los': q_at - q_dt,
                'err_los_dot': q_at_dot - q_dt_dot,
                'q_at_ddot': q_at_ddot,
                'filt_x': filt_x,
                't': t,
                'x_A': x_A, 'y_A': y_A,
                'x_D': x_D, 'y_D': y_D,
                'x_T': x_T, 'y_T': y_T
                }
        coeff = {'I1':I1, 'I2':I2, 'I3':I3, 'K':K, 'a_A':a_A}
        state = {'ZEM':ZEM, 'ZEAE1':ZEAE1, 'ZEAE2':ZEAE2, 'phi':phi, 'phi2':phi2, 'phi22':phi22}

        # --- Compute guidance commands ---
        u_D = 0.0
        u_T = 0.0

        if CASE == 1:
            u_D, u_T = guidance_case1(params, geom, coeff)
        elif CASE == 2:
            u_D, u_T, filt_x = guidance_case2(params, geom, coeff, state, tgo, sigma, alpha)
        elif CASE == 3:
            u_D, u_T = guidance_case3(params, geom, coeff, state, tgo, sigma)
        elif CASE == 4:
            u_D, u_T = guidance_case4(params, geom, coeff, state, tgo, sigma, alpha, a_weight, m_weight, n_weight)

        # Attacker uses PN guidance against the target (N=3)
        u_A = 3.0 * Va * q_at_dot

        # --- Heuristic check variables (10 quantities) ---
        # 1. K = cos(th_A - th_D), expect K < 0
        # 2. R_at * I3 = -cos(th_T - q_at), expect < 0
        # 3. I1 = -2.0 * R_at_dot / R_at, expect I1 > 0
        # 4. Phi_c = K*tgo + (delta_d_dot * R_ad / R_ad_dot^2) * sin(th_D - q_ad), expect < 0
        # 5. |sigma| * tgo^2 / (2*|ZEM|), expect < 1
        # 6. sign(u1) * sign(ZEM), expect > 0 (u1 and ZEM same sign)
        # 7. tgo / tgo_B, expect < 1
        # 8. tgo_B exists (boolean, 1.0 if exists, 0.0 otherwise)
        # 9. |delta_d_dot| / |R_ad_dot|, expect << 1
        # 10. |ZEMdot_g| / |ZEMdot_u1|, expect << 1

        # Compute Phi_c
        Phi_c = K * tgo
        if abs(R_ad_dot) > 1e-6:
            Phi_c += (delta_d_dot * R_ad / (R_ad_dot**2)) * np.sin(th_D - q_ad)

        # Compute |sigma| * tgo^2 / (2*|ZEM|)
        if abs(ZEM) > 1e-6:
            sigma_tgo2_over_2ZEM = abs(sigma) * tgo**2 / (2.0 * abs(ZEM))
        else:
            sigma_tgo2_over_2ZEM = 0.0

        # sign(u1) * sign(ZEM) - u1 is defender control command
        sign_u1_sign_ZEM = np.sign(u_D) * np.sign(ZEM) if abs(ZEM) > 1e-6 and abs(u_D) > 1e-6 else 1.0

        # tgo_B: critical time where B changes sign (approximate, check existence)
        # tgo_B exists if I1 > 0 and I3 != 0
        tgoB_exists = 1.0 if (I1 > 1e-6 and abs(I3) > 1e-6) else 0.0
        tgo_over_tgoB = 1.0  # placeholder, will compute properly if needed

        # |delta_d_dot| / |R_ad_dot|
        dDdot_over_Rdot_ad = abs(delta_d_dot) / abs(R_ad_dot) if abs(R_ad_dot) > 1e-6 else 0.0

        # |ZEMdot_g| / |ZEMdot_u1|
        # ZEMdot_g = (delta_d_dot * R_ad^2 / R_ad_dot^2) * q_ad_dot^2
        # ZEMdot_u1 = u1 * Phi_c (control term)
        if abs(Phi_c) > 1e-6 and abs(u_D) > 1e-6:
            ZEMdot_g_mag = abs(delta_d_dot * R_ad**2 / (R_ad_dot**2)) * q_ad_dot**2 if abs(R_ad_dot) > 1e-6 else 0.0
            ZEMdot_u1_mag = abs(u_D * Phi_c)
            ZEMdot_g_over_ZEMdot_u1 = ZEMdot_g_mag / ZEMdot_u1_mag if ZEMdot_u1_mag > 1e-6 else 0.0
        else:
            ZEMdot_g_over_ZEMdot_u1 = 0.0

        save_heur[k] = [K, R_at * I3, I1, Phi_c, sigma_tgo2_over_2ZEM,
                        sign_u1_sign_ZEM, tgo_over_tgoB, tgoB_exists,
                        dDdot_over_Rdot_ad, ZEMdot_g_over_ZEMdot_u1]

        # Add gravity compensation and clip to acceleration limits
        u_D_cmd = u_D + g*cos(th_D)
        u_T_cmd = u_T + g*cos(th_T)
        u_A_cmd = u_A + g*cos(th_A)
        u_D_cmd = np.clip(u_D_cmd, -a_D_max, a_D_max)
        u_T_cmd = np.clip(u_T_cmd, -a_T_max, a_T_max)
        u_A_cmd = np.clip(u_A_cmd, -a_A_max, a_A_max)

        # First-order acceleration dynamics (lag)
        a_A_sf_prev = a_A_sf  # save old value before update
        th_A_prev = th_A
        a_D_sf += (u_D_cmd - a_D_sf) * dt / tau_D
        a_T_sf += (u_T_cmd - a_T_sf) * dt / tau_T
        a_A_sf += (u_A_cmd - a_A_sf) * dt / tau_A

        # Net kinematic accelerations (after actuator lag update)
        a_D = a_D_sf - g*cos(th_D)
        a_T = a_T_sf - g*cos(th_T)
        a_A = a_A_sf - g*cos(th_A)

        # Flight path angle kinematics: θ̇_i = a_i / V_i
        th_T += a_T / Vt * dt
        th_A += a_A / Va * dt
        th_D += a_D / Vd * dt

        th_T = wrap_pi(th_T)
        th_A = wrap_pi(th_A)
        th_D = wrap_pi(th_D)

        # Position integration
        x_T += Vt*cos(th_T)*dt
        y_T += Vt*sin(th_T)*dt
        x_A += Va*cos(th_A)*dt
        y_A += Va*sin(th_A)*dt
        x_D += Vd*cos(th_D)*dt
        y_D += Vd*sin(th_D)*dt

        # Save trajectory data
        save_T[k] = [x_T, y_T, th_T*rad2deg, a_T_sf/g]
        save_A[k] = [x_A, y_A, th_A*rad2deg, a_A_sf/g]
        save_D[k] = [x_D, y_D, th_D*rad2deg, a_D_sf/g]
        miss_DA = np.sqrt((x_D - x_A)**2 + (y_D - y_A)**2)
        miss_AT = np.sqrt((x_T - x_A)**2 + (y_T - y_A)**2)
        save_geo[k] = [R_at, q_at*rad2deg, R_ad, q_ad*rad2deg, tgo, miss_DA]

        # Track minimum miss distance for sub-step interpolation
        if miss_DA < min_miss:
            min_miss = miss_DA
            min_idx = k

        # Termination checks
        if t >= t_max:
            termination_mode = 'timeout'
            break
        if is_target_hit(miss_AT):
            termination_mode = 'target_hit'
            break
        # Trigger post-flight continuation when closest approach is reached
        if not hit_triggered and closest_approach_reached:
            hit_triggered = True
            t_hit = t
            # Classify termination mode at closest approach
            if miss_DA <= 1.0:
                termination_mode = 'intercept'
            else:
                termination_mode = 'pass_cross'
        if extra_flight_exceeded(hit_triggered, t_hit, Vd, t, extra_dist):
            break

    # ===================== Post-processing: high-precision miss distance =====================
    idx_end = min_idx
    t_eff = t_array[:idx_end+1]
    xA_eff = save_A[:idx_end+1, 0]
    yA_eff = save_A[:idx_end+1, 1]
    xD_eff = save_D[:idx_end+1, 0]
    yD_eff = save_D[:idx_end+1, 1]

    # Use cubic spline interpolation for sub-step miss distance precision
    miss_final = calc_interpolated_miss(t_eff, xA_eff, yA_eff, xD_eff, yD_eff, min_idx)

    # Re-classify termination mode based on final high-precision miss distance
    # This ensures consistency between success_count and n_intercept
    if termination_mode == 'intercept' or termination_mode == 'pass_cross':
        if miss_final <= 1.0:
            termination_mode = 'intercept'
        else:
            termination_mode = 'pass_cross'

    # Performance metrics (computed over trajectory up to closest point)
    a_D_sf_peak = np.max(np.abs(save_D[:idx_end+1, 3]))
    a_T_sf_peak = np.max(np.abs(save_T[:idx_end+1, 3]))
    a_A_sf_peak = np.max(np.abs(save_A[:idx_end+1, 3]))

    # Maneuver effort: integral of acceleration squared (J = ∫a² dt)
    J_D = np.trapezoid(save_D[:idx_end+1, 3]**2, t_eff)
    J_T = np.trapezoid(save_T[:idx_end+1, 3]**2, t_eff)
    J_A = np.trapezoid(save_A[:idx_end+1, 3]**2, t_eff)

    # Time of peak overload
    idx_peak_D = np.argmax(np.abs(save_D[:idx_end+1, 3]))
    t_peak_D = t_eff[idx_peak_D]
    idx_peak_T = np.argmax(np.abs(save_T[:idx_end+1, 3]))
    t_peak_T = t_eff[idx_peak_T]
    idx_peak_A = np.argmax(np.abs(save_A[:idx_end+1, 3]))
    t_peak_A = t_eff[idx_peak_A]

    # Package trajectory data if requested
    traj_data = {
        't': t_eff,
        'x_T': save_T[:idx_end+1,0], 'y_T': save_T[:idx_end+1,1],
        'x_A': save_A[:idx_end+1,0], 'y_A': save_A[:idx_end+1,1],
        'x_D': save_D[:idx_end+1,0], 'y_D': save_D[:idx_end+1,1],
        'a_D_sf_g': save_D[:idx_end+1,3], 'a_T_sf_g': save_T[:idx_end+1,3], 'a_A_sf_g': save_A[:idx_end+1,3],
        'q_at_deg': save_geo[:idx_end+1,1], 'q_ad_deg': save_geo[:idx_end+1,3],
        'miss': save_geo[:idx_end+1,5],
        'ZEM': save_zem[:idx_end+1,0],
        'ZEAE1_deg': save_zem[:idx_end+1,1],
        'ZEAE2_deg': save_zem[:idx_end+1,2],
        'th_T_deg': save_T[:idx_end+1,2],
        'th_A_deg': save_A[:idx_end+1,2],
        'th_D_deg': save_D[:idx_end+1,2],
        # Heuristic check variables (10 quantities)
        'heur_K': save_heur[:idx_end+1,0],
        'heur_RI3': save_heur[:idx_end+1,1],
        'heur_I1': save_heur[:idx_end+1,2],
        'heur_Phi_c': save_heur[:idx_end+1,3],
        'heur_sigma_tgo2_over_2ZEM': save_heur[:idx_end+1,4],
        'heur_sign_u1_sign_ZEM': save_heur[:idx_end+1,5],
        'heur_tgo_over_tgoB': save_heur[:idx_end+1,6],
        'heur_tgoB_exists': save_heur[:idx_end+1,7],
        'heur_dDdot_over_Rdot_ad': save_heur[:idx_end+1,8],
        'heur_ZEMdot_g_over_ZEMdot_u1': save_heur[:idx_end+1,9],
    }
    result = {
        'CASE': CASE, 'Vd': Vd,
        'miss_final_m': miss_final,
        'a_D_sf_peak_g': a_D_sf_peak,
        'a_T_sf_peak_g': a_T_sf_peak,
        'a_A_sf_peak_g': a_A_sf_peak,
        't_peak_D': t_peak_D,
        'J_D': J_D, 'J_T': J_T, 'J_A': J_A,
        'K_min': K_min, 'RI3_min': RI3_min,
        't_end': t_eff[-1],
        'termination_mode': termination_mode,
        'traj': traj_data
    }
    return result


# ====================== 4. Plotting and I/O functions ======================
def save_trajectory_csv(traj_data, case, vd, folder='Comp_data_figure'):
    """Save trajectory data to CSV file for post-processing or external plotting."""
    os.makedirs(folder, exist_ok=True)
    base_filename = f"trajectory_CASE{case}_Vd{int(vd)}"
    filename = f"{folder}/{base_filename}.csv"
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            keys = ['t','x_T','y_T','x_A','y_A','x_D','y_D','a_D_sf_g','a_T_sf_g','a_A_sf_g','q_at_deg','q_ad_deg','miss','ZEM','ZEAE1_deg','ZEAE2_deg']
            writer.writerow(keys)
            n = len(traj_data['t'])
            for i in range(n):
                row = [traj_data['t'][i], traj_data['x_T'][i], traj_data['y_T'][i],
                       traj_data['x_A'][i], traj_data['y_A'][i],
                       traj_data['x_D'][i], traj_data['y_D'][i],
                       traj_data['a_D_sf_g'][i], traj_data['a_T_sf_g'][i], traj_data['a_A_sf_g'][i],
                       traj_data['q_at_deg'][i], traj_data['q_ad_deg'][i],
                       traj_data['miss'][i],
                       traj_data['ZEM'][i],
                       traj_data['ZEAE1_deg'][i],
                       traj_data['ZEAE2_deg'][i]]
                writer.writerow(row)
    except PermissionError:
        # If file is locked (e.g., open in Excel), save with timestamp suffix
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        alt_filename = f"{folder}/{base_filename}_{timestamp}.csv"
        with open(alt_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(keys)
            for i in range(n):
                row = [traj_data['t'][i], traj_data['x_T'][i], traj_data['y_T'][i],
                       traj_data['x_A'][i], traj_data['y_A'][i],
                       traj_data['x_D'][i], traj_data['y_D'][i],
                       traj_data['a_D_sf_g'][i], traj_data['a_T_sf_g'][i], traj_data['a_A_sf_g'][i],
                       traj_data['q_at_deg'][i], traj_data['q_ad_deg'][i],
                       traj_data['miss'][i],
                       traj_data['ZEM'][i],
                       traj_data['ZEAE1_deg'][i],
                       traj_data['ZEAE2_deg'][i]]
                writer.writerow(row)


def plot_case_comparison(results, folder='Comp_data_figure'):
    """Generate comparison plots for all cases and velocities.

    Produces trajectory, overload, and (for Case 4) ZEM/ZEAE time histories.
    """
    plt.rcParams['lines.linewidth'] = 3
    plt.rcParams['axes.labelsize'] = 15
    plt.rcParams['xtick.labelsize'] = 15
    plt.rcParams['ytick.labelsize'] = 15
    plt.rcParams['legend.fontsize'] = 14.0
    case_dict = {}
    for res in results:
        c = res['CASE']
        if c not in case_dict:
            case_dict[c] = []
        case_dict[c].append(res)

    # Marker and label styles for each defender velocity
    style_cfg = {
        400: {'marker':'D','label':r'$V_d=400\,\mathrm{m/s}$'},
        500: {'marker':'o','label':r'$V_d=500\,\mathrm{m/s}$'},
        600: {'marker':'s','label':r'$V_d=600\,\mathrm{m/s}$'}
    }

    # Line styles: target=dotted, attacker=dash-dot, defender=solid
    linestyle_T = ':'
    linestyle_A = '-.'
    linestyle_D = '-'

    mark_step = 6000  # Marker interval in data points (unified: 1.5 s per marker)
    mk_fill = 'none'
    mk_edge = 'black'

    for case, res_list in case_dict.items():
        if not res_list:
            print(f"Warning: Case {case} has no simulation results, skipping plot")
            continue
        res_list.sort(key=lambda x: x['Vd'])
        print(f"Processing Case {case} with {len(res_list)} velocity points")

        # ---- Trajectory plot ----
        plt.figure(figsize=(10,5))
        for idx, res in enumerate(res_list):
            traj = res['traj']
            vd = res['Vd']
            cfg = style_cfg[int(vd)]
            lab_T = r'$T$' if idx == 0 else ""
            lab_A = r'$A$' if idx == 0 else ""
            plt.plot(traj['x_T'], traj['y_T'], linestyle=linestyle_T,
                     color='black', label=lab_T, linewidth=1.8)
            plt.plot(traj['x_A'], traj['y_A'], linestyle=linestyle_A,
                     color='black', label=lab_A, linewidth=1.8)
            plt.plot(traj['x_D'], traj['y_D'], linestyle=linestyle_D,
                     marker=cfg['marker'], markevery=mark_step,
                     color='black', markerfacecolor=mk_fill, markeredgecolor=mk_edge,
                     label=cfg['label'], linewidth=1.8)
        plt.xlabel('$x$/m')
        plt.ylabel('$y$/m')
        plt.grid(True, alpha=0.6)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=6,
                   fontsize=12, labelspacing=1.0, borderpad=0.6, handlelength=2.0)
        plt.title(f'Trajectory ({CASE_ABBR[case]})')
        plt.tight_layout()
        plt.savefig(f"{folder}/Figure_{CASE_ABBR[case]}_Trajectory.png", dpi=150, bbox_inches='tight')
        plt.close()

        # ---- Overload plot ----
        plt.figure(figsize=(10,5))
        for idx, res in enumerate(res_list):
            traj = res['traj']
            vd = res['Vd']
            cfg = style_cfg[int(vd)]
            a_g = traj['a_D_sf_g']
            t_arr = traj['t']
            lab_T = r'$T$' if idx == 0 else ""
            lab_A = r'$A$' if idx == 0 else ""
            plt.plot(traj['t'], traj['a_T_sf_g'], linestyle=linestyle_T,
                     color='black', label=lab_T, linewidth=1.4)
            plt.plot(traj['t'], traj['a_A_sf_g'], linestyle=linestyle_A,
                     color='black', label=lab_A, linewidth=1.4)
            # Mark peak overload with a star
            peak_idx = np.argmax(np.abs(a_g))
            t_peak = t_arr[peak_idx]
            a_peak = a_g[peak_idx]
            plt.plot(t_arr, a_g, linestyle=linestyle_D,
                     marker=cfg['marker'], markevery=mark_step,
                     color='black', markerfacecolor=mk_fill, markeredgecolor=mk_edge,
                     label=cfg['label'], linewidth=1.4)
            star_label = r'$a_y^{\mathrm{max}}$' if idx == 0 else ""
            plt.plot(t_peak, a_peak, marker='*', markersize=14,
                     color='black', label=star_label)
        plt.xlabel('$t$/s')
        plt.ylabel('$N_y/g$')
        plt.grid(True, alpha=0.6)
        ax = plt.gca()
        ymin, ymax = ax.get_ylim()
        y_range = ymax - ymin
        # Adaptive y-axis tick spacing
        if y_range <= 20:
            step = 2
        elif y_range <= 40:
            step = 5
        else:
            step = 10
        ax.set_yticks(np.arange(np.floor(ymin / step) * step,
                                np.ceil(ymax / step) * step + 0.1, step))

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=6,
                   fontsize=12, framealpha=0.9, edgecolor='black',
                   labelspacing=1.0, borderpad=0.6, handlelength=2.0)
        plt.title(f'Overload ({CASE_ABBR[case]})')
        plt.tight_layout()
        plt.savefig(f"{folder}/Figure_{CASE_ABBR[case]}_Overload.png", dpi=150, bbox_inches='tight')
        plt.close()

        # ===== Case 4 specific: ZEM / ZEAE1 / ZEAE2 time histories =====
        if case == 4:
            # ZEM plot
            plt.figure(figsize=(10,5))
            for res in res_list:
                traj = res['traj']
                vd = res['Vd']
                cfg = style_cfg[int(vd)]
                plt.plot(traj['t'], traj['ZEM'], linestyle=linestyle_D,
                         marker=cfg['marker'], markevery=mark_step,
                         color='black', markerfacecolor=mk_fill, markeredgecolor=mk_edge,
                         label=cfg['label'], linewidth=1.4)
            plt.xlabel('Time (s)')
            plt.ylabel('ZEM (m)')
            plt.grid(True, alpha=0.6)
            plt.legend(loc='best', fontsize=14.5, labelspacing=1.2, borderpad=0.8)
            plt.title(f'ZEM ({CASE_ABBR[4]})')
            plt.tight_layout()
            plt.savefig(f"{folder}/Figure_{CASE_ABBR[4]}_ZEM.png", dpi=150)
            plt.close()

            # ZEAE1 plot
            plt.figure(figsize=(10,5))
            for res in res_list:
                traj = res['traj']
                vd = res['Vd']
                cfg = style_cfg[int(vd)]
                plt.plot(traj['t'], traj['ZEAE1_deg'], linestyle=linestyle_D,
                         marker=cfg['marker'], markevery=mark_step,
                         color='black', markerfacecolor=mk_fill, markeredgecolor=mk_edge,
                         label=cfg['label'], linewidth=1.4)
            plt.xlabel('Time (s)')
            plt.ylabel('ZEAE1 (°)')
            plt.grid(True, alpha=0.6)
            plt.legend(loc='best', fontsize=14.5, labelspacing=1.2, borderpad=0.8)
            plt.title(f'ZEAE1 ({CASE_ABBR[4]})')
            plt.tight_layout()
            plt.savefig(f"{folder}/Figure_{CASE_ABBR[4]}_ZEAE1.png", dpi=150)
            plt.close()

            # ZEAE2 plot
            plt.figure(figsize=(10,5))
            for res in res_list:
                traj = res['traj']
                vd = res['Vd']
                cfg = style_cfg[int(vd)]
                plt.plot(traj['t'], traj['ZEAE2_deg'], linestyle=linestyle_D,
                         marker=cfg['marker'], markevery=mark_step,
                         color='black', markerfacecolor=mk_fill, markeredgecolor=mk_edge,
                         label=cfg['label'], linewidth=1.4)
            plt.xlabel('Time (s)')
            plt.ylabel('ZEAE2 (°)')
            plt.grid(True, alpha=0.6)
            plt.legend(loc='best', fontsize=14.5, labelspacing=1.2, borderpad=0.8)
            plt.title(f'ZEAE2 ({CASE_ABBR[4]})')
            plt.tight_layout()
            plt.savefig(f"{folder}/Figure_{CASE_ABBR[4]}_ZEAE2.png", dpi=150)
            plt.close()

        # ===== Case 2 specific: intercept angle (deleted - Head-On removed) =====


def plot_summary_figures(results, folder='Comp_data_figure'):
    """Generate summary figures: miss distance bar chart and terminal error boxplots."""
    categories = [f'C{r["CASE"]} V{int(r["Vd"])}' for r in results]
    miss_vals = [r['miss_final_m'] for r in results]
    plt.figure(figsize=(12,6))
    case_colors = {1:'black', 2:'dimgray', 3:'gray', 4:'lightgray', 5:'steelblue'}
    bar_colors = [case_colors[r['CASE']] for r in results]
    plt.bar(categories, miss_vals, color=bar_colors, edgecolor='black')
    plt.ylabel('Terminal miss distance (m)')
    plt.title('Miss distance comparison')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{folder}/Figure_Miss_Summary.png", dpi=150)
    plt.close()

    # Terminal error boxplots across all cases
    cases = sorted(set(r['CASE'] for r in results))
    zem_vals = [[r['traj']['ZEM'][-1] for r in results if r['CASE']==c] for c in cases]
    zeae1_vals = [[r['traj']['ZEAE1_deg'][-1] for r in results if r['CASE']==c] for c in cases]
    zeae2_vals = [[r['traj']['ZEAE2_deg'][-1] for r in results if r['CASE']==c] for c in cases]

    fig, axes = plt.subplots(1,3, figsize=(15,5))
    axes[0].boxplot(zem_vals, tick_labels=[f'C{c}' for c in cases], patch_artist=True,
                    boxprops=dict(facecolor='lightgray', color='black'),
                    whiskerprops=dict(color='black'), capprops=dict(color='black'),
                    medianprops=dict(color='black'))
    axes[0].set_ylabel('ZEM (m)'); axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[1].boxplot(zeae1_vals, tick_labels=[f'C{c}' for c in cases], patch_artist=True,
                    boxprops=dict(facecolor='lightgray', color='black'),
                    whiskerprops=dict(color='black'), capprops=dict(color='black'),
                    medianprops=dict(color='black'))
    axes[1].set_ylabel('ZEAE1 (°)'); axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[2].boxplot(zeae2_vals, tick_labels=[f'C{c}' for c in cases], patch_artist=True,
                    boxprops=dict(facecolor='lightgray', color='black'),
                    whiskerprops=dict(color='black'), capprops=dict(color='black'),
                    medianprops=dict(color='black'))
    axes[2].set_ylabel('ZEAE2 (deg)'); axes[2].grid(True, linestyle='--', alpha=0.5)
    plt.suptitle('Terminal errors distribution')
    plt.tight_layout()
    plt.savefig(f"{folder}/Figure_Terminal_Errors_Summary.png", dpi=150)
    plt.close()


def ecdg_weight_sweep(vd_list=[400, 500, 600],
                      ratios=[(1.0, 1.0), (0.1, 1.0), (0.25, 1.0), (0.5, 1.0),
                              (2.0, 1.0), (4.0, 1.0), (10.0, 1.0),
                              (1.0, 0.1), (1.0, 0.25), (1.0, 0.5),
                              (1.0, 2.0), (1.0, 4.0), (1.0, 10.0)]):
    """Sweep the C-ECDG (Case 3) R_D/R_T weights to support a fair Pareto comparison.

    Ref[17] (Li et al., 2022) uses the positive weight functions R_D(t), R_T(t) to
    trade defender and target effort. Case 3 previously fixed R_D=R_T=1. This sweep
    reruns Case 3 under several (R_D, R_T) ratios and reports miss distance, defender
    peak overload, and defender/target effort for each configuration, so the LOS-OCG
    comparison can be matched at common miss-distance or effort levels.
    """
    print("\n" + "=" * 100)
    print("R_D / R_T weight sweep for C-ECDG (Case 3) -- fair Pareto comparison")
    print("=" * 100)
    header = f"{'R_D':>6} {'R_T':>6} {'Vd(m/s)':>9} {'miss(m)':>9} {'aD_peak(g)':>10} {'J_D(s)':>8} {'J_T(s)':>8}"
    print(header)
    print("-" * len(header))
    rows = []
    for (rd, rt) in ratios:
        for Vd in vd_list:
            params_override = {'R_D_weight': rd, 'R_T_weight': rt}
            res = run_single_simulation(3, Vd, save_traj=False, params_override=params_override)
            rows.append((rd, rt, Vd, res['miss_final_m'], res['a_D_sf_peak_g'], res['J_D'], res['J_T']))
            print(f"{rd:>6} {rt:>6} {Vd:>9} {res['miss_final_m']:>9.4f} {res['a_D_sf_peak_g']:>10.3f} {res['J_D']:>8.2f} {res['J_T']:>8.2f}")
    print("=" * 100)
    return rows


def batch_run_and_plot(case_list=[1,2,3,4], vd_list=[400,500,600]):
    """Run all cases and velocities, generate plots, tables, and Excel output."""
    all_results = []
    for CASE in case_list:
        for Vd in vd_list:
            print(f"Running CASE={CASE}, Vd={Vd} m/s ...")
            # Case 3 (C-ECDG): use R_D/R_T = 10 for fair comparison
            if CASE == 3:
                params_override = {'R_D_weight': 10.0, 'R_T_weight': 1.0}
                res = run_single_simulation(CASE, Vd, save_traj=True, params_override=params_override)
            else:
                res = run_single_simulation(CASE, Vd, save_traj=True)
            all_results.append(res)
            save_trajectory_csv(res['traj'], CASE, Vd)
    plot_case_comparison(all_results)
    plot_summary_figures(all_results)
    print(f"All done. Results saved in 'Comp_data_figure' directory.")

    # ===== Jerk term comparison (Case 4, all Vd) - DISABLED =====
    # global USE_JERK
    # print("\n===== Jerk term sensitivity comparison (Case 4) =====")
    # for Vd_test in [400, 500, 600]:
    #     USE_JERK = True
    #     print(f"  -> Running WITH jerk (USE_JERK=True)")
    #     res_jerk = run_single_simulation(4, Vd_test, save_traj=True)
#         USE_JERK = False
#         print(f"  -> Running WITHOUT jerk (USE_JERK=False)")
#         res_nojerk = run_single_simulation(4, Vd_test, save_traj=True)
#         print(f"Vd={Vd_test}: With jerk miss={res_jerk['miss_final_m']:.4f}m DPeak={res_jerk['a_D_sf_peak_g']:.2f}g | "
#               f"No jerk miss={res_nojerk['miss_final_m']:.4f}m DPeak={res_nojerk['a_D_sf_peak_g']:.2f}g")
# 
#         traj_j = res_jerk['traj']
#         traj_n = res_nojerk['traj']
#         fig, axes = plt.subplots(2, 2, figsize=(12, 9))
# 
#         ax = axes[0,0]
#         ax.plot(traj_j['x_T'], traj_j['y_T'], 'k-', label='Target (with jerk)', linewidth=1.5)
#         ax.plot(traj_n['x_T'], traj_n['y_T'], 'k--', label='Target (no jerk)', linewidth=1.5)
#         ax.plot(traj_j['x_D'], traj_j['y_D'], 'k-.', label='Defender (with jerk)', linewidth=1.5)
#         ax.plot(traj_n['x_D'], traj_n['y_D'], 'k:', label='Defender (no jerk)', linewidth=1.5)
#         ax.plot(traj_j['x_A'], traj_j['y_A'], 'k-', label='Attacker (with jerk)', linewidth=1.5, alpha=0.5)
#         ax.plot(traj_n['x_A'], traj_n['y_A'], 'k--', label='Attacker (no jerk)', linewidth=1.5, alpha=0.5)
#         ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
#         ax.set_title(f'Trajectories: jerk ON vs OFF (Vd={Vd_test} m/s)'); ax.legend(fontsize=10.0); ax.grid(True, ls='--', alpha=0.5)
# 
#         ax = axes[0,1]
#         R_ad_j = np.sqrt((traj_j['x_A']-traj_j['x_D'])**2 + (traj_j['y_A']-traj_j['y_D'])**2)
#         R_ad_n = np.sqrt((traj_n['x_A']-traj_n['x_D'])**2 + (traj_n['y_A']-traj_n['y_D'])**2)
#         ax.plot(traj_j['t'], R_ad_j, 'k-', label='With jerk', linewidth=1.5)
#         ax.plot(traj_n['t'], R_ad_n, 'k--', label='No jerk', linewidth=1.5)
#         ax.set_xlabel('t (s)'); ax.set_ylabel('$R_{ad}$ (m)')
#         ax.set_title('Range $R_{ad}$'); ax.legend(); ax.grid(True, ls='--', alpha=0.5)
# 
#         ax = axes[1,0]
#         ax.plot(traj_j['t'], traj_j['a_D_sf_g'], 'k-', label='With jerk', linewidth=1.5)
#         ax.plot(traj_n['t'], traj_n['a_D_sf_g'], 'k--', label='No jerk', linewidth=1.5)
#         ax.set_xlabel('t (s)'); ax.set_ylabel('$N_D$ (g)')
#         ax.set_title('Defender overload'); ax.legend(); ax.grid(True, ls='--', alpha=0.5)
# 
#         ax = axes[1,1]; ax.axis('off')
#         info = (f"Vd = {Vd_test} m/s\n\n"
#                 f"With jerk:    miss = {res_jerk['miss_final_m']:.4f} m\n"
#                 f"               DPeak = {res_jerk['a_D_sf_peak_g']:.2f} g\n"
#                 f"Without jerk: miss = {res_nojerk['miss_final_m']:.4f} m\n"
#                 f"               DPeak = {res_nojerk['a_D_sf_peak_g']:.2f} g\n"
#                 f"Δ miss = {abs(res_jerk['miss_final_m']-res_nojerk['miss_final_m']):.4f} m\n"
#                 f"Δ DPeak = {abs(res_jerk['a_D_sf_peak_g']-res_nojerk['a_D_sf_peak_g']):.3f} g")
#         ax.text(0.1, 0.5, info, fontsize=11, family='monospace', va='center')
# 
#         plt.tight_layout()
#         plt.savefig(f'Comp_data_figure/Jerk_Comparison_Vd{Vd_test}.png', dpi=150)
#         plt.close()
#         print(f"  Saved: Comp_data_figure/Jerk_Comparison_Vd{Vd_test}.png")
# 
#     USE_JERK = True  # reset to default
# 
    # Print Table 1: Miss distance and peak overload comparison
    print("\n" + "=" * 90)
    print("Table 1. Comparison of miss distance and peak overload for five guidance laws")
    print("=" * 90)
    print("Case\tDefender Velocity (m/s)\tmiss distances(m)\ta_d^max(g)\tt_max(s)")
    for CASE in case_list:
        for i, Vd in enumerate(vd_list):
            res = [r for r in all_results if r['CASE'] == CASE and r['Vd'] == Vd][0]
            case_label = f"Case {CASE}" if i == 0 else ""
            print(f"{case_label}\tV_d={Vd}\t{res['miss_final_m']:.4f}\t{res['a_D_sf_peak_g']:.3f}\t{res['t_peak_D']:.3f}")

    # Print Table 2: Maneuver effort
    print("\n" + "=" * 100)
    print("Table 2. Maneuver effort of the target and defender.")
    print("=" * 100)
    print("Case\tDefender Velocity (m/s)\tTarget\tAttacker\tDefender\tJt+Jd (s)")
    for CASE in case_list:
        for i, Vd in enumerate(vd_list):
            res = [r for r in all_results if r['CASE'] == CASE and r['Vd'] == Vd][0]
            case_label = f"Case {CASE}" if i == 0 else ""
            jt_jd = res['J_T'] + res['J_D']
            print(f"{case_label}\tV_d={Vd}\t{res['J_T']:.2f}\t{res['J_A']:.2f}\t{res['J_D']:.2f}\t{jt_jd:.2f}")
    print("=" * 100)

    # Export results to Excel
    wb = Workbook()
    thin = Side(border_style="thin", color="000000")
    thick = Side(border_style="thick", color="000000")
    border_top_thick = Border(top=thick)
    border_bottom_thick = Border(bottom=thick)
    border_header = Border(top=thick, bottom=thin)
    center = Alignment(horizontal="center", vertical="center", wrap_text=True)
    bold = Font(bold=True)

    def fill_sheet(ws, title, headers, data_rows, case_col=0):
        """Helper to fill an Excel sheet with formatted table."""
        ws.append([title])
        ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(headers))
        ws.cell(row=1, column=1).font = bold
        ws.cell(row=1, column=1).alignment = center

        ws.append(headers)
        for j in range(1, len(headers) + 1):
            c = ws.cell(row=2, column=j)
            c.font = bold
            c.alignment = center
            c.border = border_header

        for row in data_rows:
            ws.append(row)

        n_data = len(data_rows)
        # Merge case label cells (3 rows per case)
        for case_idx in range(len(case_list)):
            start_r = 3 + case_idx * 3
            end_r = start_r + 2
            ws.merge_cells(start_row=start_r, start_column=case_col+1,
                           end_row=end_r, end_column=case_col+1)

        for r in range(3, 3 + n_data):
            for j in range(1, len(headers) + 1):
                c = ws.cell(row=r, column=j)
                c.alignment = center
                if isinstance(c.value, (int, float)):
                    c.number_format = '0.00'
                if r == 2 + n_data:
                    c.border = border_bottom_thick

        widths = [10, 22, 14, 14, 14, 18]
        for j, w in enumerate(widths[:len(headers)], 1):
            ws.column_dimensions[chr(64 + j)].width = w

    # Table 1 sheet
    ws1 = wb.active
    ws1.title = "Table1"
    headers1 = ["Case", "Defender Velocity (m/s)", "miss distances(m)",
                "a_d^max(g)", "t_max(s)"]
    data1 = []
    for CASE in case_list:
        for Vd in vd_list:
            res = [r for r in all_results if r['CASE'] == CASE and r['Vd'] == Vd][0]
            data1.append([f"Case {CASE}", f"V_d={Vd}",
                          round(res['miss_final_m'], 2),
                          round(res['a_D_sf_peak_g'], 2),
                          round(res['t_peak_D'], 2)])
    fill_sheet(ws1,
               "Table 1. Comparison of miss distance and peak overload for four guidance laws",
               headers1, data1)

    # Table 2 sheet
    ws2 = wb.create_sheet("Table2")
    headers2 = ["Case", "Defender Velocity (m/s)", "Target (s)", "Attacker (s)", "Defender (s)", "Jt+Jd (s)", "Engagement Time (s)"]
    data2 = []
    for CASE in case_list:
        for Vd in vd_list:
            res = [r for r in all_results if r['CASE'] == CASE and r['Vd'] == Vd][0]
            jt_jd = res['J_T'] + res['J_D']
            data2.append([f"Case {CASE}", f"V_d={Vd}",
                          round(res['J_T'], 2),
                          round(res['J_A'], 2),
                          round(res['J_D'], 2),
                          round(jt_jd, 2),
                          round(res['t_end'], 2)])
    fill_sheet(ws2,
               "Table 2. Maneuver effort of the target and defender.",
               headers2, data2)

    excel_path = os.path.join('Comp_data_figure', 'simulation_results.xlsx')
    wb.save(excel_path)
    print(f"\nExcel saved: {excel_path}")


def case3_weight_comparison(vd_list=[400, 500, 600], weight_ratios=[1.0, 2.0, 4.0, 6.0, 10.0]):
    """
    Case 3 (C-ECDG) weight comparison: run R_D/R_T = 1, 2, 4, 6, 10 five groups
    Output comparison curves and summary table
    """
    print("\n" + "=" * 100)
    print(f"Case 3 (C-ECDG) Weight Comparison: R_D/R_T = {weight_ratios}")
    print("=" * 100)

    # Store all results
    all_results = []
    for ratio in weight_ratios:
        for Vd in vd_list:
            print(f"  Running: R_D/R_T = {ratio}, Vd = {Vd} m/s ...")
            params_override = {'R_D_weight': ratio, 'R_T_weight': 1.0}
            res = run_single_simulation(3, Vd, save_traj=False, params_override=params_override)
            res['R_D/R_T'] = ratio
            all_results.append(res)

    # Print summary table
    print("\n" + "=" * 130)
    print("Summary Table: Miss Distance, Peak Overload, Maneuver Effort")
    print("=" * 130)
    header = f"{'R_D/R_T':>8} {'Vd(m/s)':>8} {'miss(m)':>10} {'aD_peak(g)':>10} {'aT_peak(g)':>10} {'t_D_peak(s)':>10} {'t_T_peak(s)':>10} {'J_D(s)':>10} {'J_T(s)':>10} {'J_D+J_T(s)':>12} {'t_end(s)':>10}"
    print(header)
    print("-" * 130)

    for res in all_results:
        jd_jt = res['J_D'] + res['J_T']
        # Find peak time
        traj = res['traj']
        idx_D_peak = np.argmax(np.abs(traj['a_D_sf_g']))
        idx_T_peak = np.argmax(np.abs(traj['a_T_sf_g']))
        t_D_peak = traj['t'][idx_D_peak]
        t_T_peak = traj['t'][idx_T_peak]
        print(f"{res['R_D/R_T']:8.1f} {res['Vd']:8.0f} {res['miss_final_m']:10.4f} {res['a_D_sf_peak_g']:10.3f} {res['a_T_sf_peak_g']:10.3f} {t_D_peak:10.3f} {t_T_peak:10.3f} {res['J_D']:10.2f} {res['J_T']:10.2f} {jd_jt:12.2f} {res['t_end']:10.3f}")

    print("=" * 130)

    # Save results to CSV
    csv_path = os.path.join('Comp_data_figure', 'case3_weight_comparison_results.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Header
        writer.writerow(['R_D/R_T', 'Vd(m/s)', 'miss(m)', 'aD_peak(g)', 'aT_peak(g)',
                         't_D_peak(s)', 't_T_peak(s)', 'J_D(s)', 'J_T(s)', 'J_D+J_T(s)', 't_end(s)'])
        # Data rows
        for res in all_results:
            jd_jt = res['J_D'] + res['J_T']
            traj = res['traj']
            idx_D_peak = np.argmax(np.abs(traj['a_D_sf_g']))
            idx_T_peak = np.argmax(np.abs(traj['a_T_sf_g']))
            t_D_peak = traj['t'][idx_D_peak]
            t_T_peak = traj['t'][idx_T_peak]
            writer.writerow([
                f"{res['R_D/R_T']:.1f}",
                f"{res['Vd']:.0f}",
                f"{res['miss_final_m']:.4f}",
                f"{res['a_D_sf_peak_g']:.3f}",
                f"{res['a_T_sf_peak_g']:.3f}",
                f"{t_D_peak:.3f}",
                f"{t_T_peak:.3f}",
                f"{res['J_D']:.2f}",
                f"{res['J_T']:.2f}",
                f"{jd_jt:.2f}",
                f"{res['t_end']:.3f}"
            ])
    print(f"\nResults saved to CSV: {csv_path}")

    # Black-and-white line styles and markers (for publication)
    line_styles = {1.0: '-', 2.0: '--', 4.0: '-.', 6.0: ':', 10.0: (0, (3, 1, 1, 1))}
    markers = {1.0: 'D', 2.0: 'o', 4.0: 's', 6.0: '^', 10.0: 'v'}
    labels = {1.0: r'$R_D/R_T=1$', 2.0: r'$R_D/R_T=2$', 4.0: r'$R_D/R_T=4$', 6.0: r'$R_D/R_T=6$', 10.0: r'$R_D/R_T=10$'}

    # Plot: overload comparison (black and white)
    from matplotlib.ticker import MultipleLocator
    fig, axes = plt.subplots(3, 3, figsize=(14, 11))

    for i, Vd in enumerate(vd_list):
        show_legend = (i == 0)  # only first column shows legend
        # Defender overload
        ax = axes[0, i]
        for ratio in weight_ratios:
            res = [r for r in all_results if r['R_D/R_T'] == ratio and r['Vd'] == Vd][0]
            traj = res['traj']
            ax.plot(traj['t'], traj['a_D_sf_g'], color='black', linestyle=line_styles[ratio],
                    marker=markers[ratio], markevery=6000, markersize=5,
                    markeredgewidth=1.2, markerfacecolor='none',
                    label=labels[ratio], linewidth=1.5)
        ax.set_xlabel('t (s)', fontsize=16)
        ax.set_ylabel(r'$N_{yd}$ (g)', fontsize=16)
        ax.set_title(f'Defender Overload (Vd={Vd} m/s)', fontsize=16)
        ax.tick_params(axis='both', labelsize=14)
        ax.yaxis.set_major_locator(MultipleLocator(2))
        if show_legend:
            ax.legend(fontsize=11.0, loc='upper left', labelspacing=0.8, borderpad=0.5)
        ax.grid(True, ls='--', alpha=0.5)

        # Target overload
        ax = axes[1, i]
        for ratio in weight_ratios:
            res = [r for r in all_results if r['R_D/R_T'] == ratio and r['Vd'] == Vd][0]
            traj = res['traj']
            ax.plot(traj['t'], traj['a_T_sf_g'], color='black', linestyle=line_styles[ratio],
                    marker=markers[ratio], markevery=6000, markersize=5,
                    markeredgewidth=1.2, markerfacecolor='none',
                    label=labels[ratio], linewidth=1.5)
        ax.set_xlabel('t (s)', fontsize=16)
        ax.set_ylabel(r'$N_{yt}$ (g)', fontsize=16)
        ax.set_title(f'Target Overload (Vd={Vd} m/s)', fontsize=16)
        ax.tick_params(axis='both', labelsize=14)
        ax.yaxis.set_major_locator(MultipleLocator(1))
        if show_legend:
            ax.legend(fontsize=11.0, loc='lower left', labelspacing=0.8, borderpad=0.5)
        ax.grid(True, ls='--', alpha=0.5)

        # Attacker overload
        ax = axes[2, i]
        for ratio in weight_ratios:
            res = [r for r in all_results if r['R_D/R_T'] == ratio and r['Vd'] == Vd][0]
            traj = res['traj']
            ax.plot(traj['t'], traj['a_A_sf_g'], color='black', linestyle=line_styles[ratio],
                    marker=markers[ratio], markevery=6000, markersize=5,
                    markeredgewidth=1.2, markerfacecolor='none',
                    label=labels[ratio], linewidth=1.5)
        ax.set_xlabel('t (s)', fontsize=16)
        ax.set_ylabel(r'$N_{ya}$ (g)', fontsize=16)
        ax.set_title(f'Attacker Overload (Vd={Vd} m/s)', fontsize=16)
        ax.tick_params(axis='both', labelsize=14)
        ax.yaxis.set_major_locator(MultipleLocator(2))
        if show_legend:
            ax.legend(fontsize=11.0, loc='lower right', labelspacing=0.8, borderpad=0.5)
        ax.grid(True, ls='--', alpha=0.5)

    plt.tight_layout()
    save_path = os.path.join('Comp_data_figure', 'case3_weight_comparison_overload.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nOverload comparison plot saved: {save_path}")

    # Plot: trajectory comparison (black and white style)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, Vd in enumerate(vd_list):
        ax = axes[i]
        # Plot defender trajectories for different weights (solid lines)
        for ratio in weight_ratios:
            res = [r for r in all_results if r['R_D/R_T'] == ratio and r['Vd'] == Vd][0]
            traj = res['traj']
            ax.plot(traj['x_D'], traj['y_D'], color='black', linestyle=line_styles[ratio],
                    label=f'Defender {labels[ratio]}', linewidth=2)

        # Plot attacker and target trajectories (dashed / dotted lines, lighter)
        attacker_styles = {1.0: '--', 2.0: ':', 4.0: '-.', 6.0: (0, (3, 1, 1, 1)), 10.0: (0, (5, 1))}
        target_styles = {1.0: (0, (1, 1)), 2.0: (0, (2, 1)), 4.0: (0, (3, 1)), 6.0: (0, (4, 1)), 10.0: (0, (5, 1))}
        for ratio in weight_ratios:
            res = [r for r in all_results if r['R_D/R_T'] == ratio and r['Vd'] == Vd][0]
            traj = res['traj']
            # Only label on the first one to avoid messy legend
            label_A = f'Attacker {labels[ratio]}' if ratio == 1.0 else None
            label_T = f'Target {labels[ratio]}' if ratio == 1.0 else None
            ax.plot(traj['x_A'], traj['y_A'], color='black', linestyle=attacker_styles[ratio],
                    alpha=0.6, linewidth=1.2, label=label_A)
            ax.plot(traj['x_T'], traj['y_T'], color='black', linestyle=target_styles[ratio],
                    alpha=0.4, linewidth=1.0, label=label_T)

        ax.set_xlabel('x (m)', fontsize=12)
        ax.set_ylabel('y (m)', fontsize=12)
        ax.set_title(f'Trajectories (Vd={Vd} m/s)', fontsize=13)
        ax.legend(fontsize=13.0, loc='best', labelspacing=1.2, borderpad=0.8)
        ax.grid(True, ls='--', alpha=0.5)
        ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    save_path2 = os.path.join('Comp_data_figure', 'case3_weight_comparison_trajectory.png')
    plt.savefig(save_path2, dpi=150, bbox_inches='tight')
    print(f"Trajectory comparison plot saved: {save_path2}")

    # Save all results to pickle for later re-plotting
    import pickle
    pickle_path = os.path.join('Comp_data_figure', 'case3_weight_comparison_results.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump({'all_results': all_results, 'vd_list': vd_list, 'weight_ratios': weight_ratios}, f)
    print(f"Results saved to pickle: {pickle_path}")

    return all_results


if __name__ == "__main__":
    batch_run_and_plot()
    # Optional: R_D/R_T sweep for the C-ECDG (Case 3) fair-Pareto comparison.
    # Enable with:  set C_ECDG_SWEEP=1  (or $env:C_ECDG_SWEEP='1')
    if os.environ.get('C_ECDG_SWEEP', '0') == '1':
        ecdg_weight_sweep()

    # Case 3 weight comparison: R_D/R_T = 1, 2, 4, 6, 10
    print("\n" + "=" * 100)
    print("Running Case 3 weight comparison (R_D/R_T = 1, 2, 4, 6, 10)...")
    print("=" * 100)
    case3_weight_comparison(vd_list=[400, 500, 600], weight_ratios=[1.0, 2.0, 4.0, 6.0, 10.0])

    # Case 4 weight comparison: scan m_max, n_max
    print("\n" + "=" * 100)
    print("Running Case 4 weight comparison (a_mult=1, m_mult=[0.5,1,2,4], n_mult=[0.5,1,2,4])...")
    print("=" * 100)