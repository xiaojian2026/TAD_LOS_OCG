# -*- coding: utf-8 -*-
"""Section 4.5 robustness Monte Carlo with the unified RK45 kernel (Round 3, M2).

Rate separation (response to Major Comment 2)
--------------------------------------------
The closed loop contains four independent clocks:

    integration      : Dormand-Prince 5(4) RK45, rtol=1e-9, atol=1e-11,
                       adaptive step, no fixed ceiling; restarted every
                       sensor interval.
    sensor sampling  : 50 Hz, period = 20 ms.  The complete measured state
                       vector is drawn once at the sample instant.
    tracker/filter   : 50 Hz.  The first-order dirty differentiator state z
                       is updated once per sample and held inside the interval.
    guidance command : 50 Hz.  The guidance law is recomputed once per sample
                       using the held measured values; no separate controller
                       clock inside the 20 ms interval.

No true kinematic quantity is refreshed between two 50 Hz samples.  Inside an
interval every measured channel, the true angular-rate feed-forward term, and
the filter state are all zero-order held.

Noise model (9 channels, zero-mean Gaussian, no truncation)
----------------------------------------------------------
    Va        sigma = 1.0 m/s      seed 1000
    theta_A   sigma = 0.1 deg      seed 1001
    R_at      sigma = 20 m         seed 1002
    q_at      sigma = 0.1 deg      seed 1003
    R_ad_dot  sigma = 1.0 m/s      seed 1004  (Doppler, LOS range-rate)
    a_T peak  3 g / 6 g random     seed 1005  (target overload upper bound)
    R_ad      sigma = 20 m         seed 1006
    q_ad      sigma = 0.1 deg      seed 1007
    R_at_dot  sigma = 1.0 m/s      seed 1008  (Doppler, LOS range-rate)

Seed 1004 is NOT attacker-maneuver randomization: the +2 g / -2 g attacker
evasion pulse remains fully deterministic.  Seed 1004 randomizes the
defender-seeker Doppler measurement error.

Error-injection architecture for a_A
-----------------------------------
a_A is not directly measurable.  Following Eq. (79), the real-time attacker
normal-acceleration estimate is reconstructed as

    a_A_meas = Va_meas * theta_a_dot_meas

where theta_a_dot_meas is built at each sample instant as

    theta_a_dot_meas = theta_a_dot_true + theta_a_dot_err

    theta_a_dot_true = a_A_true / Va        (true angular rate, held)
    theta_a_dot_err = (n_theta - z) / tau_track   (filtered angle error)
    z(k+1) = z(k) + alpha * (n_theta - z(k))    (first-order dirty diff.,
                                                   tau_track = 0.20 s)

This is an error-injection model, not a physical estimator: the true angular
rate is held at the sample instants, and the filtered angle error propagates
through to a_A.  The purpose is to quantify how the radar measurement noise
budget degrades the guidance loop, not to model a specific seeker.

The defender's own speed and flight-path angle and the target's speed and
flight-path angle are exact (self-contained inertial sensing).

Integration architecture
------------------------
The integration is split into sensor intervals of length ``sensor_dt``
(20 ms, i.e. 50 Hz).  At the beginning of each interval the complete measured
state vector is drawn once and held strictly constant inside that interval:

    attacker-relative : Va, theta_A, R_at, q_at, R_at_dot, q_at_dot
    defender seeker    : R_ad, q_ad, R_ad_dot, vd, theta_D
    target             : Vt, theta_T

Filter states are updated at the sample instant only and are held between
samples; the differentiator is not integrated continuously.  Inside an
interval the closed loop is smooth, so an adaptive Dormand-Prince 5(4) step
can be used without step-size collapse at the sample boundaries.

State vector (16 components):
    0-3    target:    x_T, y_T, theta_T, a_T_sf
    4-7    attacker:  x_A, y_A, theta_A, a_A_sf
    8-11   defender:  x_D, y_D, theta_D, a_D_sf
    12     inactive slot (filter states are updated per sample, not integrated)
    13-15  Va, Vt, Vd (velocity decay due to induced drag)

Closest approach is located by a dR_ad/dt = 0 event; no spline/Brent is used.

Output layer (inherited from the Euler reference implementation)
----------------------------------------------------------------
``run_mc_with_noise_rk45`` returns the same 14-item tuple as
``legacy/TAD_Robust_eluler_backup.py::run_mc_with_noise``, and reproduces the
same set of files under ``Robust_data_figure``:

    mc_raw_{dist}_{Vd}.npz            raw per-configuration arrays (15 fields)
    overload_history_{dist}_{Vd}.png overload history of the last run
    Error_Distributions.png           only for R0 = 6000 m, Vd = 400 m/s
    Acceleration_Error.png            only for R0 = 6000 m, Vd = 400 m/s
    mc_summary.csv                    19-column summary with [low, high] CIs
    Terminal_Scatter.png              18,000 terminal hit points
    Miss_Scatter.png                   18,000 miss distances, config-labeled
    Miss_CDF.png                       aggregate CDF
    Miss_Histogram.png                 aggregate histogram
"""

import os
import sys
import csv
import warnings
from math import sqrt, exp, pi, atan2, cos, sin

import numpy as np
from scipy.integrate import solve_ivp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---- Global figure typography (reviewers asked for larger annotation/legend text)
plt.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'legend.title_fontsize': 17,
    'figure.titlesize': 20,
})

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import TAD_Comp as tc
from TAD_Comp import (g, deg2rad, rad2deg, eps, t_max, wrap_pi,
                      get_guidance_params, compute_dynamic_weights,
                      guidance_case4)

# ---- Operational limits and numerical settings -----------------------------
A_D_MAX = 15.0 * g          # defender load cap used throughout Section 4.4
SENSOR_DT = 0.02            # 50 Hz measurement sampling and hold (20 ms)
STEP_MAX = 0.001            # integrator step ceiling (1 ms)
RTOL = 1e-9
ATOL = 1e-11
TGO_MAX = 20.0              # upper guard on t_go (covers the t_go^2 / t_go^4 terms)
R_PROTECT = 100.0           # ZEAE freeze range in the terminal phase

DEFAULT_FOLDER = 'Robust_data_figure'

_DEBUG = None               # optional diagnostics list


def g3(rng, std):
    """Zero-mean Gaussian sample, no truncation."""
    return float(rng.standard_normal() * std)


def _pulse(t, amplitude=None, t_start=1.0, duration=1.0, period=2.0, num=2):
    """Deterministic attacker evasion pulse: +2 g then -2 g, 1 s each."""
    a_amp = 2.0 * g if amplitude is None else amplitude
    for i in range(num):
        t0 = t_start + i * period
        if t0 < t < t0 + duration:
            return a_amp if i % 2 == 0 else -a_amp
    return 0.0


def _miss_ad(y):
    """True attacker-defender separation for a state vector."""
    return sqrt((y[4] - y[8]) ** 2 + (y[5] - y[9]) ** 2)


def _true_geometry(y):
    """Current true geometry, used for diagnostics.

    Returns (R_ad, R_ad_dot, R_at, R_at_dot).
    """
    x_T, y_T, th_T = y[0], y[1], wrap_pi(y[2])
    x_A, y_A, th_A = y[4], y[5], wrap_pi(y[6])
    x_D, y_D, th_D = y[8], y[9], wrap_pi(y[10])
    Va, Vt, Vd = y[13], y[14], y[15]

    dx_ad, dy_ad = x_A - x_D, y_A - y_D
    R_ad = sqrt(dx_ad ** 2 + dy_ad ** 2)
    vx_ad = Va * cos(th_A) - Vd * cos(th_D)
    vy_ad = Va * sin(th_A) - Vd * sin(th_D)
    R_ad_dot = (dx_ad * vx_ad + dy_ad * vy_ad) / (R_ad + eps)

    dx_at, dy_at = x_A - x_T, y_A - y_T
    R_at = sqrt(dx_at ** 2 + dy_at ** 2)
    vx_at = Va * cos(th_A) - Vt * cos(th_T)
    vy_at = Va * sin(th_A) - Vt * sin(th_T)
    R_at_dot = (dx_at * vx_at + dy_at * vy_at) / (R_at + eps)

    return R_ad, R_ad_dot, R_at, R_at_dot


def run_mc_with_noise_rk45(Vd=400.0, num_runs=2000,
                           std_pos=20.0, std_ang=0.1 * deg2rad, std_vel=1.0,
                           std_rdot=1.0,
                           tau_track=0.20, sensor_dt=SENSOR_DT, seed_base=1000,
                           seeker_noise=True, diag=None,
                           plot_last=False, plot_name='overload_history'):
    """Monte Carlo robustness study on the segmented RK45 kernel.

    Parameters
    ----------
    Vd : float
        Defender velocity (m/s).
    num_runs : int
        Number of Monte Carlo runs.
    std_pos, std_ang, std_vel, std_rdot : float
        Standard deviations of the range (m), angle (rad), speed (m/s) and
        range-rate (m/s) measurement errors; no truncation.
    tau_track : float
        Time constant of the tracking differentiator (s); the filter state is
        updated once per sampling interval and held inside it.
    sensor_dt : float
        Measurement sampling and hold period (s); 0.02 s = 50 Hz.
    seed_base : int
        Base seed; each noise source uses seed_base + offset.
    seeker_noise : bool
        Kept for backward comparison only.  The default True applies the full
        20 m / 0.1 deg noise set to R_ad and q_ad.
    diag : dict or None
        Optional accumulator for step-level diagnostics (steps, tgo floor,
        command saturation).
    plot_last : bool
        If True, record the state history of the last run and save the overload
        history figure as ``{plot_name}.png`` (same behaviour as the Euler
        reference implementation).
    plot_name : str
        File name (without extension) of the overload history figure.

    Returns
    -------
    tuple
        (miss_vals, miss_x, miss_y, a_D_peak, a_T_peak, a_A_peak,
         ZEM_end, ZEAE1_end, ZEAE2_end,
         err_Va, err_theta_a, err_R_at, err_q_at, err_a_a,
         termination_modes)
    """
    rng_Va = np.random.default_rng(seed_base)          # 1000 Va
    rng_theta = np.random.default_rng(seed_base + 1)   # 1001 theta_A
    rng_Rat = np.random.default_rng(seed_base + 2)     # 1002 R_at
    rng_qat = np.random.default_rng(seed_base + 3)     # 1003 q_at
    rng_Raddot = np.random.default_rng(seed_base + 4)  # 1004 R_ad_dot (Doppler)
    rng_target = np.random.default_rng(seed_base + 5)  # 1005 target ceiling
    rng_Rad = np.random.default_rng(seed_base + 6)     # 1006 R_ad
    rng_qad = np.random.default_rng(seed_base + 7)     # 1007 q_ad
    rng_Ratdot = np.random.default_rng(seed_base + 8)  # 1008 R_at_dot (Doppler)

    dx0 = tc.pos_A0[0] - tc.pos_T0[0]
    dy0 = tc.pos_A0[1] - tc.pos_T0[1]
    high_geometry = bool(tc.pos_A0[1] >= tc.pos_D0[1])
    if high_geometry:
        theta_D0 = atan2(dy0, dx0) - 6.0 * deg2rad
    else:
        theta_D0 = atan2(dy0, dx0) + 6.0 * deg2rad
    theta_T0 = theta_D0
    theta_A0 = pi

    Va0 = float(getattr(tc, 'Va', 600.0))
    Vt0 = float(getattr(tc, 'Vt', 270.0))
    Vd_init = float(Vd)

    params = get_guidance_params(4, Vd_init)
    theta_s = float(params.get('theta_s', pi / 2.0))
    a_A_max = 50.0 * g
    alpha0 = float(params.get('alpha', 1.0))

    miss_vals, miss_x, miss_y = [], [], []
    a_D_peak, a_T_peak, a_A_peak = [], [], []
    ZEM_end, ZEAE1_end, ZEAE2_end = [], [], []
    err_Va, err_theta_a, err_R_at, err_q_at, err_a_a = [], [], [], [], []
    termination_modes = []

    # history of the last run (only when plot_last is requested)
    hist_t, hist_a_A_true, hist_a_A_meas = [], [], []
    hist_xT, hist_yT, hist_xA, hist_yA, hist_xD, hist_yD = [], [], [], [], [], []
    hist_aD, hist_aT, hist_Va, hist_Vt, hist_Vd = [], [], [], [], []

    for run in range(num_runs):
        record = bool(plot_last and (run == num_runs - 1))
        if record:
            hist_t, hist_a_A_true, hist_a_A_meas = [], [], []
            hist_xT, hist_yT, hist_xA, hist_yA, hist_xD, hist_yD = [], [], [], [], [], []
            hist_aD, hist_aT, hist_Va, hist_Vt, hist_Vd = [], [], [], [], []

        a_T_max_sim = float(rng_target.choice([3.0, 6.0])) * g

        y = np.array([tc.pos_T0[0], tc.pos_T0[1], wrap_pi(theta_T0), 0.0,
                      tc.pos_A0[0], tc.pos_A0[1], wrap_pi(theta_A0), 0.0,
                      tc.pos_D0[0], tc.pos_D0[1], wrap_pi(theta_D0), 0.0,
                      0.0, Va0, Vt0, Vd_init])

        hold = {'Va': Va0, 'th_A': theta_A0, 'R_at': 1.0, 'q_at': 0.0,
                'R_at_dot': 0.0, 'q_at_dot': 0.0,
                'R_ad': 1.0, 'q_ad': 0.0, 'R_ad_dot': 0.0,
                'vd': Vd_init, 'th_D': theta_D0, 'vt': Vt0, 'th_T': theta_T0,
                'the_dot': 0.0, 'the_dot_true': 0.0,
                'noise_theta': 0.0, 'noise_Va': 0.0,
                'noise_R': 0.0, 'noise_q': 0.0}
        err_init = {'Va': 0.0, 'th': 0.0, 'R': 0.0, 'q': 0.0, 'a': 0.0}
        prev = {'q_at': None, 'qdot': 0.0, 'z': 0.0}
        freeze = {'on': False, 'z1': 0.0, 'z2': 0.0}
        last = {}

        def sample_hold(t, y, first):
            """Draw and freeze the complete measured state vector at time t."""
            x_T, y_T, th_T = y[0], y[1], wrap_pi(y[2])
            x_A, y_A, th_A = y[4], y[5], wrap_pi(y[6])
            va, vt = y[13], y[14]
            x_D, y_D, th_D = y[8], y[9], wrap_pi(y[10])
            vd = y[15]

            dx_at, dy_at = x_A - x_T, y_A - y_T
            R_at_true = sqrt(dx_at ** 2 + dy_at ** 2)
            q_at_true = wrap_pi(atan2(dy_at, dx_at))
            vx_at = va * cos(th_A) - vt * cos(th_T)
            vy_at = va * sin(th_A) - vt * sin(th_T)
            R_at_dot_true = (dx_at * vx_at + dy_at * vy_at) / (R_at_true + eps)

            dx_ad, dy_ad = x_A - x_D, y_A - y_D
            R_ad_true = sqrt(dx_ad ** 2 + dy_ad ** 2)
            q_ad_true = wrap_pi(atan2(dy_ad, dx_ad))
            vx_ad = va * cos(th_A) - vd * cos(th_D)
            vy_ad = va * sin(th_A) - vd * sin(th_D)
            R_ad_dot_true = (dx_ad * vx_ad + dy_ad * vy_ad) / (R_ad_true + eps)

            n_Va = g3(rng_Va, std_vel)
            n_th = g3(rng_theta, std_ang)
            n_R = g3(rng_Rat, std_pos)
            n_q = g3(rng_qat, std_ang)
            n_Rad = g3(rng_Rad, std_pos)
            n_qad = g3(rng_qad, std_ang)
            n_Raddot = g3(rng_Raddot, std_rdot)
            n_Ratdot = g3(rng_Ratdot, std_rdot)

            hold['Va'] = max(va + n_Va, 1.0)
            hold['th_A'] = wrap_pi(th_A + n_th)
            hold['R_at'] = max(R_at_true + n_R, 1.0)
            hold['q_at'] = wrap_pi(q_at_true + n_q)
            hold['R_at_dot'] = R_at_dot_true + n_Ratdot
            hold['R_ad'] = max(R_ad_true + n_Rad, 1.0) if seeker_noise else max(R_ad_true, 1.0)
            hold['q_ad'] = wrap_pi(q_ad_true + n_qad) if seeker_noise else wrap_pi(q_ad_true)
            hold['R_ad_dot'] = R_ad_dot_true + n_Raddot
            hold['vd'] = vd
            hold['th_D'] = th_D
            hold['vt'] = vt
            hold['th_T'] = th_T

            # ---- first-order tracking differentiators ---------------------
            # q_at rate: backward difference of consecutive 50 Hz samples plus
            # fixed first-order differentiator: tau_track = 0.20 s, 50 Hz update
            tau_now = tau_track
            alpha_now = sensor_dt / (sensor_dt + tau_now)

            # a discrete first-order low pass; the value is held in the interval.
            if prev['q_at'] is not None:
                qd_raw = wrap_pi(hold['q_at'] - prev['q_at']) / sensor_dt
                prev['qdot'] += alpha_now * (qd_raw - prev['qdot'])
            prev['q_at'] = hold['q_at']
            hold['q_at_dot'] = prev['qdot']

            # theta_A rate: the differentiator tracks the angle measurement
            # error n_th, which is the error-injection formulation of Eq. (80);
            # its state is updated once per sample and held inside the interval.
            hold['the_dot'] = (n_th - prev['z']) / tau_now
            prev['z'] += alpha_now * (n_th - prev['z'])

            a_a_true = y[7] - g * cos(th_A)
            # the true rate is sampled and held at the same instant as the
            # measurement, so no quantity is refreshed between 50 Hz samples
            hold['the_dot_true'] = a_a_true / max(va, eps)

            hold['noise_Va'] = n_Va
            hold['noise_theta'] = n_th
            hold['noise_R'] = n_R
            hold['noise_q'] = n_q

            if first:
                err_init['Va'] = n_Va
                err_init['th'] = n_th
                err_init['R'] = n_R
                err_init['q'] = n_q
                err_init['a'] = hold['Va'] * hold['the_dot'] + (hold['Va'] - va) * hold['the_dot_true']

        def deriv(t, y):
            x_T, y_T, th_T, a_T_sf = y[0], y[1], y[2], y[3]
            x_A, y_A, th_A, a_A_sf = y[4], y[5], y[6], y[7]
            x_D, y_D, th_D, a_D_sf = y[8], y[9], y[10], y[11]
            va, vt, vd = y[13], y[14], y[15]

            # ---- the guidance law reads only the held measurement set -----
            Va_m = hold['Va']
            th_A_m = hold['th_A']
            R_at_m = hold['R_at']
            q_at_m = hold['q_at']
            R_at_dot_m = hold['R_at_dot']
            q_at_dot_m = hold['q_at_dot']
            R_ad = hold['R_ad']
            q_ad = hold['q_ad']
            R_ad_dot = hold['R_ad_dot']
            vd_m = hold['vd']
            th_D_m = hold['th_D']
            vt_m = hold['vt']
            th_T_m = hold['th_T']

            # ---- time-to-go with upper guard ------------------------------
            if R_ad_dot < -1e-3:
                tgo = min(-R_ad / R_ad_dot, TGO_MAX)
            else:
                tgo = 0.01
            tgo = max(tgo, 0.001)

            # ---- reconstructed attacker rate -----------------------------
            theta_a_dot_meas = hold['the_dot_true'] + hold['the_dot']
            a_A_meas = Va_m * theta_a_dot_meas

            delta_d = R_ad * sin(th_A_m - q_ad)
            delta_d_dot = (-vd_m * sin(th_A_m - th_D_m)
                           + R_ad * cos(th_A_m - q_ad) * theta_a_dot_meas)
            x1, x2, x3, x4, x5 = delta_d, delta_d_dot, q_at_m, q_at_dot_m, th_T_m

            # ---- guidance coefficients ------------------------------------
            I1 = -2.0 * R_at_dot_m / R_at_m
            I2 = cos(th_A_m - q_at_m) / R_at_m
            I3 = -cos(th_T_m - q_at_m) / R_at_m
            K = cos(th_A_m - th_D_m)

            sigma = ((1.0 - 2.0 * vd_m / max(Va_m, eps) * cos(th_A_m - th_D_m)) * a_A_meas
                     - sin(th_A_m - q_ad) * R_ad * (a_A_meas ** 2) / (Va_m ** 2))

            # ---- transition-matrix integrals -------------------------------
            if abs(I1) < 1e-6:
                phi = tgo
                phi2 = tgo ** 2 / 2.0
                phi22 = tgo ** 3 / 3.0
            else:
                arg = min(max(I1 * tgo, -350.0), 350.0)
                e1 = exp(arg)
                e2 = exp(2 * arg)
                phi = (e1 - 1.0) / I1
                phi2 = (e1 - I1 * tgo - 1.0) / (I1 ** 2)
                phi22 = (e2 - 4.0 * e1 + 2.0 * I1 * tgo + 3.0) / (2.0 * I1 ** 3)
            phi2 = float(np.clip(phi2, -1e6, 1e6))
            phi22 = float(np.clip(phi22, -1e8, 1e8))

            alpha = alpha0 * (20 ** min(max(tgo * 0.8, 1.0), 5))
            a_weight, m_weight, n_weight = compute_dynamic_weights(t, params, 4)

            # ---- ZEM / ZEAE -----------------------------------------------
            ZEM = x1 + x2 * tgo
            ZEAE1_raw = x3 + x4 * phi - q_ad
            if high_geometry:
                ZEAE2_raw = th_A_m - x5 + theta_s
            else:
                ZEAE2_raw = th_A_m - x5 + theta_s - pi
            ZEAE2_raw = wrap_pi(ZEAE2_raw)

            if R_ad < R_PROTECT:
                if not freeze['on']:
                    freeze['on'] = True
                    freeze['z1'] = ZEAE1_raw
                    freeze['z2'] = ZEAE2_raw
                ZEAE1, ZEAE2 = freeze['z1'], freeze['z2']
            else:
                ZEAE1, ZEAE2 = ZEAE1_raw, ZEAE2_raw

            geom = {'Vd': vd_m, 'Vt': vt_m,
                    'R_ad': R_ad, 'R_ad_dot': R_ad_dot,
                    'q_ad': q_ad, 'q_ad_dot': 0.0,
                    'R_at': R_at_m, 'R_at_dot': R_at_dot_m,
                    'q_at': q_at_m, 'q_at_dot': q_at_dot_m,
                    'q_dt': 0.0, 'q_dt_dot': 0.0,
                    'R_td': 0.0, 'R_td_dot': 0.0,
                    'th_A': th_A_m, 'th_D': th_D_m, 'th_T': th_T_m,
                    'err_los': 0.0, 'err_los_dot': 0.0,
                    'q_at_ddot': 0.0, 'filt_x': 0.0, 't': t,
                    'x_A': x_A, 'y_A': y_A, 'x_D': x_D, 'y_D': y_D,
                    'x_T': x_T, 'y_T': y_T}
            coeff = {'I1': I1, 'I2': I2, 'I3': I3, 'K': K, 'a_A': a_A_meas}
            state = {'ZEM': ZEM, 'ZEAE1': ZEAE1, 'ZEAE2': ZEAE2,
                     'phi': phi, 'phi2': phi2, 'phi22': phi22}

            u_D, u_T = guidance_case4(params, geom, coeff, state, tgo,
                                      sigma, alpha, a_weight, m_weight, n_weight)

            u_A = 3.0 * va * q_at_dot_m + _pulse(t)

            u_D_cmd = float(np.clip(u_D + g * cos(th_D_m), -A_D_MAX, A_D_MAX))
            u_T_cmd = float(np.clip(u_T + g * cos(th_T_m), -a_T_max_sim, a_T_max_sim))
            u_A_cmd = float(np.clip(u_A + g * cos(th_A), -a_A_max, a_A_max))

            if diag is not None:
                diag['steps'] = diag.get('steps', 0) + 1
                if tgo <= 0.0011:
                    diag['tgo_floor'] = diag.get('tgo_floor', 0) + 1
                if abs(u_D_cmd) >= A_D_MAX - 1e-9:
                    diag['sat_D'] = diag.get('sat_D', 0) + 1
                if abs(u_T_cmd) >= a_T_max_sim - 1e-9:
                    diag['sat_T'] = diag.get('sat_T', 0) + 1

            last.update({'t': t, 'u1': u_D, 'u2': u_T, 'ZEM': ZEM, 'tgo': tgo,
                         'alpha': alpha, 'R_ad': R_ad, 'K': K,
                         'ZEAE1': ZEAE1, 'ZEAE2': ZEAE2})

            return [vt * cos(th_T), vt * sin(th_T), (a_T_sf - g * cos(th_T)) / vt,
                    (u_T_cmd - a_T_sf) / tc.tau_T,
                    va * cos(th_A), va * sin(th_A), (a_A_sf - g * cos(th_A)) / va,
                    (u_A_cmd - a_A_sf) / tc.tau_A,
                    vd * cos(th_D), vd * sin(th_D), (a_D_sf - g * cos(th_D)) / vd,
                    (u_D_cmd - a_D_sf) / tc.tau_D,
                    0.0,
                    -0.002 * abs(a_A_sf) / g * va,
                    -0.002 * abs(a_T_sf) / g * vt,
                    -0.002 * abs(a_D_sf) / g * vd]

        def ev_target(t, y):
            return sqrt((y[0] - y[4]) ** 2 + (y[1] - y[5]) ** 2) - 1.0
        ev_target.terminal = True
        ev_target.direction = -1.0

        def ev_closest(t, y):
            dx = y[4] - y[8]
            dy = y[5] - y[9]
            R = sqrt(dx ** 2 + dy ** 2) + eps
            vx = y[13] * cos(y[6]) - y[15] * cos(y[10])
            vy = y[13] * sin(y[6]) - y[15] * sin(y[10])
            return (dx * vx + dy * vy) / R
        ev_closest.terminal = True
        ev_closest.direction = 1.0

        # ---- segmented integration: sensor 50 Hz, guidance 50 Hz ----
        hist_aD, hist_aT, hist_aA = [], [], []
        t_cur = 0.0
        t_miss = None
        mode = None
        stopped = False

        while t_cur < t_max - 1e-12:
            sample_hold(t_cur, y, first=(t_cur == 0.0))
            t_next = min(t_cur + sensor_dt, t_max)
            sol = solve_ivp(deriv, (t_cur, t_next), y, method='RK45',
                            rtol=RTOL, atol=ATOL,
                            events=(ev_target, ev_closest))
            if sol.y.shape[1] == 0:
                t_cur = t_next
                continue

            hist_aD.append(np.abs(sol.y[11]) / g)
            hist_aT.append(np.abs(sol.y[3]) / g)
            hist_aA.append(np.abs(sol.y[7]) / g)

            if record:
                hist_t.append(sol.t)
                hist_xT.append(sol.y[0]); hist_yT.append(sol.y[1])
                hist_xA.append(sol.y[4]); hist_yA.append(sol.y[5])
                hist_xD.append(sol.y[8]); hist_yD.append(sol.y[9])
                a_true_seg = sol.y[7] - g * np.cos(sol.y[6])
                a_meas_scalar = hold['Va'] * hold['the_dot']
                a_meas_seg = np.full_like(sol.t, a_meas_scalar)
                hist_a_A_true.append(a_true_seg)
                hist_a_A_meas.append(a_meas_seg)
                hist_Va.append(sol.y[13]); hist_Vt.append(sol.y[14]); hist_Vd.append(sol.y[15])

            if sol.status == 1:
                cand = []
                if sol.t_events[0].size:
                    cand.append(('target_hit', float(sol.t_events[0][0])))
                if sol.t_events[1].size:
                    cand.append(('closest', float(sol.t_events[1][0])))
                cand.sort(key=lambda item: item[1])
                kind, t_miss = cand[0]
                mode = 'target_hit' if kind == 'target_hit' else None
                y = sol.y[:, -1]
                stopped = True
                break

            y = sol.y[:, -1]
            t_cur = t_next

        if not stopped:
            t_miss = t_cur
            mode = 'timeout'

        miss_final = float(sqrt((y[4] - y[8]) ** 2 + (y[5] - y[9]) ** 2))
        if mode is None:
            mode = 'intercept' if miss_final <= 1.0 else 'pass_cross'

        if hist_aD:
            aD_max = float(np.max(np.concatenate(hist_aD)))
            aT_max = float(np.max(np.concatenate(hist_aT)))
            aA_max = float(np.max(np.concatenate(hist_aA)))
        else:
            aD_max = aT_max = aA_max = 0.0

        miss_vals.append(miss_final)
        # Rotate miss vector into line-of-sight frame at intercept/pass point
        q_miss = atan2(y[5] - y[1], y[4] - y[0])
        _dx = float(y[8] - y[4])
        _dy = float(y[9] - y[5])
        miss_x.append(_dx * cos(q_miss) + _dy * sin(q_miss))       # along LOS (range)
        miss_y.append(-_dx * sin(q_miss) + _dy * cos(q_miss))      # cross LOS (cross-range)
        a_D_peak.append(aD_max)
        a_T_peak.append(aT_max)
        a_A_peak.append(aA_max)
        termination_modes.append(mode)
        ZEM_end.append(float(last.get('ZEM', 0.0)))
        ZEAE1_end.append(float(last.get('ZEAE1', 0.0)) * rad2deg)
        ZEAE2_end.append(float(last.get('ZEAE2', 0.0)) * rad2deg)
        err_Va.append(err_init['Va'])
        err_theta_a.append(err_init['th'])
        err_R_at.append(err_init['R'])
        err_q_at.append(err_init['q'])
        err_a_a.append(err_init['a'])

        if (run + 1) % 100 == 0:
            mv = np.array(miss_vals)
            dv = np.array(a_D_peak)
            print("  MC run %d/%d | miss: mean=%.3f m, max=%.3f m | DPeak: mean=%.2f g, max=%.2f g"
                  % (run + 1, num_runs, mv.mean(), mv.max(), dv.mean(), dv.max()), flush=True)

    print("  Valid runs: %d/%d" % (len(miss_vals), num_runs), flush=True)

    # ---- overload history of the last run (same output as the Euler kernel) --
    if plot_last and len(hist_t) > 0:
        hist_t_arr = np.concatenate(hist_t)
        a_true_arr = np.concatenate(hist_a_A_true)
        a_meas_arr = np.concatenate(hist_a_A_meas)
        aD_arr = np.concatenate([np.atleast_1d(np.asarray(v)) for v in hist_aD])
        aT_arr = np.concatenate([np.atleast_1d(np.asarray(v)) for v in hist_aT])

        plt.figure(figsize=(10, 4))
        plt.plot(hist_t_arr, a_true_arr / g, 'k-', linewidth=1.5)
        plt.xlabel('Time (s)')
        plt.ylabel('Acceleration (g)')
        plt.title('True Attacker Acceleration (PN + Bang-Bang Pulse)')
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(hist_t_arr, aD_arr, 'k-', linewidth=1.5, label='Defender (D)')
        plt.plot(hist_t_arr, aT_arr, 'k--', linewidth=1.5, label='Target (T)')
        plt.plot(hist_t_arr, a_true_arr / g, 'k:', linewidth=1.5, label='Attacker (A)')
        plt.xlabel('Time (s)')
        plt.ylabel('Acceleration (g)')
        plt.title('Overload History')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f'{plot_name}.png', dpi=150)
        plt.close()
        print(f"  Saved {plot_name}.png")

    return (miss_vals, miss_x, miss_y,
            a_D_peak, a_T_peak, a_A_peak,
            ZEM_end, ZEAE1_end, ZEAE2_end,
            err_Va, err_theta_a, err_R_at, err_q_at, err_a_a,
            termination_modes)


# ---------------------------------------------------------------- report ----
def summarize(miss, peak):
    """Mean / median / 95th percentile with intervals and Wilson success band."""
    from scipy import stats as st

    miss = np.asarray(miss, dtype=float)
    peak = np.asarray(peak, dtype=float)
    n = miss.size
    mean = float(np.mean(miss))
    se = float(np.std(miss, ddof=1) / sqrt(n))
    tcrit = float(st.t.ppf(0.975, n - 1))
    p95 = float(np.percentile(miss, 95.0))
    rng = np.random.default_rng(2026)
    boot = np.array([np.percentile(rng.choice(miss, n, replace=True), 95.0)
                     for _ in range(1000)])
    k = int(np.count_nonzero(miss <= 1.0))
    p = k / n
    z = 1.959963984540054
    den = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / den
    half = z * sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / den
    return {'n': n, 'mean': mean, 'mean_ci': (mean - tcrit * se, mean + tcrit * se),
            'median': float(np.median(miss)), 'p95': p95,
            'p95_ci': (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))),
            'max': float(np.max(miss)), 'k': k, 'p': p,
            'success_ci': (max(0.0, center - half), min(1.0, center + half)),
            'peak_mean': float(np.mean(peak)),
            'peak_p95': float(np.percentile(peak, 95.0)),
            'peak_max': float(np.max(peak))}


# ===================== Plotting functions =====================
def plot_error_distributions(err_Va, err_theta_a, err_R_at, err_q_at,
                             folder=DEFAULT_FOLDER):
    """Initial measurement error series across Monte Carlo samples (Fig. 19/20)."""
    os.makedirs(folder, exist_ok=True)
    n = len(err_Va)
    x = np.arange(1, n + 1)

    fig, axes = plt.subplots(4, 1, figsize=(10, 12))

    axes[0].plot(x, err_Va, 'k-', linewidth=0.6)
    axes[0].set_ylabel(r'$V_a^e$ (m/s)', fontsize=16)
    axes[0].tick_params(axis='both', labelsize=13)
    axes[0].margins(y=0.05)
    axes[0].grid(True, linestyle='--', alpha=0.5)

    axes[1].plot(x, np.rad2deg(err_theta_a), 'k-', linewidth=0.6)
    axes[1].set_ylabel(r'$\theta_a^e$ ($^\circ$)', fontsize=16)
    axes[1].tick_params(axis='both', labelsize=13)
    axes[1].margins(y=0.05)
    axes[1].grid(True, linestyle='--', alpha=0.5)

    axes[2].plot(x, err_R_at, 'k-', linewidth=0.6)
    axes[2].set_ylabel(r'$R_{at}^e$ (m)', fontsize=16)
    axes[2].tick_params(axis='both', labelsize=13)
    axes[2].margins(y=0.05)
    axes[2].grid(True, linestyle='--', alpha=0.5)

    axes[3].plot(x, np.rad2deg(err_q_at), 'k-', linewidth=0.6)
    axes[3].set_ylabel(r'$q_{at}^e$ ($^\circ$)', fontsize=16)
    axes[3].set_xlabel('Number of trajectories', fontsize=16)
    axes[3].tick_params(axis='both', labelsize=13)
    axes[3].margins(y=0.05)
    axes[3].grid(True, linestyle='--', alpha=0.5)

    plt.suptitle('Initial measurement errors across Monte Carlo samples', fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(f'{folder}/Error_Distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved Error_Distributions")


def plot_acceleration_error(err_a_a, folder=DEFAULT_FOLDER):
    """Attacker acceleration estimation error across samples (Fig. 18)."""
    os.makedirs(folder, exist_ok=True)
    n = len(err_a_a)
    x = np.arange(1, n + 1)
    plt.figure(figsize=(12, 7))
    plt.plot(x, err_a_a, 'k-', linewidth=0.6)
    plt.xlabel('Number of trajectories', fontsize=14)
    plt.ylabel(r'$a_a^e$ (m/s$^2$)', fontsize=14)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.title('Acceleration error across Monte Carlo samples', fontsize=16, pad=15)
    plt.tight_layout()
    plt.savefig(f'{folder}/Acceleration_Error.png', dpi=300, bbox_inches='tight')
    plt.close()
    max_abs = np.max(np.abs(err_a_a)) if len(err_a_a) > 0 else 0
    print(f"Saved Acceleration_Error (max |a_a^e| = {max_abs:.2f} m/s^2)")


def plot_terminal_scatter(miss_x, miss_y, folder=DEFAULT_FOLDER):
    """Terminal hit point scatter in the relative position plane."""
    os.makedirs(folder, exist_ok=True)
    plt.figure(figsize=(7, 6))
    plt.plot(miss_x, miss_y, 'k.', markersize=4)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.title('Terminal hit accuracy')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f'{folder}/Terminal_Scatter.png', dpi=150)
    plt.close()
    print("Saved Terminal_Scatter")


def plot_miss_scatter(miss_vals, folder=DEFAULT_FOLDER):
    """Miss distance against trajectory index (Figure 22)."""
    os.makedirs(folder, exist_ok=True)
    miss_clean = [m for m in miss_vals if not np.isnan(m)]
    if len(miss_clean) == 0:
        print("Warning: miss_vals all NaN, skip scatter")
        return
    n = len(miss_clean)
    x = np.arange(1, n + 1)
    plt.figure(figsize=(14, 7))
    plt.plot(x, miss_clean, 'k.', markersize=3)
    plt.xlabel('Number of trajectories', fontsize=14)
    plt.ylabel('Miss distance (m)', fontsize=14)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlim(0, n)
    plt.ylim(bottom=0)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.title('Monte Carlo miss distance statistics', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(f'{folder}/Miss_Scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved Miss_Scatter (Figure 22)")


def plot_cdf_miss(miss_vals, folder=DEFAULT_FOLDER):
    """Cumulative distribution function of the miss distance."""
    os.makedirs(folder, exist_ok=True)
    miss_clean = [m for m in miss_vals if not np.isnan(m)]
    if len(miss_clean) == 0:
        print("Warning: miss_vals all NaN, skip CDF")
        return
    sorted_miss = np.sort(miss_clean)
    cdf = np.arange(1, len(sorted_miss) + 1) / len(sorted_miss)
    plt.figure(figsize=(8, 6))
    plt.plot(sorted_miss, cdf, 'k-', linewidth=1.5)
    plt.xlabel('Miss distance (m)', fontsize=14)
    plt.ylabel('Cumulative probability', fontsize=14)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.title('Monte Carlo terminal hit accuracy (CDF)', fontsize=16, pad=15)
    plt.tight_layout()
    plt.savefig(f'{folder}/Miss_CDF.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved Miss_CDF")


def plot_miss_histogram(miss_vals, folder=DEFAULT_FOLDER):
    """Histogram of the miss distance distribution."""
    os.makedirs(folder, exist_ok=True)
    miss_clean = [m for m in miss_vals if not np.isnan(m)]
    if len(miss_clean) == 0:
        print("Warning: miss_vals all NaN, skip histogram")
        return
    plt.figure(figsize=(8, 6))
    plt.hist(miss_clean, bins=50, edgecolor='black', color='lightgray')
    plt.xlabel('Miss distance (m)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.title('Monte Carlo miss distance statistics', fontsize=16, pad=15)
    plt.tight_layout()
    plt.savefig(f'{folder}/Miss_Histogram.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved Miss_Histogram")


def plot_ecdf_grouped(results, folder=DEFAULT_FOLDER, thr=1.0):
    """Grouped ECDF of the terminal miss distance, one curve per configuration.

    results: list of dicts, each carrying 'dist', 'Vd' and '_miss_vals'.
    The vertical reading at x = thr gives the success rate of that configuration.
    """
    os.makedirs(folder, exist_ok=True)
    if not results:
        print("Warning: no results, skip grouped ECDF")
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    line_styles = {400: ':', 500: '--', 600: '-'}
    markers = {6000: 'o', 8000: 's', 10000: '^'}

    x_max = 0.0
    for r in sorted(results, key=lambda q: (int(q['dist']), int(q['Vd']))):
        mv = np.asarray(r['_miss_vals'], dtype=float)
        mv = np.sort(mv[~np.isnan(mv)])
        if mv.size == 0:
            continue
        y = np.arange(1, mv.size + 1) / float(mv.size)
        ax.step(mv, y, where='post', color='k',
                linestyle=line_styles.get(int(r['Vd']), '-'),
                linewidth=1.6,
                marker=markers.get(int(r['dist']), None),
                markevery=max(1, int(mv.size / 8)), markersize=6,
                markerfacecolor='white', markeredgecolor='k',
                label=r"$R_0$ = %d m, $V_d$ = %d m/s" % (int(r['dist']), int(r['Vd'])))
        x_max = max(x_max, float(mv[-1]))

    ax.axvline(thr, color='k', linestyle='--', linewidth=1.3)
    ax.text(thr + 0.03, 0.5, '%g m threshold' % thr, rotation=90,
            va='center', ha='left', fontsize=16)

    ax.set_xlabel('Miss distance (m)', fontsize=16)
    ax.set_ylabel('Cumulative probability', fontsize=16)
    ax.tick_params(labelsize=16)
    ax.set_xlim(0.0, min(x_max * 1.04, 6.0))
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower right', fontsize=14, framealpha=0.95,
              borderpad=0.6, labelspacing=0.4, handlelength=2.5)
    ax.set_title('Terminal miss distance ECDF by configuration', fontsize=16, pad=16)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, 'Miss_ECDF_by_config.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved Miss_ECDF_by_config")


def write_run_meta(folder, results):
    """Write run metadata next to the raw arrays so every npz is self-documenting."""
    import json
    import datetime

    meta = {
        'script': 'TAD_Robust.py',
        'generated_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'n_configs': len(results),
        'n_runs_per_config': int(results[0]['n_runs']) if results else None,
        'total_runs': int(sum(r['n_runs'] for r in results)),
        'solver': 'Dormand-Prince 5(4) RK45 (scipy solve_ivp), max_step = 1 ms',
        'rtol': RTOL,
        'atol': ATOL,
        'sensor_dt_s': SENSOR_DT,
        'rates': {'integration': 'RK45, max_step 1 ms, restart every 20 ms',
                  'sensor': '50 Hz',
                  'filter': '50 Hz, state updated once per sample, zero-order hold',
                  'guidance': 'evaluated at every integrator step from the held set'},
        'sample_and_hold': '50 Hz full-state sample-and-hold; no interpolation',
        'noise_scope': ('Va 1 m/s, theta_A 0.1 deg, R_at 20 m, q_at 0.1 deg, '
                        'R_ad 20 m, q_ad 0.1 deg, R_ad_dot 1 m/s, R_at_dot 1 m/s'),
        'derived': ('theta_a_dot and q_at_dot from the held angle samples by a '
                    'first-order tracking differentiator (tau = 0.25 s); '
                    'a_A = Va * theta_a_dot'),
        'tgo_max_s': TGO_MAX,
        'r_protect_m': R_PROTECT,
        'a_D_max_g': A_D_MAX / g,
        'success_threshold_m': 1.0,
        'seed_base': 1000,
        'seed_offsets': {'Va': 0, 'theta_A': 1, 'R_at': 2, 'q_at': 3,
                         'R_ad_dot': 4, 'target_overload': 5,
                         'R_ad': 6, 'q_ad': 7, 'R_at_dot': 8},
        'bootstrap_seed': 42,
        'npz_fields': ['miss_vals', 'miss_x', 'miss_y', 'a_D_peak', 'a_T_peak',
                       'a_A_peak', 'ZEM_end', 'ZEAE1_end', 'ZEAE2_end',
                       'err_Va', 'err_theta_a', 'err_R_at', 'err_q_at',
                       'err_a_a', 'termination_modes'],
        'configs': [{'R0_m': int(r['dist']), 'Vd_m_s': int(r['Vd']),
                     'n_runs': int(r['n_runs'])} for r in results],
    }
    path = os.path.join(folder, 'run_meta.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Run metadata saved: {path}")


# ===================== Main program =====================
def run_one_config(args):
    """Run one MC configuration (parallel worker)."""
    dist, Vd_test = args
    folder = DEFAULT_FOLDER
    tc.pos_A0[0] = float(dist)
    tc.pos_A0[1] = 8000.0

    print(f"\nRunning MC: dist={dist}m, Vd={Vd_test}m/s...")
    print("  [scope] 50 Hz sample-and-hold, no interpolation; noise on "
          "Va/theta_A/R_at/q_at/R_ad/q_ad/range rates")
    plot_name = f'overload_history_{dist}_{Vd_test}'
    (miss_vals, miss_x, miss_y,
     a_D_peak, a_T_peak, a_A_peak,
     zem_end, zeae1_end, zeae2_end,
     err_Va, err_theta, err_R, err_q,
     err_a, termination_modes) = run_mc_with_noise_rk45(
        Vd=float(Vd_test), num_runs=2000,
        sensor_dt=0.01, seeker_noise=True,
        tau_track=0.20, seed_base=1000,
        plot_last=True, plot_name=plot_name)

    # move the overload plot into the output folder
    src = f'{plot_name}.png'
    if os.path.exists(src):
        dst = os.path.join(folder, src)
        if os.path.exists(dst):
            os.remove(dst)
        os.rename(src, dst)

    if len(miss_vals) > 0:
        miss_arr = np.array(miss_vals)
        n = len(miss_arr)
        miss_mean = float(np.mean(miss_arr))
        miss_std = float(np.std(miss_arr, ddof=1))
        t_975 = 1.96 + 2.41 / n
        ci_mean_low = miss_mean - t_975 * miss_std / np.sqrt(n)
        ci_mean_high = miss_mean + t_975 * miss_std / np.sqrt(n)

        rng_boot = np.random.default_rng(42)
        boot_q95 = []
        for _ in range(500):
            sample = rng_boot.choice(miss_arr, size=n, replace=True)
            boot_q95.append(np.percentile(sample, 95))
        boot_q95 = np.array(boot_q95)
        ci_q95_low = float(np.percentile(boot_q95, 2.5))
        ci_q95_high = float(np.percentile(boot_q95, 97.5))

        result_row = {
            'dist': dist, 'Vd': Vd_test,
            'n_runs': n,
            'miss_mean': miss_mean,
            'miss_mean_ci95_low': float(ci_mean_low),
            'miss_mean_ci95_high': float(ci_mean_high),
            'miss_median': float(np.median(miss_arr)),
            'miss_max': float(np.max(miss_arr)),
            'miss_95': float(np.percentile(miss_arr, 95)),
            'miss_95_ci95_low': ci_q95_low,
            'miss_95_ci95_high': ci_q95_high,
            'a_D_peak_mean': float(np.mean(a_D_peak)),
            'a_D_peak_max': float(np.max(a_D_peak)),
            'a_D_peak_95': float(np.percentile(a_D_peak, 95)),
        }

        success_count = np.sum(miss_arr <= 1.0)
        p_success = success_count / n
        z = 1.96
        denominator = 1 + z ** 2 / n
        center = (p_success + z ** 2 / (2 * n)) / denominator
        margin = z * np.sqrt(p_success * (1 - p_success) / n + z ** 2 / (4 * n ** 2)) / denominator
        result_row['success_count'] = int(success_count)
        result_row['p_success'] = float(p_success)
        result_row['p_success_ci95_low'] = float(max(0, center - margin))
        result_row['p_success_ci95_high'] = float(min(1, center + margin))

        term_arr = np.array(termination_modes)
        result_row['n_intercept'] = int(np.sum(term_arr == 'intercept'))
        result_row['n_target_hit'] = int(np.sum(term_arr == 'target_hit'))
        result_row['n_pass_cross'] = int(np.sum(term_arr == 'pass_cross'))
        result_row['n_timeout'] = int(np.sum(term_arr == 'timeout'))

        print(f"  -> miss: mean={result_row['miss_mean']:.3f} "
              f"[{ci_mean_low:.3f}, {ci_mean_high:.3f}], "
              f"95%={result_row['miss_95']:.3f} "
              f"[{ci_q95_low:.3f}, {ci_q95_high:.3f}]")
        print(f"  -> success (miss<=1m): {success_count}/{n} = {p_success:.4f} "
              f"[{result_row['p_success_ci95_low']:.4f}, {result_row['p_success_ci95_high']:.4f}]")
        print(f"  -> DPeak: mean={result_row['a_D_peak_mean']:.2f}g, "
              f"95%={result_row['a_D_peak_95']:.2f}g, "
              f"max={result_row['a_D_peak_max']:.2f}g")

        os.makedirs(folder, exist_ok=True)
        np.savez(os.path.join(folder, f'mc_raw_{dist}_{Vd_test}.npz'),
                 miss_vals=np.array(miss_vals), miss_x=np.array(miss_x), miss_y=np.array(miss_y),
                 a_D_peak=np.array(a_D_peak), a_T_peak=np.array(a_T_peak), a_A_peak=np.array(a_A_peak),
                 ZEM_end=np.array(zem_end), ZEAE1_end=np.array(zeae1_end), ZEAE2_end=np.array(zeae2_end),
                 err_Va=np.array(err_Va), err_theta_a=np.array(err_theta),
                 err_R_at=np.array(err_R), err_q_at=np.array(err_q), err_a_a=np.array(err_a),
                 termination_modes=np.array(termination_modes))

        if dist == 6000 and Vd_test == 400:
            plot_error_distributions(err_Va, err_theta, err_R, err_q, folder)
            plot_acceleration_error(err_a, folder)

        result_row['_miss_x'] = np.array(miss_x)
        result_row['_miss_y'] = np.array(miss_y)
        result_row['_miss_vals'] = np.array(miss_vals)
        return result_row
    return None


if __name__ == "__main__":
    import multiprocessing as mp

    folder = DEFAULT_FOLDER
    os.makedirs(folder, exist_ok=True)

    geoms = [6000, 8000, 10000]
    vds = [400, 500, 600]

    configs = [(dist, Vd) for dist in geoms for Vd in vds]
    print(f"Total configurations: {len(configs)} x 2000 runs = {len(configs) * 2000}")
    print("Noise scope : 50 Hz sample-and-hold, no interpolation; "
          "Va 1 m/s, angles 0.1 deg, ranges 20 m, range rates 1 m/s (Doppler)")
    print(f"Output dir  : {folder}")

    with mp.Pool(9) as pool:
        results = pool.map(run_one_config, configs)

    all_results = [r for r in results if r is not None]
    all_results.sort(key=lambda r: (r['dist'], r['Vd']))

    print("\n" + "=" * 120)
    print(f"{'dist':>8} {'Vd':>5} {'n_runs':>6} {'miss_mean':>10} {'Mean_95CI':>20} "
          f"{'miss_95':>10} {'P95_95CI':>20} {'p_succ':>8} {'p_succ_95CI':>20} "
          f"{'Dpeak_mean':>10} {'Dpeak_95':>10} {'Dpeak_max':>10}")
    print("-" * 120)
    for r in all_results:
        ci_mean = f"[{r['miss_mean_ci95_low']:.2f},{r['miss_mean_ci95_high']:.2f}]"
        ci_q95 = f"[{r['miss_95_ci95_low']:.2f},{r['miss_95_ci95_high']:.2f}]"
        ci_ps = f"[{r['p_success_ci95_low']:.2f},{r['p_success_ci95_high']:.2f}]"
        print(f"{r['dist']:>8} {r['Vd']:>5} {r['n_runs']:>6} "
              f"{r['miss_mean']:>10.2f} {ci_mean:>20} "
              f"{r['miss_95']:>10.2f} {ci_q95:>20} "
              f"{r['p_success']:>8.2f} {ci_ps:>20} "
              f"{r['a_D_peak_mean']:>10.2f} {r['a_D_peak_95']:>10.2f} {r['a_D_peak_max']:>10.2f}")
    print("=" * 120)

    csv_path = os.path.join(folder, 'mc_summary.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['dist', 'Vd', 'n_runs', 'miss_mean', 'Mean_95CI', 'miss_median', 'miss_max',
                         'miss_95', 'P95_95CI', 'success_count', 'p_success', 'p_success_95CI',
                         'a_D_peak_mean', 'a_D_peak_95', 'a_D_peak_max',
                         'n_intercept', 'n_target_hit', 'n_pass_cross', 'n_timeout'])
        for r in all_results:
            mean_ci = f"[{r['miss_mean_ci95_low']:.2f}, {r['miss_mean_ci95_high']:.2f}]"
            p95_ci = f"[{r['miss_95_ci95_low']:.2f}, {r['miss_95_ci95_high']:.2f}]"
            p_succ_ci = f"[{r['p_success_ci95_low']:.2f}, {r['p_success_ci95_high']:.2f}]"
            writer.writerow([
                r['dist'], r['Vd'], r['n_runs'],
                f"{r['miss_mean']:.2f}", mean_ci,
                f"{r['miss_median']:.2f}", f"{r['miss_max']:.2f}",
                f"{r['miss_95']:.2f}", p95_ci,
                r['success_count'], f"{r['p_success']:.2f}", p_succ_ci,
                f"{r['a_D_peak_mean']:.2f}", f"{r['a_D_peak_95']:.2f}", f"{r['a_D_peak_max']:.2f}",
                r['n_intercept'], r['n_target_hit'], r['n_pass_cross'], r['n_timeout']
            ])
    print(f"Summary saved: {csv_path}")

    # ===== aggregate plots: all 9 configurations (18,000 runs) =====
    all_miss_x = np.concatenate([r['_miss_x'] for r in all_results if '_miss_x' in r])
    all_miss_y = np.concatenate([r['_miss_y'] for r in all_results if '_miss_y' in r])
    all_miss_vals = np.concatenate([r['_miss_vals'] for r in all_results if '_miss_vals' in r])
    print(f"Aggregate: {len(all_miss_vals)} total runs across {len(all_results)} configurations")

    plt.figure(figsize=(8, 7))
    plt.plot(all_miss_x, all_miss_y, 'k.', markersize=3)
    plt.xlabel('x (m)', fontsize=14)
    plt.ylabel('y (m)', fontsize=14)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('Terminal hit accuracy (all configurations)', fontsize=16, pad=15)
    plt.axis('equal'); plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, 'Terminal_Scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()

    n_total = len(all_miss_vals)
    n_per_config = len(all_results[0]['_miss_vals']) if all_results else 2000
    plt.figure(figsize=(14, 7))
    plt.plot(np.arange(1, n_total + 1), all_miss_vals, 'k.', markersize=3)
    plt.xlabel('Number of trajectories', fontsize=14)
    plt.ylabel('Miss distance (m)', fontsize=14)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlim(0, n_total); plt.ylim(bottom=0)
    plt.grid(True, linestyle='--', alpha=0.5)
    for i, r in enumerate(all_results):
        x_start = i * n_per_config + 1
        x_mid = x_start + n_per_config / 2
        if i > 0:
            plt.axvline(x=x_start, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
        label = f"{r['dist']}m\nVd={r['Vd']}m/s"
        plt.text(x_mid, plt.ylim()[1] * 0.95, label,
                 ha='center', va='top', fontsize=16,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
    plt.title('Monte Carlo miss distance statistics (all configurations)', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, 'Miss_Scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()

    plot_cdf_miss(all_miss_vals, folder)
    plot_miss_histogram(all_miss_vals, folder)
    plot_ecdf_grouped(all_results, folder)
    write_run_meta(folder, all_results)
    print("Aggregate plots saved.")
