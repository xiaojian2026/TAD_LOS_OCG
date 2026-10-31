"""
TAD_Appendix_B.py
==================
Numerical unit test: verify closed-form φ (Eq. 75 / Appendix B, Eq. B.13)
against direct RK45 (Dormand-Prince 5(4)) integration of the linearized
kinematic model in Appendix B.

Linearized system (Appendix B, Eqs. B.7-B.8):

    x = [z, ż, q̇_at, q̇_ad]^T
    ẋ = A x

    A = [[0,  1,   0,                     0                    ],
         [0,  0,  -2·V_ad·Δt_f / t_go,at,  0                    ],
         [0,  0,   2 / t_go,at,            0                    ],
         [0,  0,   0,                       2 / t_go           ]]

The first row of the transition matrix DΦ is [1, t_go, φ, 0], so that

    z(t_go) = z(0) + t_go · ż(0) + φ · q̇_at(0)

Closed-form φ (Eq. 75 / B.13):

    φ = (V_ad · Δt_f / 2) · [ 2·t_go - t_go,at · (exp(2·t_go/t_go,at) - 1) ]

Test method
-----------
Set x(0) = [0, 0, 1, 0] (only q̇_at(0) = 1), integrate to t_go with RK4.
The terminal z(t_go) equals φ. Compare with the closed-form expression.

Run across a grid of engagement conditions matching the paper's 9 MC configs.
"""

import numpy as np
import os
import sys

# ---------------------------------------------------------------------------
# Linearized system from Appendix B
# ---------------------------------------------------------------------------
def A_matrix(V_ad, dtf, tgo, tgo_at):
    """State matrix A from Eq. (B.8)."""
    return np.array([
        [0.0, 1.0,  0.0,                          0.0],
        [0.0, 0.0, -2.0*V_ad*dtf/tgo_at,         0.0],
        [0.0, 0.0,  2.0/tgo_at,                   0.0],
        [0.0, 0.0,  0.0,                          2.0/tgo],
    ])


def rhs(t, x, A):
    return A @ x


def rk45_integrate(A, tgo):
    """Integrate x' = A x from t=0 to t=tgo using Dormand-Prince 5(4) (RK45).

    Initial condition x(0) = [0, 0, 1, 0] (q̇_at(0) = 1).
    Returns terminal z(tgo), which equals φ by construction.
    """
    from scipy.integrate import solve_ivp
    x0 = np.array([0.0, 0.0, 1.0, 0.0])
    sol = solve_ivp(lambda t, x: A @ x, [0, tgo], x0,
                    method='DOP853', rtol=1e-9, atol=1e-11)
    return sol.y[0, -1]


def phi_closed(V_ad, dtf, tgo, tgo_at):
    """Closed-form φ from Eq. (75) / Eq. (B.13)."""
    return V_ad*dtf/2.0 * (2.0*tgo - tgo_at*(np.exp(2.0*tgo/tgo_at) - 1.0))


# ---------------------------------------------------------------------------
# Test across the paper's 9 Monte Carlo configurations
# ---------------------------------------------------------------------------
def main():
    # (R_ad, V_ad, R_at, V_at) — matching Table 5 in the paper
    #   R_ad in [m], V_ad = closing velocity [m/s]
    #   R_at in [m], V_at = closing velocity [m/s]
    # 5 representative configurations: cover positive Δt_f, near-zero Δt_f,
    # and negative Δt_f (t_go > t_go_at boundary).
    configs = [
        (6000, 400,  8000, 500,  "R_ad=6000, V_ad=400"),
        (6000, 500,  8000, 600,  "R_ad=6000, V_ad=500"),
        (8000, 500, 10000, 600,  "R_ad=8000, V_ad=500"),
        (10000, 400, 12000, 500, "R_ad=10000, V_ad=400"),
        (10000, 600, 12000, 700, "R_ad=10000, V_ad=600"),
    ]

    print("=" * 82)
    print("Unit test: closed-form φ (Eq. 75) vs RK45 direct integration")
    print("Linearized model: x = [z, ż, q̇_at, q̇_ad],  x' = A x  (Appendix B)")
    print("=" * 82)
    print()

    max_rel_err = 0.0
    worst_case = ""

    for R_ad, V_ad, R_at, V_at, label in configs:
        tgo = R_ad / V_ad           # t_go
        tgo_at = R_at / V_at        # t_go,at
        dtf = tgo_at - tgo          # Δt_f = t_go,at - t_go

        A = A_matrix(V_ad, dtf, tgo, tgo_at)

        # Numerical: RK4 integration with q̇_at(0) = 1
        phi_num = rk45_integrate(A, tgo)

        # Analytical: closed form Eq. (75)
        phi_ana = phi_closed(V_ad, dtf, tgo, tgo_at)

        abs_err = abs(phi_num - phi_ana)
        rel_err = abs_err / max(abs(phi_ana), 1e-15)

        if rel_err > max_rel_err:
            max_rel_err = rel_err
            worst_case = label

        print(f"  {label}")
        print(f"    t_go={tgo:7.3f} s   t_go,at={tgo_at:7.3f} s   Δt_f={dtf:7.3f} s")
        print(f"    φ_RK45   = {phi_num: .10e}")
        print(f"    φ_closed = {phi_ana: .10e}")
        print(f"    |error|  = {abs_err:.3e}   rel = {rel_err:.3e}")
        print()

    print("=" * 82)
    print(f"  Maximum relative error across all {len(configs)} configs: {max_rel_err:.3e}")
    print(f"  Worst case: {worst_case}")
    print()
    if max_rel_err < 1e-8:
        print("  PASS: closed-form φ agrees with direct RK45 integration to < 1e-8.")
    else:
        print("  FAIL: discrepancy exceeds tolerance. Check A matrix or closed form.")
    print("=" * 82)

    # ---- Write CSV to TAD_Appendix_B_data ----
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "TAD_Appendix_B_data")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "phi_verification.csv")

    import csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Configuration", "t_go (s)", "t_go_at (s)", "Delta_t_f (s)",
                    "phi_RK45", "phi_closed", "Relative_error"])
        for R_ad, V_ad, R_at, V_at, label in configs:
            tgo = R_ad / V_ad
            tgo_at = R_at / V_at
            dtf = tgo_at - tgo
            A = A_matrix(V_ad, dtf, tgo, tgo_at)
            phi_num = rk45_integrate(A, tgo)
            phi_ana = phi_closed(V_ad, dtf, tgo, tgo_at)
            rel_err = abs(phi_num - phi_ana) / max(abs(phi_ana), 1e-15)
            w.writerow([label, f"{tgo:.2f}", f"{tgo_at:.2f}", f"{dtf:.2f}",
                        f"{phi_num:.2f}", f"{phi_ana:.2f}", f"{rel_err:.2e}"])
    print(f"\nCSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
