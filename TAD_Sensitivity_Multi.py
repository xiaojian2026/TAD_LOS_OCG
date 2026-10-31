"""
Multi-geometry sensitivity analysis
Vd = 400, 600 m/s
R0 = 6000, 10000 m
Alt = 3000 m (fixed)
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import TAD_Comp
from TAD_Comp import run_single_simulation, get_guidance_params


def set_geometry(R0, alt, Vd):
    """Set initial geometry by modifying global variables."""
    TAD_Comp.pos_D0 = np.array([0.0, 0.0])  # Defender at origin
    TAD_Comp.pos_T0 = TAD_Comp.pos_D0.copy()  # Target at same position as defender
    TAD_Comp.pos_A0 = np.array([R0, alt])  # Attacker at (R0, alt)


def run_case4_sensitivity(Vd, R0=6000, alt=3000, a_max=None, m_max=None, n_max=None, alpha=None, beta=None, m1=None, m1_a=None):
    """Run Case 4 with customized weight parameters via monkey patching."""
    
    set_geometry(R0, alt, Vd)
    
    original_get_params = TAD_Comp.get_guidance_params
    
    def patched_get_params(CASE, Vd_val):
        params = original_get_params(CASE, Vd_val)
        if CASE == 4:
            if a_max is not None:
                params['a_max'] = a_max
            if m_max is not None:
                params['m_max'] = m_max
            if n_max is not None:
                params['n_max'] = n_max
            if alpha is not None:
                params['alpha'] = alpha
            if beta is not None:
                params['beta'] = beta
            if m1 is not None:
                params['m1'] = m1
            if m1_a is not None:
                params['m1_a'] = m1_a
        return params
    
    TAD_Comp.get_guidance_params = patched_get_params
    
    try:
        result = run_single_simulation(4, Vd, save_traj=False)
    finally:
        TAD_Comp.get_guidance_params = original_get_params
    
    return result


def main():
    print("=" * 90)
    print("Multi-geometry Sensitivity Analysis")
    print("=" * 90)
    
    # Baseline parameters
    base_a_max = 3e4
    base_m_max = 1e5
    base_n_max = 1e5
    base_alpha = 1.0
    base_beta = 1.0
    base_m1 = 2.0
    base_m1_a = 4.0
    
    # Test geometries
    geometries = [
        {"Vd": 400, "R0": 6000, "alt": 3000},
        {"Vd": 400, "R0": 10000, "alt": 3000},
        {"Vd": 600, "R0": 6000, "alt": 3000},
        {"Vd": 600, "R0": 10000, "alt": 3000},
    ]
    
    # Sensitivity cases
    cases = [
        ("Baseline",                    None,       None,       None,       None,   None,   None,       None),
        ("a_max -20%",                  base_a_max*0.8, None,   None,       None,   None,   None,       None),
        ("a_max +20%",                  base_a_max*1.2, None,   None,       None,   None,   None,       None),
        ("m_max -20%",                  None,       base_m_max*0.8, None,   None,   None,   None,       None),
        ("m_max +20%",                  None,       base_m_max*1.2, None,   None,   None,   None,       None),
        ("n_max -20%",                  None,       None,       base_n_max*0.8, None, None,   None,       None),
        ("n_max +20%",                  None,       None,       base_n_max*1.2, None, None,   None,       None),
        ("alpha -20%",                  None,       None,       None,       base_alpha*0.8, None, None,    None),
        ("alpha +20%",                  None,       None,       None,       base_alpha*1.2, None, None,    None),
        ("beta -20%",                   None,       None,       None,       None,   base_beta*0.8, None,  None),
        ("beta +20%",                   None,       None,       None,       None,   base_beta*1.2, None,  None),
        ("m1 -20%",                     None,       None,       None,       None,   None,   base_m1*0.8, None),
        ("m1 +20%",                     None,       None,       None,       None,   None,   base_m1*1.2, None),
        ("m1_a -20%",                   None,       None,       None,       None,   None,   None,       base_m1_a*0.8),
        ("m1_a +20%",                   None,       None,       None,       None,   None,   None,       base_m1_a*1.2),
    ]
    
    all_results = []
    
    for geom in geometries:
        Vd = geom["Vd"]
        R0 = geom["R0"]
        alt = geom["alt"]
        
        print(f"\n{'='*90}")
        print(f"Geometry: Vd={Vd} m/s, R0={R0} m, alt={alt} m")
        print(f"{'='*90}")
        
        # Run baseline
        print("  Running baseline...")
        base_result = run_case4_sensitivity(Vd, R0=R0, alt=alt)
        base_miss = base_result['miss_final_m']
        base_dpeak = base_result['a_D_sf_peak_g']
        print(f"  Baseline: miss = {base_miss:.3f} m, a_d^max = {base_dpeak:.2f} g")
        
        # Run sensitivity cases
        for name, a_max, m_max, n_max, alpha, beta, m1, m1_a in cases[1:]:
            res = run_case4_sensitivity(Vd, R0=R0, alt=alt, a_max=a_max, m_max=m_max, n_max=n_max, alpha=alpha, beta=beta, m1=m1, m1_a=m1_a)
            miss = res['miss_final_m']
            dpeak = res['a_D_sf_peak_g']
            miss_change = (miss - base_miss) / base_miss * 100
            dpeak_change = (dpeak - base_dpeak) / base_dpeak * 100
            all_results.append({
                "Vd": Vd, "R0": R0, "alt": alt,
                "param": name.split(' ')[0],
                "perturb": name.split(' ')[1] if len(name.split(' ')) > 1 else "baseline",
                "miss": miss, "miss_change": miss_change,
                "dpeak": dpeak, "dpeak_change": dpeak_change
            })
            print(f"    {name:18s}: miss = {miss:.3f} m ({miss_change:+6.1f}%), a_d^max = {dpeak:.2f} g ({dpeak_change:+6.1f}%)")
    
    # Save results to Sensitivity_Analysis_data folder
    out_dir = 'Sensitivity_Analysis_data'
    os.makedirs(out_dir, exist_ok=True)
    
    # --- Save CSV ---
    import csv
    csv_path = os.path.join(out_dir, 'sensitivity_multi_geometry.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Vd (m/s)', 'R0 (m)', 'alt (m)', 'Parameter', 'Perturbation', 'Miss (m)', 'ΔMiss (%)', 'a_d^max (g)', 'Δa_d^max (%)'])
        for r in all_results:
            writer.writerow([r["Vd"], r["R0"], r["alt"], r["param"], r["perturb"], 
                           f"{r['miss']:.2f}", f"{r['miss_change']:+.2f}", 
                           f"{r['dpeak']:.2f}", f"{r['dpeak_change']:+.2f}"])
    print(f"\nCSV saved to: {csv_path}")
    
    # --- Generate Table 3: Miss distance variations ---
    params_list = ['a_max', 'm_max', 'n_max', 'alpha', 'beta', 'm1', 'm1_a']
    geoms_list = [(400, 6000), (400, 10000), (600, 6000), (600, 10000)]
    
    # Build lookup: (param, perturb, Vd, R0) -> miss_change
    miss_table = {}
    dpeak_table = {}
    for r in all_results:
        key = (r['param'], r['perturb'], r['Vd'], r['R0'])
        miss_table[key] = r['miss_change']
        dpeak_table[key] = r['dpeak_change']
    
    # Table 3: Miss distance variations (CSV)
    table3_path = os.path.join(out_dir, 'Table3_miss_distance_variations.csv')
    with open(table3_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['Parameter', 'Perturbation'] + [f'Vd={Vd} R0={R0}' for Vd, R0 in geoms_list])
        for p in params_list:
            for perturb in ['-20%', '+20%']:
                row = [p, perturb]
                for Vd, R0 in geoms_list:
                    val = miss_table.get((p, perturb, Vd, R0), float('nan'))
                    row.append(f"{val:+.2f}")
                writer.writerow(row)
    print(f"Table 3 saved to: {table3_path}")
    
    # Table 4: Peak overload variations (CSV)
    table4_path = os.path.join(out_dir, 'Table4_peak_overload_variations.csv')
    with open(table4_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['Parameter', 'Perturbation'] + [f'Vd={Vd} R0={R0}' for Vd, R0 in geoms_list])
        for p in params_list:
            for perturb in ['-20%', '+20%']:
                row = [p, perturb]
                for Vd, R0 in geoms_list:
                    val = dpeak_table.get((p, perturb, Vd, R0), float('nan'))
                    row.append(f"{val:+.2f}")
                writer.writerow(row)
    print(f"Table 4 saved to: {table4_path}")
    print(f"Table 4 saved to: {table4_path}")


if __name__ == "__main__":
    main()
