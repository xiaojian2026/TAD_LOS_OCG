# TAD_LOS_OCG

Simulation code for the LOS-based Optimal Cooperative Guidance (LOS-OCG) law
for two-dimensional (2D) target–attacker–defender (TAD) cooperative interception.

The method combines modified line-of-sight (LOS) geometry with optimal control
in a receding-horizon frozen single-cycle framework. It requires neither an
identified attacker guidance law nor explicit multi-step maneuver forecasting,
and applies three finite terminal penalties (zero-effort miss, LOS collinearity
angle error, target flight-direction error) with tanh-based time-varying weight
scheduling. The main closed loop is integrated with an adaptive RK45
(Dormand–Prince 5(4)) solver.

This repository contains the source code reproducing all simulation results of
the accompanying manuscript (Aerospace, manuscript ID: 4441326).

---

## Requirements

- Python 3.9+ (tested on 3.10)
- `numpy`, `scipy`, `matplotlib`, `openpyxl`

Install dependencies:

```bash
pip install numpy scipy matplotlib openpyxl
```

---

## File structure

| File | Description | Corresponding manuscript section |
|---|---|---|
| `TAD_Comp.py` | Core simulation and comparison of four guidance laws (APN, LOSIG, C-ECDG, LOS-OCG). Exports the shared simulation kernel (`run_single_simulation`, `get_guidance_params`). The main integration uses the adaptive **RK45** solver. | Section 4.2 |
| `TAD_Batch.py` | Large-scale batch simulation (6,435 trajectories) over attacker velocity, engagement distance, altitude offset, and defender velocity. | Section 4.5 |
| `TAD_Batch_Replot.py` | Re-plots batch results as heatmaps (Vd = 400/500/600 m/s) and grouped ECDF from `Batch_data_figure/batch_results.csv`. | Section 4.5 |
| `TAD_Robust.py` | Monte Carlo robustness test (18,000 runs, 9-channel noise, unified RK45 kernel, 50 Hz sensor update). | Section 4.5 |
| `TAD_StepStudy_Full.py` | Appendix A step-size / solver study: explicit Euler (fixed steps of 1, 0.5, 0.25, 0.125 ms) vs adaptive RK45 across nine configurations; writes Tables A1–A2. The fixed-step Euler is used only for this step-size study. | Appendix A |
| `TAD_ECDG_Compare.py` | Side-by-side comparison of the published ECDG (Ref. [21], Eq. 34) vs the corrected ECDG (Appendix B, Eq. B.13). | Appendix B |
| `TAD_Sensitivity_Multi.py` | Multi-geometry weight-parameter sensitivity analysis (Vd = 400/600, R0 = 6000/10000 m). | Section 4.6 |
| `TAD_Appendix_B.py` | Numerical unit test: verifies the closed-form φ (Eq. 75 / Eq. B.13) against direct RK45 integration. | Appendix B |

## Output directories

| Directory | Contents |
|---|---|
| `Comp_data_figure/` | Section 4.2 comparative trajectory/overload figures and summary tables |
| `Batch_data_figure/` | Batch simulation results (e.g., `batch_results.csv`) and figures |
| `Robust_data_figure/` | Monte Carlo robustness figures and statistics |
| `Appendix_data/` | Step-study tables (Tables A1–A2, `.csv`) |
| `TAD_ECDG_Compare_data/` | ECDG comparison results |
| `TAD_Sensitivity_data/`, `Sensitivity_Analysis_data/` | Sensitivity analysis results |
| `TAD_Appendix_B_data/` | Appendix B unit-test results |
| `legacy/` | Deprecated/older scripts (not needed to reproduce the paper) |

---

## Usage

Each script runs independently. Run the core comparison first:

python TAD_Comp.py


Then, depending on the section to reproduce:

python TAD_Batch.py            # Section 4.5 batch simulation
python TAD_Batch_Replot.py     # re-plot batch figures (requires TAD_Batch output)
python TAD_Robust.py           # Section 4.5 Monte Carlo robustness
python TAD_StepStudy_Full.py   # Appendix A step-size / solver study
python TAD_ECDG_Compare.py     # Appendix B ECDG comparison
python TAD_Sensitivity_Multi.py# Section 4.6 sensitivity analysis
python TAD_Appendix_B.py       # Appendix B closed-form φ unit test




---

## License

MIT License — see `LICENSE`.
