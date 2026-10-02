# Spur gear power-loss surrogates: code and data

Everything needed to regenerate every number, table and figure in the paper
"Interpretable Machine-Learning Surrogates for Load-Dependent Power Loss and
Constraint-Aware Design of Spur Gear Pairs under Friction-Model Uncertainty".

## Setup (once)

1. Install Python 3.10 or newer (python.org, or Anaconda).
2. Open a terminal in this folder.
3. Create an environment and install the libraries:

       python -m venv venv
       venv\Scripts\activate          (Windows)
       source venv/bin/activate       (Mac / Linux)
       pip install -r requirements.txt

## Run

    bash run_all.sh        (Mac / Linux, or Git Bash on Windows)

or run the scripts one by one in this order:

| Step | Script | What it does | Time |
|---|---|---|---|
| 1 | `verify_model.py` | Checks the physics model: H_V for the FZG C40 gear must print 0.1950 (published 0.1959) | seconds |
| 2 | `generate_dataset.py` | Latin hypercube sampling, validity filtering, writes `gear_power_loss_dataset.csv` (6,000 rows) | seconds |
| 3 | `train_models.py` | Tunes and benchmarks ANN, SVR, RF, XGBoost on random + extrapolation splits, learning curves | 20-40 min |
| 4 | `explain.py` | SHAP (TreeSHAP on XGBoost) for the loss and the spread | < 1 min |
| 5 | `explain_ann_check.py` | SHAP cross-check with a permutation explainer on the ANN | a few min |
| 6 | `optimize.py ANN` | NSGA-II nominal and robust optimisation, exact re-evaluation | 2-3 min |
| 7 | `fig_pipeline.py`, `make_figures.py ANN` | All figures, 300 dpi, in `figures/` | seconds |

The `results/` and `figures/` folders already contain the outputs used in the paper,
so you can open them without running anything.

## Files

| File | Role |
|---|---|
| `gear_model.py` | The physics: geometry with profile shift, Ohlendorf H_V, three friction formulas, stresses, validity domain |
| `features.py` | The 12 model inputs (log of torque, speed, viscosity; oil one-hot) |
| `gear_power_loss_dataset.csv` | The dataset (inputs, H_V, mu for M1-M3, losses L1-L3, Lmean, Lmax, spread, stresses) |
| `results/metrics.json` | All accuracy numbers in Tables 3 and Section 7.2-7.4 |
| `results/best_params.json` | Table 4 |
| `results/shap.json`, `results/shap_ann_check.json` | Section 7.5 |
| `results/optimisation_ANN.json`, `results/pareto_*.csv` | Section 7.6 and Table 5 |

## Expected numbers (check yours against these)

- H_V of FZG C40: 0.1950
- Valid samples: 22,283 of 120,000
- Random-split MAPE: SVR 1.05 %, ANN 1.35 %, XGBoost 3.52 %, RF 13.13 %
- Conventional design loss 0.833 %, optimised about 0.48 % at a <= 105 mm

Neural-network results can differ slightly between computers and library versions
(different floating-point order). Differences of a few tenths of a percent are normal;
the ranking of the four models should not change.
