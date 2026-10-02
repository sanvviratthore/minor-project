#!/usr/bin/env bash
# Runs the whole pipeline in order. Total time: about 30-60 minutes on a normal laptop.
set -e
python verify_model.py          # 1. checks the physics model against published values (seconds)
python generate_dataset.py      # 2. writes gear_power_loss_dataset.csv (seconds)
python train_models.py          # 3. tunes and benchmarks ANN, SVR, RF, XGBoost (longest step, 20-40 min)
python explain.py               # 4. SHAP on XGBoost (under a minute)
python explain_ann_check.py     # 5. SHAP cross-check on the ANN (a few minutes)
python optimize.py ANN          # 6. NSGA-II nominal + robust optimisation (2-3 min)
python fig_pipeline.py          # 7. pipeline diagram
python make_figures.py ANN      # 8. all result figures
