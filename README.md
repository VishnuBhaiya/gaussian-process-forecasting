# Gaussian Process Forecasting

**Does marginal likelihood select the best forecast?** A from-scratch exact locally periodic Gaussian-process study comparing log marginal likelihood with blocked rolling-origin cross-validation for joint 11-year forecasts.

## Quick numerical verification

```bash
python -m venv .venv
# Activate the environment, then:
python -m pip install -r requirements.txt
python verify_correctness.py
```

This runs the notebook's 21 checks: covariance properties, analytic-gradient finite differences, agreement with scikit-learn, observation-noise accounting, SciPy joint-density scores, chronological folds and training-only standardisation. It does not refit the full scientific study.

## Research design

- One exact locally periodic covariance family, two hyperparameter-selection criteria.
- Shared bounds, eight frozen starts, reproducible seeds and explicit failure handling.
- Correctly specified synthetic data, period drift and annual sunspot forecasting.
- Joint negative log predictive density as the primary score, with error and calibration diagnostics.
- Distinct latent uncertainty, observation noise and numerical jitter.

`gp.py` is exported from the original notebook model cells. The [portfolio notebook](notebooks/gaussian-process-forecasting.ipynb) contains the complete protocol and experiment workflow; install `requirements-notebook.txt` to run it. Full fitting is substantially slower than the correctness gate. Keep new runs separate from the preserved frozen results.

The [historical evidence archive](evidence/frozen-study.zip) preserves the original scientific outputs and manifests. It is historical evidence, not a fresh full-study execution during publication. Notebook outputs were cleared for the source-only portfolio edition. The original archive checksums refer to its historical files, not to renamed portfolio sources.

## Limits

Exact GPs have cubic factorisation cost; findings concern this model, protocol and data. Ten real-data outer blocks do not establish broad forecasting superiority. See [validation](docs/VALIDATION.md) for the release checks.

Author: Vishnu Bhaiya. NumPy/SciPy implement the model; scikit-learn is an independent correctness reference.
