# Release validation

Checked on 29 October 2026 in an isolated publishing copy.

- All 21 original E1 numerical correctness checks passed, including independent scikit-learn and SciPy references and finite-difference gradient checks.
- Notebook schema and transformed Python syntax validated across 27 code cells.
- Exported model functions preserve the notebook's covariance, gradients, objective, folds and numerical constants.
- Historical frozen-study archive retained separately; publishing files scanned for credentials.

Not performed: full scientific refitting of the 100 synthetic replicates and real-data blocks. Historical evidence is labelled as historical. Clearing notebook outputs and renaming files does not redefine the frozen scientific protocol or regenerate its original checksum manifests.
