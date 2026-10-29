"""Run the notebook's 21 numerical correctness checks without scientific refitting."""
import json
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from gp import *

E1_RESULTS = []

def record_check(name, value, threshold, passed):
    E1_RESULTS.append({'check': name, 'value': float(value), 'threshold': threshold, 'passed': bool(passed)})
    assert passed, f'E1 FAILED: {name}; value={value}; threshold={threshold}'

rng = np.random.default_rng(E1_SEED)
t_train = np.linspace(0.0, 60.0, 61)
t_test = np.linspace(60.5, 71.0, 11)
theta_ref = np.array([1.10, 35.0, 0.90, 10.8, 0.20])
Kf_ref = locally_periodic_kernel(t_train, t_train, theta_ref)
y_train = rng.multivariate_normal(np.zeros(t_train.size), training_covariance(t_train, theta_ref, jitter=0.0))

symmetry_error = np.max(np.abs(Kf_ref - Kf_ref.T))
minimum_eigenvalue = np.linalg.eigvalsh(Kf_ref).min()
kernel_diagonal_error = np.max(np.abs(np.diag(Kf_ref) - theta_ref[0] ** 2))
training_diagonal_error = np.max(np.abs(np.diag(training_covariance(t_train, theta_ref))
                                            - np.diag(Kf_ref) - theta_ref[4] ** 2 - JITTER))
record_check('kernel symmetry max error', symmetry_error, '< 1e-12', symmetry_error < 1e-12)
record_check('kernel minimum eigenvalue', minimum_eigenvalue, '>= -1e-10', minimum_eigenvalue >= -1e-10)
record_check('clean-kernel diagonal error', kernel_diagonal_error, '< 1e-12', kernel_diagonal_error < 1e-12)
record_check('training diagonal increment error', training_diagonal_error, '< 1e-12', training_diagonal_error < 1e-12)
def central_difference_gradient(function, x, step=1e-6):
    x = np.asarray(x, dtype=float)
    gradient = np.empty_like(x)
    for k in range(x.size):
        shift = np.zeros_like(x)
        shift[k] = step
        gradient[k] = (function(x + shift) - function(x - shift)) / (2.0 * step)
    return gradient

gradient_test_thetas = [
    theta_ref,
    np.array([0.75, 18.0, 1.40, 9.4, 0.12]),
    np.array([1.65, 90.0, 0.55, 12.6, 0.35]),
]
gradient_rows = []
for case, theta in enumerate(gradient_test_thetas, start=1):
    log_theta = np.log(theta)
    _, analytic = negative_lml_and_grad(log_theta, t_train, y_train)
    numeric = central_difference_gradient(lambda z: negative_lml_and_grad(z, t_train, y_train)[0], log_theta)
    relative_error = np.linalg.norm(analytic - numeric) / max(1.0, np.linalg.norm(analytic), np.linalg.norm(numeric))
    gradient_rows.append({'case': case, 'relative_error': relative_error,
                          **{f'analytic_{name}': analytic[i] for i, name in enumerate(PARAMETER_NAMES)},
                          **{f'numeric_{name}': numeric[i] for i, name in enumerate(PARAMETER_NAMES)}})
    record_check(f'gradient relative error case {case}', relative_error, '< 1e-4', relative_error < 1e-4)
pd.DataFrame(gradient_rows)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, ExpSineSquared, WhiteKernel

sigma_f, ell_d, ell_p, period, sigma_n = theta_ref
sk_kernel = (ConstantKernel(sigma_f ** 2, constant_value_bounds='fixed')
             * RBF(ell_d, length_scale_bounds='fixed')
             * ExpSineSquared(ell_p, period, length_scale_bounds='fixed', periodicity_bounds='fixed')
             + WhiteKernel(sigma_n ** 2, noise_level_bounds='fixed'))
sk_gp = GaussianProcessRegressor(kernel=sk_kernel, alpha=JITTER, optimizer=None, normalize_y=False)
sk_gp.fit(t_train[:, None], y_train)
sk_latent_matrix = sk_gp.kernel_.k1(t_train[:, None])
latent_kernel_error = np.max(np.abs(Kf_ref - sk_latent_matrix))
record_check('sklearn latent-kernel max error', latent_kernel_error, '< 1e-6', latent_kernel_error < 1e-6)
sk_mean, sk_cov = sk_gp.predict(t_test[:, None], return_cov=True)
our_mean, our_cov = gp_predict(t_train, y_train, t_test, theta_ref, observed=True)
our_nll, our_nll_grad = negative_lml_and_grad(np.log(theta_ref), t_train, y_train)
sk_lml = sk_gp.log_marginal_likelihood()

mean_error = np.max(np.abs(our_mean - sk_mean))
covariance_error = np.max(np.abs(our_cov - sk_cov))
lml_error = abs(-our_nll - sk_lml)
record_check('sklearn predictive mean max error', mean_error, '< 1e-6', mean_error < 1e-6)
record_check('sklearn predictive covariance max error', covariance_error, '< 1e-6', covariance_error < 1e-6)
record_check('sklearn LML absolute error', lml_error, '< 1e-6', lml_error < 1e-6)
predictive_symmetry_error = np.max(np.abs(our_cov - our_cov.T))
minimum_predictive_variance = np.diag(our_cov).min()
minimum_predictive_eigenvalue = np.linalg.eigvalsh(0.5 * (our_cov + our_cov.T)).min()
record_check('predictive covariance symmetry error', predictive_symmetry_error, '< 1e-12', predictive_symmetry_error < 1e-12)
record_check('minimum predictive variance', minimum_predictive_variance, '>= -1e-10', minimum_predictive_variance >= -1e-10)
record_check('predictive covariance minimum eigenvalue', minimum_predictive_eigenvalue, '>= -1e-10', minimum_predictive_eigenvalue >= -1e-10)

# A second sklearn object leaves hyperparameters free only so eval_gradient=True is available.
sk_free_kernel = (ConstantKernel(sigma_f ** 2, (1e-8, 1e8))
                  * RBF(ell_d, (1e-8, 1e8))
                  * ExpSineSquared(ell_p, period, (1e-8, 1e8), (1e-8, 1e8))
                  + WhiteKernel(sigma_n ** 2, (1e-8, 1e8)))
sk_free = GaussianProcessRegressor(kernel=sk_free_kernel, alpha=JITTER, optimizer=None, normalize_y=False)
sk_free.fit(t_train[:, None], y_train)
_, sk_gradient = sk_free.log_marginal_likelihood(sk_free.kernel_.theta, eval_gradient=True)
sk_names = [hp.name for hp in sk_free.kernel_.hyperparameters]
expected_names = [
    'k1__k1__k1__constant_value', 'k1__k1__k2__length_scale',
    'k1__k2__length_scale', 'k1__k2__periodicity', 'k2__noise_level'
]
assert sk_names == expected_names, f'Unexpected sklearn parameter order: {sk_names}'
mapped_sk_lml_gradient = np.array([2.0 * sk_gradient[0], sk_gradient[1], sk_gradient[2],
                                   sk_gradient[3], 2.0 * sk_gradient[4]])
gradient_mapping_error = np.max(np.abs(our_nll_grad + mapped_sk_lml_gradient))
record_check('sklearn mapped NLL-gradient max error', gradient_mapping_error, '< 1e-6', gradient_mapping_error < 1e-6)
pd.DataFrame({'sklearn_name': sk_names, 'mapped_LML_gradient': mapped_sk_lml_gradient,
              'our_NLL_gradient': our_nll_grad})
_, latent_cov = gp_predict(t_train, y_train, t_test, theta_ref, observed=False)
noise_increment = our_cov - latent_cov
noise_error = np.max(np.abs(noise_increment - sigma_n ** 2 * np.eye(t_test.size)))
record_check('observation noise added exactly once', noise_error, '< 1e-12', noise_error < 1e-12)

y_test = rng.multivariate_normal(our_mean, our_cov)
our_score = joint_nlpd(y_test, our_mean, our_cov, per_observation=True)
reference_score = -multivariate_normal.logpdf(y_test, mean=our_mean, cov=our_cov) / y_test.size
jnlpd_error = abs(our_score - reference_score)
record_check('JNLPD versus scipy reference', jnlpd_error, '< 1e-10', jnlpd_error < 1e-10)
folds = inner_cv_blocks(200)
expected_validation = [(156, 166), (167, 177), (178, 188), (189, 199)]
actual_validation = [(int(v[0]), int(v[-1])) for _, v in folds]
folds_valid = actual_validation == expected_validation and all(
    train.max() < valid.min() and len(valid) == 11 for train, valid in folds)
record_check('inner-fold boundary/leakage check', int(folds_valid), '== 1', folds_valid)
year_folds = inner_cv_blocks_from_year(1899)
actual_year_ranges = [(SUNSPOT_FIRST_YEAR + int(v[0]), SUNSPOT_FIRST_YEAR + int(v[-1]))
                      for _, v in year_folds]
expected_year_ranges = [(1855, 1865), (1866, 1876), (1877, 1887), (1888, 1898)]
year_semantics_valid = actual_year_ranges == expected_year_ranges
record_check('year-to-index fold semantics', int(year_semantics_valid), '== 1', year_semantics_valid)

toy_train, toy_future = np.array([1.0, 2.0, 3.0]), np.array([1000.0])
toy_mean, toy_scale = fit_standardizer(toy_train)
standardizer_error = max(abs(toy_mean - 2.0), abs(toy_scale - np.sqrt(2.0 / 3.0)))
record_check('training-only standardisation check', standardizer_error, '< 1e-12', standardizer_error < 1e-12)

scale = 7.3
shift = 42.0
mean_y, cov_y, y_y = shift + scale * our_mean, scale ** 2 * our_cov, shift + scale * y_test
score_y_direct = joint_nlpd(y_y, mean_y, cov_y)
score_y_identity = our_score + np.log(scale)
units_error = abs(score_y_direct - score_y_identity)
record_check('original-unit JNLPD identity', units_error, '< 1e-10', units_error < 1e-10)
assert len(E1_RESULTS) == 21
print(json.dumps({'checks': len(E1_RESULTS), 'passed': all(r['passed'] for r in E1_RESULTS)}, indent=2))
