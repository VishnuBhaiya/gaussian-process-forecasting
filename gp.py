"""Exact locally periodic GP implementation exported from the research notebook."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky, solve_triangular
from scipy.optimize import minimize
from scipy.stats import norm
SEED = 20260920
E1_SEED = 20260921
BOOTSTRAP_SEED = 20260922
BOOTSTRAP_RESAMPLES = 10_000
SCIENTIFIC_SEEDS = tuple(range(50))
ENGINEERING_SEEDS = (999101, 999102, 999103)
JITTER = 1e-8
PARAMETER_NAMES = ('sigma_f', 'ell_d', 'ell_p', 'period', 'sigma_n')
BOUNDS = np.array([[0.05, 5.0], [5.0, 500.0], [0.1, 5.0], [8.0, 14.0], [0.01, 2.0]])
LOG_BOUNDS = np.log(BOUNDS)
FAILURE_PENALTY = 1e12
OPTIMIZER_CONFIG = {
    'method': 'L-BFGS-B', 'maxiter': 500, 'maxfun': 5000,
    'ftol': 1e-9, 'gtol': 1e-6, 'cv_jac': '3-point',
    'finite_diff_rel_step': 1e-5, 'continuation_multiplier': 2,
}
# Literal, frozen 8x5 matrix: 5%-95% log-space Latin hypercube, seed 20260920.
STARTS_THETA = np.array([
    [0.48879528, 134.29635454, 0.57492580, 11.00940152, 0.88733950],
    [2.64964466,   7.76278442, 3.68874093, 12.91183230, 0.19192450],
    [0.15531736, 350.23716396, 0.25072509, 11.69159841, 0.03157694],
    [0.18474810,  79.42676569, 0.12626689,  8.37650750, 0.08411781],
    [0.79902856,  10.85979860, 1.08940965,  9.65241411, 0.02319772],
    [0.06306897, 212.30869876, 0.43638883, 10.36895138, 0.73715488],
    [1.35064883,  48.77297737, 2.55673145,  9.06419673, 0.07102522],
    [1.45346138,  24.41659601, 1.20994163, 12.42047826, 0.38035871],
])
STARTS_LOG_THETA = np.log(STARTS_THETA)
assert np.all((STARTS_THETA > BOUNDS[:, 0]) & (STARTS_THETA < BOUNDS[:, 1]))


def as_time_vector(t):
    return np.asarray(t, dtype=float).reshape(-1)

def unpack_log_theta(log_theta):
    log_theta = np.asarray(log_theta, dtype=float)
    if log_theta.shape != (5,):
        raise ValueError('log_theta must contain five log-parameters')
    return np.exp(log_theta)

def locally_periodic_kernel(t1, t2, theta):
    sigma_f, ell_d, ell_p, period, _ = np.asarray(theta, dtype=float)
    t1, t2 = as_time_vector(t1), as_time_vector(t2)
    d = t1[:, None] - t2[None, :]
    u = np.pi * d / period
    exponent = -(d ** 2) / (2.0 * ell_d ** 2) - 2.0 * np.sin(u) ** 2 / ell_p ** 2
    return sigma_f ** 2 * np.exp(exponent)

def kernel_log_derivatives(t, theta):
    sigma_f, ell_d, ell_p, period, sigma_n = np.asarray(theta, dtype=float)
    t = as_time_vector(t)
    d = t[:, None] - t[None, :]
    u = np.pi * d / period
    Kf = locally_periodic_kernel(t, t, theta)
    I = np.eye(t.size)
    return np.stack([
        2.0 * Kf,
        Kf * d ** 2 / ell_d ** 2,
        Kf * 4.0 * np.sin(u) ** 2 / ell_p ** 2,
        Kf * 2.0 * u * np.sin(2.0 * u) / ell_p ** 2,
        2.0 * sigma_n ** 2 * I,
    ], axis=0)

def training_covariance(t, theta, jitter=JITTER):
    sigma_n = float(theta[4])
    return locally_periodic_kernel(t, t, theta) + (sigma_n ** 2 + jitter) * np.eye(len(as_time_vector(t)))

def factorize_gp(t, y, theta, jitter=JITTER):
    t, y = as_time_vector(t), np.asarray(y, dtype=float).reshape(-1)
    if t.size != y.size:
        raise ValueError('t and y must have the same length')
    Ky = training_covariance(t, theta, jitter)
    factor = cho_factor(Ky, lower=True, check_finite=True)
    alpha = cho_solve(factor, y, check_finite=True)
    return Ky, factor, alpha

def gp_predict(t_train, y_train, t_test, theta, observed=True, jitter=JITTER):
    t_train, t_test = as_time_vector(t_train), as_time_vector(t_test)
    _, factor, alpha = factorize_gp(t_train, y_train, theta, jitter)
    Kxs = locally_periodic_kernel(t_test, t_train, theta)
    mean = Kxs @ alpha
    solved = cho_solve(factor, Kxs.T, check_finite=True)
    covariance = locally_periodic_kernel(t_test, t_test, theta) - Kxs @ solved
    covariance = 0.5 * (covariance + covariance.T)
    if observed:
        covariance = covariance + float(theta[4]) ** 2 * np.eye(t_test.size)
    return mean, covariance

def negative_lml_and_grad(log_theta, t, y, jitter=JITTER):
    theta = unpack_log_theta(log_theta)
    t, y = as_time_vector(t), np.asarray(y, dtype=float).reshape(-1)
    Ky, factor, alpha = factorize_gp(t, y, theta, jitter)
    L = np.tril(factor[0])
    log_det = 2.0 * np.log(np.diag(L)).sum()
    nll = 0.5 * y @ alpha + 0.5 * log_det + 0.5 * y.size * np.log(2.0 * np.pi)
    derivatives = kernel_log_derivatives(t, theta)
    quadratic_terms = np.array([alpha @ derivative @ alpha for derivative in derivatives])
    trace_terms = np.array([np.trace(cho_solve(factor, derivative, check_finite=True))
                            for derivative in derivatives])
    grad = 0.5 * (trace_terms - quadratic_terms)  # gradient of negative LML
    return float(nll), grad

def joint_nlpd(y, mean, covariance, per_observation=True):
    y = np.asarray(y, dtype=float).reshape(-1)
    mean = np.asarray(mean, dtype=float).reshape(-1)
    covariance = np.asarray(covariance, dtype=float)
    if y.shape != mean.shape or covariance.shape != (y.size, y.size):
        raise ValueError('Incompatible joint-density shapes')
    L = cholesky(covariance, lower=True, check_finite=True)
    residual_whitened = solve_triangular(L, y - mean, lower=True, check_finite=True)
    value = 0.5 * (residual_whitened @ residual_whitened)
    value += np.log(np.diag(L)).sum() + 0.5 * y.size * np.log(2.0 * np.pi)
    return float(value / y.size if per_observation else value)
def fit_standardizer(y_train):
    y_train = np.asarray(y_train, dtype=float)
    mean, scale = float(y_train.mean()), float(y_train.std(ddof=0))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError('Training scale must be positive and finite')
    return mean, scale

def standardize(y, mean, scale):
    return (np.asarray(y, dtype=float) - mean) / scale

def unstandardize_prediction(mean_z, covariance_z, mean_y, scale_y):
    return mean_y + scale_y * np.asarray(mean_z), scale_y ** 2 * np.asarray(covariance_z)

def inner_cv_blocks(outer_origin_index):
    T = int(outer_origin_index)
    validation = [np.arange(T - 44, T - 33), np.arange(T - 33, T - 22),
                  np.arange(T - 22, T - 11), np.arange(T - 11, T)]
    folds = []
    for valid in validation:
        train = np.arange(0, int(valid[0]))
        if train.size == 0 or train[-1] >= valid[0]:
            raise ValueError('Invalid expanding-window fold')
        folds.append((train, valid))
    return folds

SUNSPOT_FIRST_YEAR = 1700

def year_to_index(year, first_year=SUNSPOT_FIRST_YEAR):
    year = int(year)
    index = year - int(first_year)
    if index < 0:
        raise ValueError('year precedes the series')
    return index

def inner_cv_blocks_from_year(outer_start_year):
    return inner_cv_blocks(year_to_index(outer_start_year))

def blocked_cv_objective(log_theta, t, y, folds, standardize_target):
    # False for E2/E3; True for every real-data/sunspot fit.
    theta = unpack_log_theta(log_theta)
    t, y = as_time_vector(t), np.asarray(y, dtype=float).reshape(-1)
    scores = []
    try:
        for train_idx, valid_idx in folds:
            t_train, t_valid = t[train_idx], t[valid_idx]
            y_train, y_valid = y[train_idx], y[valid_idx]
            if standardize_target:
                train_mean, train_scale = fit_standardizer(y_train)
                z_train = standardize(y_train, train_mean, train_scale)
                mean_z, covariance_z = gp_predict(t_train, z_train, t_valid, theta, observed=True)
                mean_pred, covariance_pred = unstandardize_prediction(
                    mean_z, covariance_z, train_mean, train_scale)
            else:
                mean_pred, covariance_pred = gp_predict(
                    t_train, y_train, t_valid, theta, observed=True)
            scores.append(joint_nlpd(y_valid, mean_pred, covariance_pred))
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        return FAILURE_PENALTY
    value = float(np.mean(scores))
    return value if np.isfinite(value) else FAILURE_PENALTY

def _minimize_once(objective, x0, analytic_gradient, budget_multiplier=1):
    options = {
        'maxiter': OPTIMIZER_CONFIG['maxiter'] * budget_multiplier,
        'maxfun': OPTIMIZER_CONFIG['maxfun'] * budget_multiplier,
        'ftol': OPTIMIZER_CONFIG['ftol'], 'gtol': OPTIMIZER_CONFIG['gtol'],
    }
    jac = True if analytic_gradient else OPTIMIZER_CONFIG['cv_jac']
    if not analytic_gradient:
        options['finite_diff_rel_step'] = OPTIMIZER_CONFIG['finite_diff_rel_step']
    return minimize(objective, np.asarray(x0), method=OPTIMIZER_CONFIG['method'],
                    jac=jac, bounds=LOG_BOUNDS, options=options)

def _optimisation_record(result, start_id, stage):
    within_bounds = bool(np.all(result.x >= LOG_BOUNDS[:, 0]) and
                         np.all(result.x <= LOG_BOUNDS[:, 1]))
    admissible = bool(np.isfinite(result.fun) and result.fun < FAILURE_PENALTY and within_bounds)
    theta = np.exp(result.x)
    span = BOUNDS[:, 1] - BOUNDS[:, 0]
    boundary = (theta - BOUNDS[:, 0] <= 1e-4 * span) | (BOUNDS[:, 1] - theta <= 1e-4 * span)
    return {
        'start_id': int(start_id), 'stage': stage, 'objective': float(result.fun),
        'success': bool(result.success), 'admissible': admissible,
        'message': str(result.message), 'nit': int(result.nit), 'nfev': int(result.nfev),
        'log_theta': result.x.copy(), 'theta': theta, 'boundary_flags': boundary,
    }

def multistart_optimize(objective, analytic_gradient, starts=STARTS_LOG_THETA):
    initial_pairs, records = [], []
    for start_id, x0 in enumerate(np.asarray(starts)):
        try:
            result = _minimize_once(objective, x0, analytic_gradient)
            record = _optimisation_record(result, start_id, 'initial')
            initial_pairs.append((result, record)); records.append(record)
        except Exception as exc:
            records.append({'start_id': start_id, 'stage': 'initial', 'objective': np.inf,
                            'success': False, 'admissible': False, 'message': repr(exc),
                            'nit': 0, 'nfev': 0, 'log_theta': np.full(5, np.nan),
                            'theta': np.full(5, np.nan), 'boundary_flags': np.zeros(5, dtype=bool)})
    converged = [(r, rec) for r, rec in initial_pairs if rec['admissible'] and rec['success']]
    best_converged_value = min((rec['objective'] for _, rec in converged), default=np.inf)
    nonconverged = [(r, rec) for r, rec in initial_pairs
                    if rec['admissible'] and not rec['success']]
    if nonconverged:
        candidate, candidate_record = min(nonconverged, key=lambda item: item[1]['objective'])
        if candidate_record['objective'] < best_converged_value:
            try:
                continued = _minimize_once(objective, candidate.x, analytic_gradient,
                                           OPTIMIZER_CONFIG['continuation_multiplier'])
                continued_record = _optimisation_record(continued, candidate_record['start_id'], 'continuation')
                records.append(continued_record)
                if continued_record['admissible'] and continued_record['success']:
                    converged.append((continued, continued_record))
            except Exception as exc:
                records.append({'start_id': candidate_record['start_id'], 'stage': 'continuation',
                                'objective': np.inf, 'success': False, 'admissible': False,
                                'message': repr(exc), 'nit': 0, 'nfev': 0,
                                'log_theta': np.full(5, np.nan), 'theta': np.full(5, np.nan),
                                'boundary_flags': np.zeros(5, dtype=bool)})
    if not converged:
        return {'success': False, 'result': None, 'theta': None, 'records': records}
    best_result, best_record = min(converged, key=lambda item: item[1]['objective'])
    return {'success': True, 'result': best_result, 'theta': np.exp(best_result.x),
            'records': records, 'best_record': best_record}

def seasonal_naive_predictive(train_years, train_y, test_years, lag=11):
    train_years = np.asarray(train_years, dtype=int)
    train_y = np.asarray(train_y, dtype=float)
    test_years = np.asarray(test_years, dtype=int)
    if train_y.size <= lag:
        raise ValueError('Insufficient history for lag-11 baseline')
    residuals = train_y[lag:] - train_y[:-lag]
    variance = float(np.mean(residuals ** 2))  # frozen zero-mean residual MLE
    lookup = dict(zip(train_years.tolist(), train_y.tolist()))
    try:
        mean = np.array([lookup[int(year - lag)] for year in test_years])
    except KeyError as exc:
        raise ValueError('A required lagged target is unavailable') from exc
    return mean, variance * np.eye(test_years.size), variance

def absolute_log_parameter_error(theta_hat, theta_true):
    theta_hat, theta_true = np.asarray(theta_hat, dtype=float), np.asarray(theta_true, dtype=float)
    if np.any(theta_hat <= 0.0) or np.any(theta_true <= 0.0):
        raise ValueError('Log-parameter recovery requires positive parameters')
    return np.abs(np.log(theta_hat / theta_true))

def paired_bootstrap_mean_interval(deltas, confidence=0.95,
                                   seed=BOOTSTRAP_SEED, resamples=BOOTSTRAP_RESAMPLES):
    deltas = np.asarray(deltas, dtype=float)
    deltas = deltas[np.isfinite(deltas)]
    if deltas.size == 0:
        raise ValueError('No successful paired deltas')
    rng = np.random.default_rng(seed)
    sampled = rng.choice(deltas, size=(resamples, deltas.size), replace=True).mean(axis=1)
    alpha = 1.0 - confidence
    return tuple(np.quantile(sampled, [alpha / 2.0, 1.0 - alpha / 2.0]))

def cv_win_summary(deltas, confidence=0.95):
    deltas = np.asarray(deltas, dtype=float)
    deltas = deltas[np.isfinite(deltas)]
    n = int(deltas.size)
    if n == 0:
        raise ValueError('No successful paired deltas')
    wins, ties = int(np.sum(deltas < 0.0)), int(np.sum(deltas == 0.0))
    proportion = wins / n
    z = float(norm.ppf(0.5 + confidence / 2.0))
    denominator = 1.0 + z ** 2 / n
    centre = (proportion + z ** 2 / (2.0 * n)) / denominator
    half_width = z * np.sqrt(proportion * (1.0 - proportion) / n + z ** 2 / (4.0 * n ** 2)) / denominator
    return {'successful_pairs': n, 'cv_wins': wins, 'exact_ties': ties,
            'cv_win_proportion': proportion, 'wilson_low': centre - half_width,
            'wilson_high': centre + half_width}
