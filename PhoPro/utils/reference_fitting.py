from typing import Any, Callable
from numpy.lib.stride_tricks import sliding_window_view
from scipy.optimize import least_squares
from statsmodels.nonparametric.smoothers_lowess import lowess

from .equations import neg_bi_exponential_5

import numpy as np
import statsmodels.api as sm

###########################
#region --- PB. FITTING ---
###########################
def strided_median(signal, window, stride=None):
    '''Compute overlapping median windows with optional stride (downsampling).'''
    if stride is None:
        stride = window // 2

    # create sliding windows
    windows = sliding_window_view(signal, window_shape=window)
    # subsample windows with stride
    windows = windows[::stride]

    # compute median and centers per window
    medians = np.median(windows, axis=1)
    centers = np.arange(window//2, window//2 + len(medians)*stride, stride)

    return medians, centers

def mad_std(x):
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))

def positive_robust_scale(residuals: np.ndarray, signal: np.ndarray) -> float:
    scale = mad_std(residuals)

    if np.isfinite(scale) and scale > 0:
        return float(scale)

    scale = np.nanstd(residuals)
    if np.isfinite(scale) and scale > 0:
        return float(scale)

    signal_scale = max(
        np.nanstd(signal),
        np.ptp(signal),
        np.nanmax(np.abs(signal)),
        1.0,
    )

    return float(signal_scale * 1e-6)

def fit_photobleaching(
        signal: np.ndarray,
        time: np.ndarray,
        window: int,
        stride: int | None = None,
        ) -> tuple[np.ndarray, list[float]]:
    '''Fit a negative bi-exponential photobleaching trend to signal'''
    # reduce signal with strided_median
    sig_reduced, t_idxs = strided_median(
        signal=signal,
        window=window,
        stride=stride,
    )
    time_reduced = time[t_idxs]

    # define residuals equation
    def residuals(params, x, y) -> np.ndarray:
        return y - neg_bi_exponential_5(x, *params)
    
    # rough LOWESS baseline
    bleach0 = lowess(sig_reduced, time_reduced, frac=0.05, it=3, return_sorted=False)
    # robust residual scale
    sigma0 = positive_robust_scale(sig_reduced - bleach0, sig_reduced)

    # construct bounds
    T = time.max() - time.min()
    ymin, ymax = signal.min(), signal.max()
    yrange = ymax - ymin

    bounds = (
        [0, T/1000, 0, T/100, ymin - yrange],
        [5*yrange, 5*T, 5*yrange, 10*T, ymax + yrange]
    )

    # initial parameters
    dt = time_reduced[-1] - time_reduced[0]
    init_guess = [
        sig_reduced[0] - sig_reduced[-1], # a1
        dt / 10, # tau1
        (sig_reduced[0] - sig_reduced[-1]) / 2, # a2
        dt, # tau2
        sig_reduced[-1], # c
    ]

    # clip initial guesses to bounds
    for i, (guess, lower, upper) in enumerate(zip(init_guess, bounds[0], bounds[1])):
        init_guess[i] = float(np.clip(guess, lower, upper))

    res = least_squares(
        residuals,
        args=(time_reduced, sig_reduced),
        x0=init_guess,
        bounds=bounds,
        loss='soft_l1',
        f_scale=sigma0,
    )

    # generate full bleaching curve
    fitted_params = res.x
    bleach_curve: np.ndarray = neg_bi_exponential_5(time, *fitted_params)

    return bleach_curve, fitted_params

#endregion


#############################
#region --- ISOS. FITTING ---
#############################
def _process_arr(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype=np.float64, copy=False).ravel()

def OLS_fit(
        signal: np.ndarray,
        isosbestic: np.ndarray,
        add_intercept: bool = True,
        ) -> tuple[np.ndarray, Any]:
    sig = _process_arr(signal)
    iso = _process_arr(isosbestic)
    if add_intercept:
        iso = sm.add_constant(iso)

    model = sm.OLS(endog=sig, exog=iso)
    res = model.fit()
    fitted_iso = res.fittedvalues.astype(np.float32, copy=False)
    return fitted_iso, res.params

def IRLS_fit(
        signal: np.ndarray,
        isosbestic: np.ndarray,
        maxiter: int,
        c: float,
        add_intercept: bool = True,
        ) -> tuple[np.ndarray, Any]:
    sig = _process_arr(signal)
    iso = _process_arr(isosbestic)
    if add_intercept:
        iso = sm.add_constant(iso)

    model = sm.RLM(endog=sig, exog=iso, M=sm.robust.norms.TukeyBiweight(c=c))
    res = model.fit(maxiter=maxiter)
    fitted_iso = res.fittedvalues.astype(np.float32, copy=False)
    return fitted_iso, res.params

def windowed_OLS_fit(
        signal: np.ndarray,
        isosbestic: np.ndarray,
        n_windows: int,
    ) -> tuple[np.ndarray, Any]:

    sig_windows = np.array_split(signal, n_windows)
    iso_windows = np.array_split(isosbestic, n_windows)

    fitted_windows = []
    fitted_params = []
    for sig, iso in zip(sig_windows, iso_windows):
        fit, params = OLS_fit(sig, iso)
        fitted_windows.append(fit)
        fitted_params.append(params)

    fitted_iso = np.concat(fitted_windows)
    return fitted_iso, fitted_params

def detrend_retrend(
        signal: np.ndarray,
        isosbestic: np.ndarray,
        time: np.ndarray,
        window: int,
        stride: int | None = None,
        ) -> tuple[np.ndarray, Any]:

    bleach_exp, params_exp = fit_photobleaching(signal, time, window, stride)
    bleach_iso, params_iso = fit_photobleaching(isosbestic, time, window, stride)

    dBB_iso = (isosbestic - bleach_iso) / bleach_iso
    fitted_iso = dBB_iso * bleach_exp + bleach_exp
    return fitted_iso, params_iso + params_exp

#endregion
