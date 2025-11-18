import numpy as np
import os

def reconstruct_time_points(bounds: tuple, freq: float) -> np.ndarray:
    """
    Reconstruct a uniform time axis from bounds and sampling frequency.
    Args:
        bounds (tuple): A 2-tuple ``(t_low, t_high)`` i.e. start and end times.
        freq (float): Sampling frequency in Hz.
    Returns:
        np.ndarray: One-dimensional array of time points
    """
    tlow, thigh = bounds
    target_len = np.floor((thigh - tlow) * freq).astype(int)
    tp = np.arange(tlow, thigh, step=(1/freq))[:target_len]
    return tp

def downsample_ndarray(arr: np.ndarray, factor: int, axis: int = 1) -> np.ndarray:
    """
    Downsample higher-dimensional array by mean pooling.
    Args:
        arr (np.ndarray): Input array to downsample.
        factor (int): Integer downsampling factor.
        axis (int, optional): Axis along which to downsample. Defaults to 1.
    Returns:
        np.ndarray: Downsampled array.
    """
    if factor in (None, 1):
        return arr
    arr = np.asarray(arr)
    L = arr.shape[axis]
    trim = L % factor
    if trim:
        slicer = [slice(None)] * arr.ndim
        slicer[axis] = slice(0, L - trim)
        arr = arr[tuple(slicer)]
        L = arr.shape[axis]
    # reshape to (..., new_len, factor, ...) and average over the factor axis
    new_shape = list(arr.shape)
    new_shape[axis] = L // factor
    new_shape.insert(axis + 1, factor)
    arr = arr.reshape(new_shape).mean(axis=axis + 1)
    return arr

def downsample_1d(arr, factor):
    """
    Downsample a 1D array by an integer factor using mean pooling with trimming.
    Args:
        arr (array-like): One-dimensional array to downsample.
        factor (int): Integer downsampling factor.
    Returns:
        np.ndarray: One-dimensional array of mean-pooled values.
    """
    arr = np.asarray(arr)
    L = len(arr)
    trim = L % factor
    if trim:
        arr = arr[:L - trim]
    return arr.reshape(-1, factor).mean(axis=1)

def neg_exponential_3(x, a, b, c):
    """
    Negative single exponential for photobleaching curve fitting.
    Args:
        x (array-like): Independent variable (e.g., time).
        a (float): Amplitude of the exponential component.
        b (float): Decay rate of the exponential component.
        c (float): Constant offset term.
    Returns:
        np.ndarray: Evaluated exponential curve with the same shape as ``x``.
    """
    return a * np.exp(-b * x) + c

def neg_bi_exponential_5(x, a1, b1, a2, b2, c):
    """
    Negative bi-exponential for photobleaching curve fitting.
    Args:
        x (array-like): Independent variable (e.g., time).
        a1 (float): Amplitude of the fast exponential component.
        b1 (float): Decay rate of the fast component.
        a2 (float): Amplitude of the slow exponential component.
        b2 (float): Decay rate of the slow component.
        c (float): Constant offset term.
    Returns:
        np.ndarray: Evaluated bi-exponential curve with the same shape as ``x``.
    """
    return a1 * np.exp(-b1 * x) + a2 * np.exp(-b2 * x) + c

def sem(arr, axis=0):
    """
    Compute the standard error of the mean along a given axis.
    Args:
        arr (array-like): Input data.
        axis (int, optional): Axis along which to compute the SEM. Defaults to 0.
    Returns:
        np.ndarray or float: Standard error of the mean along the specified axis.
    """
    n = arr.shape[axis]
    std = np.std(arr, axis=axis)
    return std / np.sqrt(n)

def zscore_signal(signal: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """
    Compute trial-wise z-scored signal using baseline mean and std.
    Args:
        signal (np.ndarray): Trial signal windows of shape (n_trials, n_time).
        baseline (np.ndarray): Baseline windows of shape (n_trials, n_time).
    Returns:
        np.ndarray: Z-scored signal windows.
    """
    base_mean = baseline.mean(axis=1, keepdims=True)
    base_std = baseline.std(axis=1, ddof=0, keepdims=True)
    base_std = np.where(base_std == 0.0, np.finfo(baseline.dtype).eps, base_std)
    return (signal - base_mean) / base_std

def center_signal(signal: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """
    Center trial-wise signal by subtracting the baseline mean.
    Args:
        signal (np.ndarray): Trial signal windows of shape (n_trials, n_time).
        baseline (np.ndarray): Baseline windows of shape (n_trials, n_time).
    Returns:
        np.ndarray: Mean-centered signal windows.
    """
    base_mean = baseline.mean(axis=1, keepdims=True)
    return (signal - base_mean)