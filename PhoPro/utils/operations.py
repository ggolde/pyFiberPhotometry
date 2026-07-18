import numpy as np

from numpy.lib.array_utils import normalize_axis_index
from scipy.signal import resample_poly

from ..types import DownsampleMethods

##########################
#region --- DOWNSAMPLE ---
##########################

def downsample_signal(
        arr: np.ndarray,
        factor: int | None,
        axis: int = -1,
        method: DownsampleMethods = 'resample',
        upsample: int = 1, 
        window = ('kaiser', 5),
        padtype: str = 'line',
        cval: float | None = None,
        ) -> np.ndarray:
    # no downsampling case
    if factor in (None, 0, 1, 0.0, 1.0):
        return arr
    
    # normalize inputs
    arr = np.asarray(arr)
    factor = int(factor)
    axis = normalize_axis_index(axis, ndim=arr.ndim)
    
    # perform method
    match method:
        case 'mean':
            out = downsample_mean(arr, factor, axis)
        case 'resample':
            out = downsample_resample(
                arr, factor, axis, 
                upsample=upsample, window=window,
                padtype=padtype, cval=cval,
            )
        case _:
            raise ValueError(f'Downsampling method {method} not recognized.')
        
    return out

def downsample_time(
        time: np.ndarray,
        factor: int | None,
        method: DownsampleMethods = 'resample',
        upsample: int = 1, 
        **kwargs
        ) -> np.ndarray:
    # no downsampling case
    if factor in (None, 0, 1, 0.0, 1.0):
        return time
    
    # normalize inputs
    time = np.asarray(time)
    factor = int(factor)
    axis = 0
    
    # execute method
    match method:
        case 'mean':
            out = downsample_mean(time, factor, axis)
        case 'resample':
            out = downsample_time_resample(time, factor, upsample=upsample)
        case _:
            raise ValueError(f'Downsampling method {method} not recognized.')
    
    return out

def downsample_mean(
        arr: np.ndarray,
        factor: int | None,
        axis: int = -1,
        ) -> np.ndarray:
    n_samples = arr.shape[axis]
    n_output = n_samples // factor

    if n_output == 0:
        raise ValueError(
            f"factor ({factor}) exceeds the number of samples "
            f"along axis {axis} ({n_samples})"
        )

    # discard trailing block.
    n_used = n_output * factor
    index = [slice(None)] * arr.ndim
    index[axis] = slice(0, n_used)
    arr = arr[tuple(index)]

    # split the selected axis into `(n_output, factor)`
    new_shape = list(arr.shape)
    new_shape[axis] = n_output
    new_shape.insert(axis + 1, factor)

    # average each block
    return arr.reshape(new_shape).mean(axis=axis + 1)

def downsample_resample(
        arr: np.ndarray, 
        factor: int, 
        axis: int = 0, 
        upsample: int = 1, 
        window = ('kaiser', 5),
        padtype: str = 'line',
        cval: float | None = None,
        ) -> np.ndarray:
    return resample_poly(
        arr,
        up=upsample,
        down=factor,
        axis=axis,
        window=window,
        padtype=padtype,
        cval=cval,
    )

def downsample_time_resample(
        time: np.ndarray,
        factor: int,
        upsample: int = 1,
        ) -> np.ndarray:
    intervals = np.diff(time)
    dt = np.median(intervals)

    if not np.allclose(intervals, dt, rtol=1e-5, atol=1e-8):
        raise ValueError("FIR downsampling requires regularly spaced timestamps")

    # resample_poly returns ceil(n_samples / factor) samples for up=1.
    n_output = (time.size * upsample + factor - 1) // factor
    return time[0] + np.arange(n_output) * factor * dt

#endregion

###############################
#region --- TRIAL-WISE NORM ---
###############################

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

def mad_norm_signal(signal: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """
    Compute trial-wise MAD normalized signal.
    Args:
        signal (np.ndarray): Trial signal windows of shape (n_trials, n_time).
        baseline (np.ndarray): Baseline windows of shape (n_trials, n_time).
    Returns:
        np.ndarray: MAD normalized signal windows.
    """
    median = np.median(baseline, axis=1, keepdims=True)
    mad = np.median(np.abs(baseline - median), axis=1, keepdims=True)
    return (signal - median) / (1.4826 * mad)

def amp_norm_signal(signal: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """
    Compute trial-wise amplitude normalized signal.
    Args:
        signal (np.ndarray): Trial signal windows of shape (n_trials, n_time).
        baseline (np.ndarray): Baseline windows of shape (n_trials, n_time).
    Returns:
        np.ndarray: Amplitude (scaled to 1) normalized signal windows.
    """
    zerod_sig = center_signal(signal, baseline)
    return (zerod_sig / np.max(np.abs(zerod_sig), axis=1, keepdims=True))

#endregion
