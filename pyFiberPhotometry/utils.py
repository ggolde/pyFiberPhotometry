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

def write_dict_to_txt(d: dict, filename: str, indent: int = 0) -> None:
    """
    Write a nested dictionary to a human-readable, indented text file.
    Args:
        d (dict): Dictionary to serialize. Nested dictionaries are supported.
        filename (str): Path to the output text file.
        indent (int, optional): Initial indentation level (in tabs).
    Returns:
        None
    """
    def _write_dict(fd, obj, lvl):
        for k, v in obj.items():
            if isinstance(v, dict):
                fd.write("\t" * lvl + f"{k}:\n")
                _write_dict(fd, v, lvl + 1)
            else:
                fd.write("\t" * lvl + f"{k}: {v}\n")
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    with open(filename, "w") as f:
        _write_dict(f, d, indent)

def read_txt_to_dict(filename: str) -> dict:
    """
    Read a nested dictionary from a text file created by ``write_dict_to_txt``.
    Args:
        filename (str): Path to the input text file.
    Returns:
        dict: Nested dictionary reconstructed from the file contents
    """
    with open(filename, "r") as f:
        lines = f.readlines()

    def parse(lines, start=0, level=0):
        out = {}
        i = start
        while i < len(lines):
            raw = lines[i]
            curr = len(raw) - len(raw.lstrip("\t"))
            if curr < level:
                break
            s = raw.strip()
            i += 1
            if not s:
                continue
            if s.endswith(":"):  # nested block
                key = s[:-1]
                sub, nxt = parse(lines, i, level + 1)
                out[key] = sub
                i = nxt
            else:
                if ":" in s:
                    k, v = s.split(":", 1)
                    out[k.strip()] = v.strip()
        return out, i

    d, _ = parse(lines, 0, 0)
    return d

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