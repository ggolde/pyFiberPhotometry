"""Peak detection and peak metric extraction utilities."""

from __future__ import annotations
from typing import Literal, Callable, Any, ClassVar, Self
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from ..types import PeakCenterMethod, PeakDirection, PeakScaleMethod

#######################
#region --- UTILITY ---
#######################
def nanmad(x: np.ndarray, *, axis: int):
    """Calculate median absolute deviation while ignoring NaNs.

    Parameters
    ----------
    x : np.ndarray
        Input values.
    axis : int
        Axis along which to calculate the statistic.

    Returns
    -------
    np.ndarray
        NaN-robust MAD values scaled by ``1.4826``.
    """
    med = np.nanmedian(x, axis=axis, keepdims=True)
    return 1.4826 * np.nanmedian(np.abs(x - med), axis=axis)

def _shape_without_axes(shape: tuple[int, ...], axis: int | tuple[int, ...]) -> tuple[int, ...]:
    """Return the shape that remains after removing one or more axes."""
    axes = (axis,) if isinstance(axis, int) else axis
    axes = tuple(ax % len(shape) for ax in axes)
    return tuple(size for i, size in enumerate(shape) if i not in axes)

def one_scale(x: np.ndarray, *, axis: int):
    """Return ones with the input shape reduced over ``axis``.

    Parameters
    ----------
    x : np.ndarray
        Input values.
    axis : int
        Axis removed from the output shape.

    Returns
    -------
    np.ndarray
        Array of ones with the reduced shape.
    """
    return np.ones(shape=_shape_without_axes(x.shape, axis), dtype=float)

def zero_center(x: np.ndarray, *, axis: int):
    """Return zeros with the input shape reduced over ``axis``.

    Parameters
    ----------
    x : np.ndarray
        Input values.
    axis : int
        Axis removed from the output shape.

    Returns
    -------
    np.ndarray
        Array of zeros with the reduced shape.
    """
    return np.zeros(shape=_shape_without_axes(x.shape, axis), dtype=float)

#endregion

#############################
#region --- RESULT CLASS  ---
#############################
@dataclass
class PeakResult:
    """Container for a table of detected peak metrics."""

    df: pd.DataFrame

    # --- dunders ---
    def __post_init__(self) -> None:
        """Validate and sort the peak table."""
        if 'trial_idx' not in self.df:
            raise ValueError(f'PeakResult class data must contain the column "trial_idx"')
        if not self.empty:
            self.df = self.df.sort_values(['trial_idx', 'start_idx']).copy()

    def __str__(self) -> str:
        """Return the string representation of the peak table."""
        return self.df.__str__()

    def __repr__(self) -> str:
        """Return the interactive representation of the peak table."""
        return self.df.__repr__()

    # --- properties ---
    @property
    def empty(self) -> bool:
        """Whether no peaks were detected."""
        return self.df.shape[0] == 0

    @property
    def metrics(self) -> list[str]:
        """Metric columns present in the result table."""
        return self.df.columns.to_list()

    @property
    def n_peaks(self) -> int:
        """Number of detected peaks."""
        return int(self.df.shape[0])

    @property
    def one_peak_per_trial(self) -> bool:
        """Whether each trial represented in the table has exactly one peak."""
        return self.n_peaks == self.df['trial_idx'].nunique()

    # --- operations ---
    def filter(
            self,
            on: str,
            how: Literal['lowest', 'highest', 'closest', 'between', 'equals'],
            value: None | str | float | tuple[float, float],
            ) -> Self:
        """Filter peaks globally or within each trial.

        Parameters
        ----------
        on : str
            Column used for filtering.
        how : {'lowest', 'highest', 'closest', 'between', 'equals'}
            Filtering strategy. ``'lowest'``, ``'highest'``, and
            ``'closest'`` select one row per trial; ``'between'`` and
            ``'equals'`` keep all matching rows.
        value : None, str, float, or tuple[float, float]
            Comparison value. ``'between'`` requires a two-value tuple;
            ``'closest'`` requires a scalar numeric value.

        Returns
        -------
        PeakResult
            Filtered peak result.

        Raises
        ------
        KeyError
            If ``on`` is not a column.
        ValueError
            If ``how`` is unknown or ``value`` is incompatible with ``how``.
        """
        df = self.df.copy()

        # validate inputs
        if on not in df:
            raise KeyError(f'{on} not found as a column')

        # execute filter
        match how:
            case 'lowest':
                idx = df[on].eq(df.groupby('trial_idx')[on].transform('min'))
                out = df[idx]
            case 'highest':
                idx = df[on].eq(df.groupby('trial_idx')[on].transform('max'))
                out = df[idx]
            case 'between':
                if not isinstance(value, tuple):
                    raise ValueError('value must be a tuple with how as "between"')
                if (value[0] > value[1]):
                    raise ValueError('value[0] must be <= value[1]')

                idx = (df[on] >= value[0]) & (df[on] <= value[1])
                out = df[idx]
            case 'closest':
                if isinstance(value, (tuple, str)) or (value is None):
                    raise ValueError(f'value must be scalar-like with how as "closest"')

                idx = (df[on] - value).abs().groupby(df["trial_idx"]).idxmin()
                out = df.loc[idx]
            case 'equals':
                idx = df[on] == value
                out = df[idx]
            case _:
                raise ValueError(f'how {how} not recognized')

        return type(self)(out)

    # --- I/O ---
    def to_csv(self, path: str) -> None:
        """Write peak metrics to CSV.

        Parameters
        ----------
        path : str
            Output CSV path.
        """
        self.df.to_csv(path, index=False)

    @classmethod
    def from_csv(cls, path: str) -> Self:
        """Read peak metrics from CSV.

        Parameters
        ----------
        path : str
            Input CSV path.

        Returns
        -------
        PeakResult
            Peak result loaded from disk.
        """
        return cls(pd.read_csv(path))

#endregion

############################
#region --- SINGLE PEAK  ---
############################
class SinglePeak:
    """Calculate metrics for one detected peak interval."""

    EXPORT_ATTRS: ClassVar[tuple[str, ...]] = (
        "trial_idx",
        "direction",
        "start_idx",
        "stop_idx",
        "start_time",
        "stop_time",
        "peak_baseline",
        "peak_idx",
        "peak_time",
        "peak_value",
        "height",
        "prominence",
        "duration",
        "width",
        "area",
    )

    EXPORT_ATTRS_FULL: ClassVar[tuple[str, ...]] = EXPORT_ATTRS + (
        "left_prominence",
        "right_prominence",
    )

    def __init__(
            self,
            trial_idx: int,
            bounds: tuple[int, int],
            signal: np.ndarray,
            time: np.ndarray,
            centers: np.ndarray = np.asarray(0.0),
            direction: Literal['positive', 'negative'] = 'positive',
            ) -> None:
        """Initialize one peak and calculate its metrics.

        Parameters
        ----------
        trial_idx : int
            Trial index containing this peak.
        bounds : tuple[int, int]
            Inclusive ``(start_idx, stop_idx)`` bounds for the peak.
        signal : np.ndarray
            One-dimensional signal trace for the trial.
        time : np.ndarray
            Time values aligned with ``signal``.
        centers : np.ndarray, default=np.asarray(0.0)
            Baseline or center trace aligned with ``signal``.
        direction : {'positive', 'negative'}, default='positive'
            Direction used to calculate signed peak metrics.
        """
        self.trial_idx = trial_idx
        self.direction = direction
        self.sign = 1 if direction == 'positive' else -1

        self.start_idx = int(bounds[0])
        self.stop_idx = int(bounds[1])
        self.start_time = time[self.start_idx]
        self.stop_time = time[self.stop_idx]

        y = signal[self.start_idx : self.stop_idx + 1]
        x = time[self.start_idx : self.stop_idx + 1]
        b = centers[self.start_idx : self.stop_idx + 1]

        self.calc_metrics(y, x, b)

    # --- preallocation helpers ---
    @classmethod
    def export_attrs(cls, full: bool = False) -> tuple[str, ...]:
        """Return attribute names exported for each peak.

        Parameters
        ----------
        full : bool, default=False
            If ``True``, include extra left and right prominence metrics.

        Returns
        -------
        tuple[str, ...]
            Exported attribute names.
        """
        return cls.EXPORT_ATTRS_FULL if full else cls.EXPORT_ATTRS

    @classmethod
    def n_export_attrs(cls, full: bool = False) -> int:
        """Return the number of exported attributes.

        Parameters
        ----------
        full : bool, default=False
            If ``True``, include extra left and right prominence metrics.

        Returns
        -------
        int
            Number of exported attributes.
        """
        return len(cls.export_attrs(full))

    @classmethod
    def export_columns(cls, full: bool = False) -> list[str]:
        """Return exported attribute names as a list.

        Parameters
        ----------
        full : bool, default=False
            If ``True``, include extra left and right prominence metrics.

        Returns
        -------
        list[str]
            Exported column names.
        """
        return list(cls.export_attrs(full))

    # --- metrics calculations ---
    def calc_metrics(self, y: np.ndarray, x: np.ndarray, b: np.ndarray) -> None:
        """Calculate and store metrics for this peak.

        Parameters
        ----------
        y : np.ndarray
            Peak-window signal values.
        x : np.ndarray
            Peak-window time values.
        b : np.ndarray
            Peak-window baseline values.
        """
        # transform
        ytrans = self.sign * (y - b)

        # peak location
        peak_local_idx = int(np.nanargmax(ytrans))
        self.peak_idx = self.start_idx + peak_local_idx
        self.peak_time = float(x[peak_local_idx])

        # vertical metrics
        self.peak_value = y[peak_local_idx]
        self.peak_baseline = b[peak_local_idx]
        self.height = self.sign * ytrans[peak_local_idx]

        self.left_prominence = self.height - self.sign * ytrans[0]
        self.right_prominence = self.height - self.sign * ytrans[-1]
        self.prominence = min(self.left_prominence, self.right_prominence)

        # horizontal
        self.duration = x[-1] - x[0]
        self.width = self.width_at_half_max(y, x, b, peak_local_idx)

        # area
        self.area = np.trapezoid(y=y - b, x=x)

    def width_at_half_max(self, y: np.ndarray, x: np.ndarray, b: np.ndarray, peak_local_idx: int) -> float:
        """Calculate peak width at half maximum.

        Parameters
        ----------
        y : np.ndarray
            Peak-window signal values.
        x : np.ndarray
            Peak-window time values.
        b : np.ndarray
            Peak-window baseline values.
        peak_local_idx : int
            Peak index relative to the peak window.

        Returns
        -------
        float
            Width between the left and right half-maximum crossings.
        """
        # transform
        ytrans = self.sign * (y - b)

        peak_height = ytrans[peak_local_idx]
        halfmax = b[peak_local_idx] + 0.5 * (peak_height - b[peak_local_idx])

        left_canidate = np.flatnonzero(ytrans[:peak_local_idx + 1] <= halfmax)
        left_idx = 0 if left_canidate.size == 0 else left_canidate[-1]

        right_canidate = np.flatnonzero(ytrans[peak_local_idx:] <= halfmax)
        right_idx = -1 if right_canidate.size == 0 else peak_local_idx + right_canidate[0]

        return (x[right_idx] - x[left_idx])

    # --- export functions ---
    def export_as_array(self, full: bool = False) -> np.ndarray:
        """Export peak metrics as an array.

        Parameters
        ----------
        full : bool, default=False
            If ``True``, include extra left and right prominence metrics.

        Returns
        -------
        np.ndarray
            One-dimensional array of exported metric values.
        """
        attrs = self.export_attrs(full)
        return np.array([getattr(self, attr) for attr in attrs])

    def export_into_array(self, target: np.ndarray, row: int, full: bool = False) -> None:
        """Write peak metrics into a preallocated array row.

        Parameters
        ----------
        target : np.ndarray
            Output array.
        row : int
            Row in ``target`` to write into.
        full : bool, default=False
            If ``True``, include extra left and right prominence metrics.
        """
        attrs = self.export_attrs(full)
        target[row, :] = [getattr(self, attr) for attr in attrs]

    def export_dict(self, full: bool = False) -> dict[str, Any]:
        """Export peak metrics as a dictionary.

        Parameters
        ----------
        full : bool, default=False
            If ``True``, include extra left and right prominence metrics.

        Returns
        -------
        dict[str, Any]
            Mapping from metric names to values.
        """
        attrs = self.export_attrs(full)
        return {attr : getattr(self, attr, pd.NA) for attr in attrs}

#endregion

##########################
#region --- PEAK MASK ---
##########################
class PeakMask:
    """Mutable two-dimensional boolean mask of detected peak samples."""

    # --- contruction ---
    def __init__(self, mask: np.ndarray) -> None:
        """Initialize a peak mask.

        Parameters
        ----------
        mask : np.ndarray
            Two-dimensional boolean-like array with shape
            ``(n_trials, n_times)``.
        """
        self.mask = self._validate_input(mask)

    def _validate_input(self, mask: np.ndarray) -> np.ndarray:
        """Validate and copy a peak mask input."""
        mask = np.asarray(mask, dtype=bool).copy()
        if mask.ndim != 2:
            raise ValueError(f'Mask must be 2D, has ndim {mask.ndim}')
        return mask

    # --- operations ---
    def merge_close(self, min_distance: int | None) -> None:
        """Merge peaks separated by short gaps.

        Parameters
        ----------
        min_distance : int or None
            Maximum gap length, in samples, to fill between adjacent peaks.
            ``None`` and ``0`` leave the mask unchanged.
        """
        if (min_distance == 0) or (min_distance is None): return

        n_rows, n_cols = self.mask.shape
        cols = np.arange(n_cols)

        # index of nearest True to the left
        left_true = np.where(self.mask, cols, -1)
        left_true = np.maximum.accumulate(left_true, axis=1)

        # index of nearest True to the right
        right_true = np.where(self.mask, cols, n_cols)
        right_true = np.minimum.accumulate(right_true[:, ::-1], axis=1)[:, ::-1]

        # is a gap between True's
        bounded_gap = (~self.mask) & (left_true >= 0) & (right_true < n_cols)

        # calc gap length
        gap_len = right_true - left_true - 1

        # fill
        fill = bounded_gap & (gap_len <= min_distance)
        self.mask = self.mask | fill

    def filter_size(self, min_size: int | None = None, max_size: int | None = None) -> None:
        """Remove peak runs outside size bounds.

        Parameters
        ----------
        min_size : int or None, default=None
            Minimum peak size in samples.
        max_size : int or None, default=None
            Maximum peak size in samples.
        """
        # label only row-wise neighbors
        structure = np.array([
            [0, 0, 0],
            [1, 1, 1],
            [0, 0, 0],
        ], dtype=bool)

        labels, n_labels = ndi.label(self.mask, structure=structure) # type: ignore
        if n_labels == 0: return

        # calculate sizes
        sizes = np.bincount(labels.ravel())
        keep = np.ones(n_labels + 1, dtype=bool)
        keep[0] = False

        # filter sizes
        if min_size is not None:
            keep &= sizes >= min_size

        if max_size is not None:
            keep &= sizes <= max_size

        self.mask = keep[labels]

    def get_peak_intervals(self) -> tuple[np.ndarray, np.ndarray]:
        """Return inclusive peak intervals from the mask.

        Returns
        -------
        trial_start : np.ndarray
            Trial index for each detected peak.
        peak_intervals : np.ndarray
            Integer array with shape ``(n_peaks, 2)`` containing inclusive
            ``start_idx`` and ``stop_idx`` bounds.

        Raises
        ------
        ValueError
            If detected start and stop edges do not belong to the same trials.
        """
        # detect bool change
        diff = np.diff(self.mask, axis=1, prepend=0, append=0)

        trial_start, start_idx = np.nonzero(diff == 1)
        trial_stop, stop_idx = np.nonzero(diff == -1)

        # falling edges are exclusive, subtract 1 to make inclusive
        stop_idx = stop_idx - 1

        if np.any(trial_start != trial_stop):
            raise ValueError(f'Trial starts and stops not equivalent')

        # coerce into shape (n_peaks, 2)
        peak_intervals = np.concat(
            [start_idx[:, np.newaxis], stop_idx[:, np.newaxis]], axis=1
        )

        return trial_start, peak_intervals

#endregion

#################################
#region --- ABSTRACT DETECTOR ---
#################################
class PeakDetector(ABC):
    """Base class for peak detectors."""

    # --- peak handling ---
    def handle_peaks_iterative(
            self,
            signals: np.ndarray,
            time: np.ndarray,
            centers: np.ndarray,
            trial_idxs: np.ndarray,
            intervals: np.ndarray,
            direction: Literal['positive', 'negative'] = 'positive',
            full_output: bool = False,
            ) -> pd.DataFrame:
        """Calculate peak metrics for detected intervals.

        Parameters
        ----------
        signals : np.ndarray
            Trial-by-time signal matrix.
        time : np.ndarray
            Time values for signal columns.
        centers : np.ndarray
            Trial-by-time baseline or center matrix.
        trial_idxs : np.ndarray
            Trial index for each peak interval.
        intervals : np.ndarray
            Inclusive peak intervals with shape ``(n_peaks, 2)``.
        direction : {'positive', 'negative'}, default='positive'
            Peak direction used for signed metrics.
        full_output : bool, default=False
            If ``True``, include extra peak metrics.

        Returns
        -------
        pd.DataFrame
            Table of peak metrics.
        """

        # iteratively calc peak information
        peak_info = []

        for trial_idx, bounds in zip(trial_idxs, intervals):
            peak = SinglePeak(
                trial_idx=trial_idx,
                bounds=bounds,
                signal=signals[trial_idx, :],
                time=time,
                centers=centers[trial_idx, :],
                direction=direction,
            )
            peak_info.append(peak.export_dict(full=full_output))

        # package and return
        peak_info = pd.DataFrame(peak_info)
        return peak_info

    def process_peak_mask(
            self,
            peak_mask: PeakMask,
            frequency: float,
            min_distance_sec: float | None,
            min_duration_sec: float | None,
            max_duration_sec: float | None,
            ) -> PeakMask:
        """Merge and size-filter a peak mask.

        Parameters
        ----------
        peak_mask : PeakMask
            Peak mask to process in place.
        frequency : float
            Sampling frequency in Hz.
        min_distance_sec : float or None
            Minimum distance between peaks, in seconds.
        min_duration_sec : float or None
            Minimum peak duration, in seconds.
        max_duration_sec : float or None
            Maximum peak duration, in seconds.

        Returns
        -------
        PeakMask
            Processed peak mask.
        """
        # merge close peaks
        min_distance = None if min_distance_sec is None else int(min_distance_sec * frequency)
        peak_mask.merge_close(min_distance)

        # filter peaks for size
        min_size = None if min_duration_sec is None else int(min_duration_sec * frequency)
        max_size = None if max_duration_sec is None else int(max_duration_sec * frequency)
        peak_mask.filter_size(min_size, max_size)
        return peak_mask

    # --- peak detection ---
    @abstractmethod
    def detect(
            self,
            signals: np.ndarray,
            time: np.ndarray,
            frequency: float,
            baselines: np.ndarray | None = None,
            min_distance_sec: float | None = None,
            min_duration_sec: float | None = None,
            max_duration_sec: float | None = None,
            direction: PeakDirection = 'both',
            detailed: bool = False,
            ) -> PeakResult:
        """Detect peaks in trial-by-time signals.

        Parameters
        ----------
        signals : np.ndarray
            Trial-by-time signal matrix.
        time : np.ndarray
            Time values for signal columns.
        frequency : float
            Sampling frequency in Hz.
        baselines : np.ndarray or None, default=None
            Optional baseline signals.
        min_distance_sec : float or None, default=None
            Minimum distance between peaks, in seconds.
        min_duration_sec : float or None, default=None
            Minimum peak duration, in seconds.
        max_duration_sec : float or None, default=None
            Maximum peak duration, in seconds.
        direction : PeakDirection, default='both'
            Direction of peaks to detect: ``'positive'`` detects peaks above
            baseline, ``'negative'`` detects peaks below baseline, and
            ``'both'`` detects both directions.
        detailed : bool, default=False
            If ``True``, include extra peak metrics.

        Returns
        -------
        PeakResult
            Detected peaks and metrics.
        """
        pass

#endregion

###################################
#region --- THRESHOLD DETECTORS ---
###################################
class ThresholdDetector(PeakDetector):
    """Base class for threshold-based peak detectors."""

    def __init__(
            self,
            center_method: PeakCenterMethod = 'median',
            scale_method: PeakScaleMethod = 'mad',
            test_magnitude: float = 3.0,
            ) -> None:
        """Initialize a threshold detector.

        Parameters
        ----------
        center_method : PeakCenterMethod, default='median'
            Center estimate: ``'median'`` uses ``np.nanmedian``, ``'mean'``
            uses ``np.nanmean``, and ``'zeros'`` uses a zero-valued baseline.
            A custom callable accepts an array and keyword argument ``axis``.
        scale_method : PeakScaleMethod, default='mad'
            Scale estimate: ``'mad'`` uses median absolute deviation,
            ``'std'`` uses ``np.nanstd``, and ``'ones'`` uses a one-valued
            scale. A custom callable accepts an array and keyword ``axis``.
        test_magnitude : float, default=3.0
            Scale multiplier used to set the detection threshold.
        """
        self.test_magnitude = test_magnitude
        self.center_func = self._resolve_center_method(center_method)
        self.scale_func = self._resolve_scale_method(scale_method)

    def _resolve_center_method(self, center_method: PeakCenterMethod) -> Callable:
        """Resolve a center method name or callable."""
        center_func: Callable
        match center_method:
            case func if callable(func):
                center_func = center_method
            case 'median':
                center_func = np.nanmedian
            case 'mean':
                center_func = np.nanmean
            case 'zeros':
                center_func = zero_center
            case _:
                raise ValueError(f'Center method ({center_method}) is not recognized')
        return center_func

    def _resolve_scale_method(self, scale_method: PeakScaleMethod) -> Callable:
        """Resolve a scale method name or callable."""
        scale_func: Callable
        match scale_method:
            case func if callable(func):
                scale_func = scale_method
            case 'mad':
                scale_func = nanmad
            case 'std':
                scale_func = np.nanstd
            case 'ones':
                scale_func = one_scale
            case _:
                raise ValueError(f'Scale method ({scale_method}) is not recognized')
        return scale_func

    def _coerce_to_rowwise_alignment(self, arr: np.ndarray, signals: np.ndarray) -> np.ndarray:
        """Coerce row-wise statistics to signal shape."""
        if arr.ndim == 1:
            arr = np.tile(arr[:, np.newaxis], reps=signals.shape[1])

        if arr.shape != signals.shape:
            raise ValueError(f'Array shape ({arr.shape}) does not match signals shape ({signals.shape})')

        return arr

    def _detect_from_threshold(
            self,
            centers: np.ndarray,
            scales: np.ndarray,
            signals: np.ndarray,
            time: np.ndarray,
            frequency: float,
            min_distance_sec: float | None = None,
            min_duration_sec: float | None = None,
            max_duration_sec: float | None = None,
            direction: PeakDirection = 'both',
            detailed: bool = False,
            ) -> PeakResult:
        """Detect positive and/or negative peaks from threshold arrays."""

        # detect peaks
        peak_info_wrapper: list[pd.DataFrame] = []

        # positive peak detection
        if direction in ['positive', 'both']:
            pos_threshold = centers + self.test_magnitude * scales
            peak_mask = PeakMask(signals > pos_threshold)

            # process peak mask
            peak_mask = self.process_peak_mask(
                peak_mask,
                frequency=frequency,
                min_distance_sec=min_distance_sec,
                min_duration_sec=min_duration_sec,
                max_duration_sec=max_duration_sec,
            )

            # calculate peak metrics and return
            trial_idxs, intervals = peak_mask.get_peak_intervals()
            pos_peak_info = self.handle_peaks_iterative(
                signals=signals,
                time=time,
                centers=centers,
                trial_idxs=trial_idxs,
                intervals=intervals,
                direction='positive',
                full_output=detailed,
            )

            peak_info_wrapper.append(pos_peak_info)

        # negative peak detection
        if direction in ['negative', 'both']:
            neg_threshold = centers - self.test_magnitude * scales
            peak_mask = PeakMask(signals < neg_threshold)

            # process peak mask
            peak_mask = self.process_peak_mask(
                peak_mask,
                frequency=frequency,
                min_distance_sec=min_distance_sec,
                min_duration_sec=min_duration_sec,
                max_duration_sec=max_duration_sec,
            )

            # calculate peak metrics and return
            trial_idxs, intervals = peak_mask.get_peak_intervals()
            neg_peak_info = self.handle_peaks_iterative(
                signals=signals,
                time=time,
                centers=centers,
                trial_idxs=trial_idxs,
                intervals=intervals,
                direction='negative',
                full_output=detailed,
            )

            peak_info_wrapper.append(neg_peak_info)

        if len(peak_info_wrapper) == 0:
            raise ValueError(f'Peak direction ({direction}) not recognized.')

        # combine results
        peak_info = PeakResult(pd.concat(peak_info_wrapper, axis=0, ignore_index=True))
        return peak_info

class StaticThresholdDetector(ThresholdDetector):
    """Detect peaks using one static threshold per trial."""

    def __init__(
            self,
            center_method: PeakCenterMethod = 'median',
            scale_method: PeakScaleMethod = 'mad',
            test_magnitude: float = 3.0,
            ) -> None:
        """Initialize a static threshold detector.

        Parameters
        ----------
        center_method : PeakCenterMethod, default='median'
            Center estimate: ``'median'`` uses ``np.nanmedian``, ``'mean'``
            uses ``np.nanmean``, and ``'zeros'`` uses a zero-valued baseline.
            A custom callable accepts an array and keyword argument ``axis``.
        scale_method : PeakScaleMethod, default='mad'
            Scale estimate: ``'mad'`` uses median absolute deviation,
            ``'std'`` uses ``np.nanstd``, and ``'ones'`` uses a one-valued
            scale. A custom callable accepts an array and keyword ``axis``.
        test_magnitude : float, default=3.0
            Scale multiplier used to set the detection threshold.
        """
        super().__init__(center_method, scale_method, test_magnitude)

    def _calc_centers(self, baselines: np.ndarray) -> np.ndarray:
        """Calculate row-wise center values from baselines."""
        return self.center_func(baselines, axis=1)

    def _calc_scales(self, baselines: np.ndarray) -> np.ndarray:
        """Calculate row-wise scale values from baselines."""
        return self.scale_func(baselines, axis=1)

    def detect(
            self,
            signals: np.ndarray,
            time: np.ndarray,
            frequency: float,
            baselines: np.ndarray | None = None,
            min_distance_sec: float | None = None,
            min_duration_sec: float | None = None,
            max_duration_sec: float | None = None,
            direction: PeakDirection = 'both',
            detailed: bool = False,
            ) -> PeakResult:
        """Detect peaks using static baseline-derived thresholds.

        Parameters
        ----------
        signals : np.ndarray
            Trial-by-time signal matrix.
        time : np.ndarray
            Time values for signal columns.
        frequency : float
            Sampling frequency in Hz.
        baselines : np.ndarray or None, default=None
            Baseline signals used to estimate threshold centers and scales. If
            ``None``, ``signals`` are used.
        min_distance_sec : float or None, default=None
            Minimum distance between peaks, in seconds.
        min_duration_sec : float or None, default=None
            Minimum peak duration, in seconds.
        max_duration_sec : float or None, default=None
            Maximum peak duration, in seconds.
        direction : PeakDirection, default='both'
            Direction of peaks to detect: ``'positive'`` detects peaks above
            baseline, ``'negative'`` detects peaks below baseline, and
            ``'both'`` detects both directions.
        detailed : bool, default=False
            If ``True``, include extra peak metrics.

        Returns
        -------
        PeakResult
            Detected peaks and metrics.
        """
        # handle None baselines
        baselines = signals if baselines is None else baselines

        # calculate scale and centers
        scales = self._calc_scales(baselines)
        centers = self._calc_centers(baselines)

        # coerce to row-wise aligment shape
        scales = self._coerce_to_rowwise_alignment(scales, signals)
        centers = self._coerce_to_rowwise_alignment(centers, signals)

        # execute detection
        peak_info = self._detect_from_threshold(
            centers=centers,
            scales=scales,
            signals=signals,
            time=time,
            frequency=frequency,
            min_distance_sec=min_distance_sec,
            min_duration_sec=min_duration_sec,
            max_duration_sec=max_duration_sec,
            direction=direction,
            detailed=detailed,
        )

        return peak_info

class RollingThresholdDetector(ThresholdDetector):
    """Detect peaks using rolling threshold estimates."""

    def __init__(
            self,
            window_width_sec: float = 5.0,
            center_method: PeakCenterMethod = 'median',
            scale_method: PeakScaleMethod = 'mad',
            test_magnitude: float = 3.0,
            ) -> None:
        """Initialize a rolling threshold detector.

        Parameters
        ----------
        window_width_sec : float, default=5.0
            Rolling window width, in seconds.
        center_method : PeakCenterMethod, default='median'
            Center estimate: ``'median'`` uses ``np.nanmedian``, ``'mean'``
            uses ``np.nanmean``, and ``'zeros'`` uses a zero-valued baseline.
            A custom callable accepts an array and keyword argument ``axis``.
        scale_method : PeakScaleMethod, default='mad'
            Scale estimate: ``'mad'`` uses median absolute deviation,
            ``'std'`` uses ``np.nanstd``, and ``'ones'`` uses a one-valued
            scale. A custom callable accepts an array and keyword ``axis``.
        test_magnitude : float, default=3.0
            Scale multiplier used to set the detection threshold.
        """
        super().__init__(center_method, scale_method, test_magnitude)

        self.window_width_sec = window_width_sec

    def _calc_centers(self, signals: np.ndarray, window_len: int) -> np.ndarray:
        """Calculate rolling center values."""
        centers = ndi.vectorized_filter(
            signals,
            function=self.center_func,
            size=(1, window_len),
            mode='reflect'
        )
        return centers

    def _calc_scales(self, signals: np.ndarray, window_len: int) -> np.ndarray:
        """Calculate rolling scale values."""
        scales = ndi.vectorized_filter(
            signals,
            function=self.scale_func,
            size=(1, window_len),
            mode='reflect',
        )
        return scales

    def detect(
            self,
            signals: np.ndarray,
            time: np.ndarray,
            frequency: float,
            baselines: np.ndarray | None = None,
            min_distance_sec: float | None = None,
            min_duration_sec: float | None = None,
            max_duration_sec: float | None = None,
            direction: PeakDirection = 'both',
            detailed: bool = False,
            ) -> PeakResult:
        """Detect peaks using rolling signal-derived thresholds.

        Parameters
        ----------
        signals : np.ndarray
            Trial-by-time signal matrix.
        time : np.ndarray
            Time values for signal columns.
        frequency : float
            Sampling frequency in Hz.
        baselines : np.ndarray or None, default=None
            Accepted for API compatibility but not used.
        min_distance_sec : float or None, default=None
            Minimum distance between peaks, in seconds.
        min_duration_sec : float or None, default=None
            Minimum peak duration, in seconds.
        max_duration_sec : float or None, default=None
            Maximum peak duration, in seconds.
        direction : PeakDirection, default='both'
            Direction of peaks to detect: ``'positive'`` detects peaks above
            baseline, ``'negative'`` detects peaks below baseline, and
            ``'both'`` detects both directions.
        detailed : bool, default=False
            If ``True``, include extra peak metrics.

        Returns
        -------
        PeakResult
            Detected peaks and metrics.
        """
        # calculate window idx len
        window_len = int(self.window_width_sec * frequency)

        # calculate scale and centers
        scales = self._calc_scales(signals, window_len)
        centers = self._calc_centers(signals, window_len)

        # validate shape
        scales = self._coerce_to_rowwise_alignment(scales, signals)
        centers = self._coerce_to_rowwise_alignment(centers, signals)

        # execute detection
        peak_info = self._detect_from_threshold(
            centers=centers,
            scales=scales,
            signals=signals,
            time=time,
            frequency=frequency,
            min_distance_sec=min_distance_sec,
            min_duration_sec=min_duration_sec,
            max_duration_sec=max_duration_sec,
            direction=direction,
            detailed=detailed,
        )

        return peak_info

class TemplateMatchingDetector(PeakDetector):
    """Placeholder for future template-matching peak detection."""

    def __init__(
            self
            ) -> None:
        """Initialize a template-matching detector placeholder."""
        pass
#endregion
