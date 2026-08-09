"""Photometry experiment processing."""

from __future__ import annotations
from collections.abc import Sequence
from numbers import Real
from typing import Any, Literal, Callable, TypeAlias
from scipy.signal import butter, sosfiltfilt
from sklearn.metrics import r2_score, mean_squared_error
from plotnine import ggplot
from plotnine.composition import Compose

import matplotlib.pyplot as plt
import matplotlib.axes
import numpy as np
import pandas as pd

from .PhotometryData import PhotometryData

from ..utils import graphing, operations, reference_fitting, window

from ..utils.window import WindowResult
from ..analysis.artifact import ArtifactResult, ArtifactDetector, ArtifactCorrector

from ..types import (
    ChannelMode,
    IsoFitMethod,
    CorrectionMethod,
    ExpNormMethod,
    TrialNormMethod,
    WindowMethod,
    InvalidWindowPolicy,
    EventSelectionLogic
)

class PhotometryExperiment:
    """Handle processing and trial extraction for continuous photometry data."""

    id: str
    metadata: dict[str, Any]
    events: dict[str, np.ndarray]
    raw_signal: np.ndarray
    raw_isosbestic: np.ndarray | None
    time: np.ndarray
    frequency: float
    artifacts: ArtifactResult | None

    signal: np.ndarray
    fitted_ref: np.ndarray
    trial_data: PhotometryData

    ###########################
    #region --- CONSTRUCTOR ---
    ###########################
    def __init__(
            self,
            raw_signal: np.ndarray,
            raw_isosbestic: np.ndarray | None,
            time: np.ndarray,
            events: dict = {},
            metadata: dict = {},
            frequency: float | None = None,
            ):
        """Initialize a photometry experiment.

        Parameters
        ----------
        raw_signal : np.ndarray
            Raw signal channel values.
        raw_isosbestic : np.ndarray or None
            Raw isosbestic channel values. If ``None``, the experiment is
            treated as single-channel.
        time : np.ndarray
            Time points corresponding to the raw signal.
        events : dict, default={}
            Mapping from event labels to timestamp arrays.
        metadata : dict, default={}
            Additional experiment metadata.
        frequency : float or None, default=None
            Sampling frequency in Hz. If ``None``, a frequency is estimated
            from ``raw_signal`` and ``time``.
        """
        # save inputs
        self.raw_signal = raw_signal
        self.raw_isosbestic = raw_isosbestic
        self.time = time
        self.events = events
        self.metadata = metadata
        self.frequency = (raw_signal.size - 1) / (self.time.max() - self.time.min()) if frequency is None else frequency

        self.id = None
        self.signal = None
        self.filt_sig = None
        self.filt_iso = None
        self.fitted_ref = None
        self.trial_data = None

    #endregion

    #######################
    #region --- DUNDERS ---
    #######################
    def __str__(self) -> str:
        """Return a compact experiment summary."""
        channel_str = 'Dual' if self.channel_mode == 'dual' else 'Single'
        out = (
            f"{channel_str} channel photometry experiment with {self.time.size} timepoints."
        )
        return out

    def __repr__(self) -> str:
        """Return the interactive representation."""
        return self.__str__()

    #endregion

    ##########################
    #region --- PROPERTIES ---
    ##########################
    @property
    def has_isosbestic(self) -> bool:
        """Whether an isosbestic channel is present."""
        return self.raw_isosbestic is not None
    @property
    def channel_mode(self) -> Literal['single', 'dual']:
        """Channel mode inferred from the presence of an isosbestic channel."""
        return 'dual' if self.has_isosbestic else 'single'
    @property
    def has_ran_preprocess(self) -> bool:
        """Whether `preprocess_signal` has populated ``self.signal``."""
        return self.signal is not None
    @property
    def has_ran_extraction(self) -> bool:
        """Whether trial data have been extracted."""
        return self.trial_data is not None
    @property
    def n_times(self) -> int:
        """Number of continuous time points."""
        return len(self.time)
    @property
    def event_labels(self) -> list:
        """Labels of events present"""
        return list(self.events.keys())

    #endregion

    ######################
    #region --- EXPORT ---
    ######################
    def to_wide_dataframe(
            self,
            downsample: int | None = None,
            export_events: bool = False,
            downsample_kwargs: dict = {},
            ) -> pd.DataFrame:
        """Export continuous traces to a wide dataframe.

        Parameters
        ----------
        downsample : int or None, default=None
            Downsampling factor applied to each exported trace. If ``None``,
            traces are exported at their current sampling.
        export_events : bool, default=False
            If ``True``, add one boolean column per event label with nearest
            sampled time points marked as event occurrences.
        downsample_kwargs : dict, default={}
            Additional keyword arguments passed to the downsampling helpers.

        Returns
        -------
        pd.DataFrame
            Wide dataframe with one row per exported time point.
        """
        def _column_add_helper(df: pd.DataFrame, col: str, val: np.ndarray | None) -> pd.DataFrame:
            """Add a downsampled array as a dataframe column when present."""
            if val is None: return df
            else: return df.assign(**{col: operations.downsample_signal(val, factor=downsample, **downsample_kwargs)})

        export_time = operations.downsample_time(self.time, factor=downsample, **downsample_kwargs)

        df = (
            pd.DataFrame({'time' : export_time}, index=np.arange(export_time.size))
            .pipe(_column_add_helper, 'raw_signal', self.raw_signal)
            .pipe(_column_add_helper, 'raw_isosbestic', self.raw_isosbestic)
            .pipe(_column_add_helper, 'processed_signal', self.signal)
            .pipe(_column_add_helper, 'fitted_reference', self.fitted_ref)
            .pipe(_column_add_helper, 'filtered_signal', self.filt_sig)
            .pipe(_column_add_helper, 'filtered_isosbestic', self.filt_iso)
        )

        if export_events:
            for label, timestamps in self.events.items():
                time_has_event = self._nearest_timestamp_mask(export_time, timestamps)
                df[label] = time_has_event

        return df

    def to_long_dataframe(
            self,
            downsample: int | None = None,
            downsample_kwargs: dict = {},
            ):
        """Export continuous traces to a long dataframe.

        Parameters
        ----------
        downsample : int or None, default=None
            Downsampling factor applied to each exported trace. If ``None``,
            traces are exported at their current sampling.
        downsample_kwargs : dict, default={}
            Additional keyword arguments passed to the downsampling helpers.

        Returns
        -------
        pd.DataFrame
            Long dataframe with ``time``, ``source``, and ``value`` columns.
        """
        def _df_builder_helper(source: str, arr: np.ndarray | None, time: np.ndarray) -> pd.DataFrame:
            """Build one long dataframe for a named trace."""
            index = np.arange(time.size)
            if arr is None:
                return pd.DataFrame(index=index)
            else:
                export_arr = operations.downsample_signal(arr, factor=downsample, **downsample_kwargs)
                return pd.DataFrame({'time':time, 'source':source, 'value':export_arr}, index=index)

        # downsample time
        export_time = operations.downsample_signal(self.time, factor=downsample, **downsample_kwargs)

        # built timeseries manifest
        manifest = dict(
            raw_signal = self.raw_signal,
            raw_isosbestic = self.raw_isosbestic,
            fitted_reference = self.fitted_ref,
            processed_signal = self.signal,
            filtered_signal = self.filt_sig,
            filtered_isosbestic = self.filt_iso,
        )

        # build list of dfs
        df_list = [_df_builder_helper(label, arr, export_time) for label, arr in manifest.items()]

        # concat result and return
        df = pd.concat(df_list, axis=0, ignore_index=True)
        return df

    def write_csv(
            self,
            file: str,
            downsample: int | None = None,
            export_events: bool = True,
            format: Literal['wide', 'long'] = 'wide',
            ) -> None:
        """Write continuous traces to CSV.

        Parameters
        ----------
        file : str
            Output CSV path.
        downsample : int or None, default=None
            Downsampling factor applied before export.
        export_events : bool, default=True
            Whether to include event indicator columns in wide format.
        format : {'wide', 'long'}, default='wide'
            Output dataframe layout.

        Raises
        ------
        ValueError
            If ``format`` is not recognized.
        """
        # fetch correct format
        if format == 'wide':
            df = self.to_wide_dataframe(downsample, export_events)
        elif format == 'long':
            df = self.to_long_dataframe(downsample)
        else:
            raise ValueError(f'Table format ({format}) not recognized.')
        # write
        df.to_csv(file)

    #endregion

    ######################
    #region --- IMPORT ---
    ######################
    @classmethod
    def load_TDT(
            cls,
            data_folder: str,
            box: str,
            event_labels: list[str],
            signal_label: str,
            isosbestic_label: str,
            downsample: int = 10,
            annotation_file: str | None = None,
            annotation_handler: Literal['json', 'yaml'] | Callable = 'json',
            ) -> PhotometryExperiment:
        """Load a photometry experiment from TDT format.

        Parameters
        ----------
        data_folder : str
            Path to the TDT block folder.
        box : str
            TDT box identifier appended to stream and epoc labels.
        event_labels : list[str]
            Event labels to extract from epocs.
        signal_label : str
            Base label for the signal stream.
        isosbestic_label : str
            Base label for the isosbestic stream.
        downsample : int, default=10
            Downsampling factor for raw streams.
        annotation_file : str or None, default=None
            Annotation file passed to the TDT loader.
        annotation_handler : {'json', 'yaml'} or Callable, default='json'
            Built-in annotation format name or custom annotation reader.

        Returns
        -------
        PhotometryExperiment
            Loaded experiment instance.
        """
        from .PhotometryLoader import TDTLoader
        loader = TDTLoader(
            data_folder=data_folder,
            box=box,
            event_labels=event_labels,
            signal_label=signal_label,
            isosbestic_label=isosbestic_label,
            downsample=downsample,
            annotation_file=annotation_file,
            annotation_handler=annotation_handler,
        )
        return loader.load()

    @classmethod
    def load_CSV(
            cls,
            csv: str,
            time_col: str = 'time',
            signal_col: str = 'signal',
            isosbestic_col: str | None = 'isosbestic',
            event_cols: str | list[str] | None = None,
            downsample: int | None = None,
            annotation_file: str | None = None,
            annotation_handler: Literal['json', 'yaml'] | Callable = 'json',
            ) -> PhotometryExperiment:
        """Load photometry experiment from a CSV.

        Parameters
        ----------
        csv : str
            Path to the CSV file containing photometry data.
        time_col : str, default='time'
            Column containing time values.
        signal_col : str, default='signal'
            Column containing signal values.
        isosbestic_col : str or None, default='isosbestic'
            Column containing isosbestic values.
        event_cols : str, list[str], or None, default=None
            Column or columns containing truthy values where events occur.
        downsample : int or None, default=None
            Downsampling factor for raw arrays.
        annotation_file : str or None, default=None
            Annotation file passed to the CSV loader.
        annotation_handler : {'json', 'yaml'} or Callable, default='json'
            Built-in annotation format name or custom annotation reader.

        Returns
        -------
        PhotometryExperiment
            Loaded experiment instance.
        """
        from .PhotometryLoader import CSVLoader
        loader = CSVLoader(
            csv=csv,
            time_col=time_col,
            signal_col=signal_col,
            isosbestic_col=isosbestic_col,
            event_cols=event_cols,
            downsample=downsample,
            annotation_file=annotation_file,
            annotation_handler=annotation_handler,
        )
        return loader.load()

    #endregion

    ##############################
    #region --- PREPROCESS API ---
    ##############################

    #TODO: make a detrend: bool arguement instead of having seperate detrend fit_using methods

    def preprocess_signal(
            self,
            cutoff_frequency: float | None = 3.0,
            order: int = 4,
            signal_normalization: ExpNormMethod = 'none',
            correction_method: CorrectionMethod = 'dF/F',
            fit_using: IsoFitMethod = 'IRLS',
            detrend: bool = False,
            add_intercept: bool = True,
            maxiter: int = 1000,
            c: float | None = 3,
            window_dur_sec: float = 5.0,
            channel_mode: ChannelMode = 'auto',
            artifact_detector: ArtifactDetector | None = None,
            artifact_corrector: ArtifactCorrector | None = None,
            ) -> None:
        """Low-pass filter and preprocess the signal using isosbestic fitting.

        Parameters
        ----------
        cutoff_frequency : float or None, default=3.0
            Low-pass cutoff frequency in Hz. If None, no lowpass filtering
            is performed.
        order : int, default=4
            Butterworth filter order.
        signal_normalization : ExpNormMethod, default='none'
            The method used for whole-experiment normalization.

            - ``zscore``: the traditional Z-score
            - ``nullZ``: division by the signals root-mean-square deviation from zero without centering
            - ``none``: no whole-experiment normalization
        correction_method : CorrectionMethod, default='dF/F'
            How to use the fitted reference signal (the isosbestic for dual-channel and 
            the fit photobleaching curve for single-channel) to correct the experimental signal.

            For dual-channel experiments:
            - ``dF/F``: ``(experiment - fit isosbestic) / fit isosbestic`` 
            (corrects for photobleaching attenuation)
            - ``dF``: ``(experiment - fit isosbestic)``,
            (does NOT correct for photobleaching attenuation on its own)

            For single-channel experiments:
            - ``dB/B``: ``(experiment - fit photobleaching) / fit photobleaching``,
            (corrects for photobleaching attenuation)
            - ``dB``: ``(experiment - fit photobleaching)``,
            (does NOT correct for photobleaching attenuation on its own)

            For both:
            - A custom function that accepts ``(signal: np.ndarray, fitted reference: np.ndarray)``
            positionally and returns the corrected signal as a ``np.ndarray``
            - ``none``: no transformation based on the fitted reference signal is performed,
            used mainly for debugging.
        fit_using : IsoFitMethod, default='IRLS'
            The method used to fit the isosbestic to experimnetal signal in dual-channel processing.

            - ``OLS``: ordinary least squares
            - ``IRLS``: iteratively reweighted least squares (recommended default)
            - A custom function that accepts ``(signal: np.ndarray, isosbestic: np.ndarray)`` 
            positionally and returns the fitted isosbestic signal as a ``np.ndarray``
        detrend : bool, default=True
            Whether to detrend the experimental and isosbestic signals before fitting.
            Only applies to dual-channel workflows.
        add_intercept : bool, default=True
            Whether to include an intercept in the isosbestic fitting.
        maxiter : int, default=1000
            Maximum iterations for IRLS fitting. Effects only ``fit_using = IRLS``.
        c : float or None, default=3
            IRLS tuning constant. Effects only ``fit_using = IRLS``.
        window_dur_sec : float, default=5
            Sliding-window duration for photobleaching strided median
            downsampling in seconds. Relevant to ``correction_method == dB or dB/B``
            and ``fit_using == detrend_retrend or detrend_fit_retrend``.
        channel_mode : ChannelMode, default='auto'
            Workflow used to preprocess the signal.

            - ``'auto'`` detects the channel mode from the presence or absence
              of an isosbestic signal.
            - ``'dual'`` uses isosbestic fitting and correction.
            - ``'single'`` uses photobleaching detrending on the experimental
              signal.
        artifact_detector : ArtifactDetector or None, default=None
            Optional detector used to identify artifacts.
        artifact_corrector : ArtifactCorrector or None, default=None
            Optional corrector used to modify the processed signal at detected
            artifacts.

        Raises
        ------
        ValueError
            If the correction method is incompatible with the channel mode, or
            if an artifact corrector is supplied without a detector.
        """
        # validate inputs
        if channel_mode == 'auto':
            channel_mode = self.channel_mode

        dual_channel_methods = ['dF/F', 'dF']
        single_channel_methods = ['dB/B', 'dB']

        if channel_mode == 'single' and correction_method in dual_channel_methods:
            raise ValueError(f'Correction methods {", ".join(dual_channel_methods)} are for dual channel experiments.')

        if channel_mode == 'dual' and correction_method in single_channel_methods:
            raise ValueError(f'Correction methods {", ".join(single_channel_methods)} are for single channel experiments.')

        if artifact_corrector is not None and artifact_detector is None:
            raise ValueError(f'An artifact_corrector cannot be passed with also passing an artifact_detector is.')

        # apply lowpass butterworth filter
        filt_sig = self.low_frequency_pass_butter(
            self.raw_signal,
            self.frequency,
            cutoff_frequency=cutoff_frequency,
            order=order
        )

        if channel_mode == 'dual':
            # apply lowpass to isosbestic
            filt_iso = self.low_frequency_pass_butter(
                self.raw_isosbestic,
                self.frequency,
                cutoff_frequency=cutoff_frequency,
                order=order
            )

            # trim to minimum length
            min_len = min(filt_sig.size, filt_iso.size)
            filt_sig = filt_sig[:min_len]
            filt_iso = filt_iso[:min_len]

            # optionally detrend signals
            if detrend:
                filt_sig, _, _ = self.detrend_signal(
                    signal=filt_sig, 
                    window_dur_sec=window_dur_sec,
                )
                filt_iso, _, _ = self.detrend_signal(
                    signal=filt_iso, 
                    window_dur_sec=window_dur_sec,
                )

            # fit isosbestic to signal
            fitted_ref, r2_val, coeffs = self.fit_isosbestic_to_signal(
                filt_sig,
                filt_iso,
                add_intercept=add_intercept,
                maxiter=maxiter,
                fit_using=fit_using,
                c=c,
            )
            reference_type = 'isosbestic'

            # save filtered iso
            self.filt_iso = filt_iso

        elif channel_mode == 'single':
            # fit photobleaching curve
            fitted_ref, r2_val, coeffs = self.fit_photobleaching_curve(
                signal=filt_sig,
                window_dur_sec=window_dur_sec,
            )
            reference_type = 'photobleaching'

        else:
            raise ValueError(f"Channel mode, {channel_mode}, not recognized.")

        # save relevant data
        self.metadata['reference_fit'] = dict(
            type = reference_type,
            r2_val = r2_val,
            coeffs = coeffs,
        )
        self.filt_sig = filt_sig
        self.fitted_ref = fitted_ref

        # apply reference curve correction
        signal = self._apply_correction_method(
            correction_method=correction_method,
            signal=filt_sig,
            fitted_ref=fitted_ref,
        )

        # apply signal normalization
        signal = self._apply_signal_normalization(
            signal_normalization=signal_normalization,
            signal=signal,
        )

        # detect artifacts
        artifacts = None
        if artifact_detector is not None:
            reference = None if channel_mode == 'single' else self.fitted_ref

            artifacts = self._detect_artifacts(
                detector=artifact_detector,
                signal=signal,
                reference=reference,
                time=self.time,
            )

            # correct artifacts
            if artifact_corrector is not None:
                signal = self._correct_artifacts(
                    corrector=artifact_corrector,
                    artifacts=artifacts,
                    signal=signal,
                    time=self.time,
                )

        # save results and end pipeline
        self.signal = signal
        self.artifacts = artifacts

        if callable(correction_method): self.metadata['correction_method'] = correction_method.__name__
        else: self.metadata['correction_method'] = correction_method

        return

    #endregion

    ##############################
    #region --- EXTRACTION API ---
    ##############################

    def extract_trial_data(
            self,
            align_to: str | Sequence[str] | float | Sequence[float],
            trial_bounds: tuple[float, float],
            center_on: str | Sequence[str] | None = None,
            baseline_bounds: tuple[float, float] | None = None,
            event_tolerences: dict[str, tuple[float, float] | None] = {},
            trial_normalization: TrialNormMethod = 'none',
            check_overlap: bool = True,
            all_events: bool = True,
            window_alignment: WindowMethod = 'nearest',
            invalid_window_policy: InvalidWindowPolicy = 'drop',
            event_conflict_logic: EventSelectionLogic = 'first',
            ) -> None:
        """Build trial-wise windows, normalize, and store trial data.

        Parameters
        ----------
        align_to : str, Sequence[str], float, or Sequence[float]
            Event label, event labels, timestamp, or timestamps used to define
            candidate trials. Multiple event labels are pooled by timestamp and
            an ``align_event`` observation column is added.
        trial_bounds : tuple[float, float]
            Window bounds relative to the selected trial center.
        center_on : str, Sequence[str], or None, default=None
            Optional event labels used to recenter each trial. If no selected
            ``center_on`` event is present in a trial, that trial remains
            centered on its ``align_to`` timestamp.
        baseline_bounds : tuple[float, float] or None, default=None
            Baseline window bounds relative to the ``align_to`` timestamp.
            Required for baseline-dependent trial normalizations.
        event_tolerences : dict[str, tuple[float, float] or None], default={}
            Event annotation windows relative to ``align_to`` timestamps.
            ``None`` values are replaced with ``trial_bounds``.
        trial_normalization : TrialNormMethod, default='none'
            The method used for trial-wise normalization based on a specified baseline region.

            - ``zscore``: the traditional Z-score using the standard deviation and mean
            of the baseline region
            - ``zero``: centers trials by the mean of the baseline
            - ``mad``: robust Z-score using the median absolute deviation and median
            of the baseline region
            - ``amp``: scales the trials by their absolute maximum
            - ``none``: no trial-wise normalization
            - A custom function that accepts ``(signal: np.ndarray, baselines: np.ndarray)``
            and returns the normalized signals as a 2D ``np.ndarray`` of the same shape of 
            ``signal``
        check_overlap : bool, default=True
            If ``True``, raise when more than one ``center_on`` event is found
            for the same trial.
        all_events : bool, default=True
            If ``True``, passdown all events even if they are not 
            present in ``event_tolerences``.
        window_alignment : WindowMethod, default='nearest'
            The method used to window time-series.

            - ``nearest``: snaps events to the nearest sampled timepoint and slices
            windows based to the nearest time point fitting within the window
            - ``interp``: exactly center windows to events and use liner interpolation to
            interpolate the signal at the exact time grid built around the center
        invalid_window_policy : InvalidWindowPolicy, default='drop'
            How to handle windows with bounds that extend outside of the time-series
            being windowed.

            -``drop``: drops the invalid windows without raising an error
            -``error``: raises a ``ValueError`` if there are invalid windows
        event_conflict_logic : EventSelectionLogic, default='first'
            Rule used to select timestamp(s) if multiple timestamps for the same 
            event label fall inside a trial annotation window. 

            -``first``: only selects the first occurence of the event
            -``last``: only selects the last occurence of the event
            -``all``: keep all occurrences, but relabels them, with the first occurrence mantaining the base label and subsequent ones being relabeled as ``f'{base_label}_occurrence_{n}'`` 
            -``mean``: uses the average timestamp of the multiple occurences

        Raises
        ------
        ValueError
            If baseline-dependent normalization lacks baseline bounds, if
            invalid windows are disallowed, or if rounded timing error exceeds
            ``time_error_threshold``.
        """
        # validate inputs
        if baseline_bounds is None:
            calc_baselines = False
            #TODO: clean this
            if trial_normalization in ['zscore', 'zero', 'mad', 'amp']:
                raise ValueError(
                    f'Baseline bounds have to be specified for normalization method {trial_normalization}'
                )
        else:
            calc_baselines = True

        # handle alignment
        align_label, align_events, align_obs = self._resolve_alignment(align_to)

        # coerce centering
        center_on = self._coerce_centering(center_on)

        # coerce tolerences
        event_tolerences = self._coerce_tolerances(event_tolerences, center_on, trial_bounds, all_events)

        # annotate events based on tolerences
        events_selected = self._annotate_intervals(
            align_label=align_label,
            series=self.time,
            centers=align_events,
            events=self.events,
            tolorences=event_tolerences,
            logic=event_conflict_logic
        )

        # find window centers using the selected events
        trial_window_centers = self._find_window_centers(
            center_on=center_on,
            align_events=align_events,
            events=events_selected,
            check_overlap=check_overlap,
        )

        # construct trial windows
        trial_windows = self._create_windows(
            signal=self.signal,
            time=self.time,
            events=events_selected,
            centers=trial_window_centers,
            bounds=trial_bounds,
            strategy=window_alignment,
        )

        # do the same for baseline
        if calc_baselines:
            baseline_windows = self._create_windows(
                signal=self.signal,
                time=self.time,
                events=events_selected,
                centers=align_events,
                bounds=baseline_bounds,
                strategy=window_alignment,
            )
        else:
            baseline_windows = None

        # handle invalid windows
        valid_trial_mask = self._handle_invalid_windows(
            trial_windows=trial_windows,
            baseline_windows=baseline_windows,
            policy=invalid_window_policy,
        )

        # apply trial-wise normalization method
        processed_trial_signals = self._apply_trial_normalization(
            trial_normalization=trial_normalization,
            trial_windows=trial_windows,
            baseline_windows=baseline_windows,
        )

        # construct trial df info
        trial_obs = pd.concat(
                [
                    trial_windows.to_obs(add_trial_num=True),
                    align_obs.loc[valid_trial_mask].reset_index(drop=True),
                ],
                axis=1,
            )

        # save trial data
        self.trial_data = PhotometryData.from_arrays(
            obs=trial_obs,
            data=processed_trial_signals,
            time_points=trial_windows.time_grid,
            metadata=self.metadata.copy(),
        )

        # save baseline data if applicable
        if calc_baselines:
            baseline_obs = pd.concat(
                    [
                        baseline_windows.to_obs(add_trial_num=True),
                        align_obs.loc[valid_trial_mask].reset_index(drop=True),
                    ],
                    axis=1,
                )
            self.baseline_data = PhotometryData.from_arrays(
                obs=baseline_obs,
                data=baseline_windows.signals,
                time_points=baseline_windows.time_grid,
                metadata=self.metadata.copy(),
            )

    #endregion

    ##################################
    #region --- PREPROCESS HELPERS ---
    ##################################
    def low_frequency_pass_butter(
            self,
            signal: np.ndarray,
            sample_frequency: float,
            cutoff_frequency: float | None = 3.0,
            order: int = 4,
            axis: int = 0,
        ) -> np.ndarray:
        """Apply a low-pass Butterworth filter to a signal.

        Parameters
        ----------
        signal : np.ndarray
            Input signal array.
        sample_frequency : float
            Sampling frequency in Hz.
        cutoff_frequency : float or None, default=3.0
            Low-pass cutoff frequency in Hz. If None, no lowpass filtering
            is performed.
        order : int, default=4
            Butterworth filter order.
        axis : int, default=0
            Axis along which filtering is applied.

        Returns
        -------
        np.ndarray
            Filtered signal.
        """
        # do nothing in None case
        if cutoff_frequency is None:
            return signal
        # perform lowpass
        normalized_frequency = cutoff_frequency / (sample_frequency / 2)
        sos = butter(order, normalized_frequency, btype='low', output='sos')
        return sosfiltfilt(sos, signal, axis=axis, padtype='odd', padlen=None)

    def fit_isosbestic_to_signal(
            self,
            signal: np.ndarray,
            isosbestic: np.ndarray,
            fit_using: IsoFitMethod = 'IRLS',
            add_intercept: bool = True,
            maxiter: int = 1000,
            c: float | None = None,
            ) -> tuple[np.ndarray, float, Any]:
        """Fit the isosbestic channel to the signal.

        Parameters
        ----------
        signal : np.ndarray
            Filtered signal trace.
        isosbestic : np.ndarray
            Filtered isosbestic trace.
        fit_using : IsoFitMethod, default='IRLS'
            Method used to fit the isosbestic to the experimental signal.

            - ``'OLS'`` uses ordinary least squares.
            - ``'IRLS'`` uses iteratively reweighted least squares.
            - A custom callable accepts ``signal`` and ``isosbestic`` arrays
              positionally and returns the fitted isosbestic signal and fit
              parameters.
        maxiter : int, default=1000
            Maximum iterations for IRLS fitting.
        c : float or None, default=None
            IRLS tuning constant.
        window_dur_sec : float, default=5
            Sliding-window duration for photobleaching strided median
            downsampling in seconds.

        Returns
        -------
        fitted_iso : np.ndarray
            Fitted reference trace.
        r2_val : float
            Coefficient of determination for the fit.
        params : Any
            Fit coefficients or method-specific fit parameters.

        Raises
        ------
        ValueError
            If ``fit_using`` is unknown or the fit contains NaN values.
        """
        # fit using specified model
        match fit_using:
            case func if callable(func):
                fitted_iso, params = fit_using(signal, isosbestic)
            case 'OLS':
                fitted_iso, params = reference_fitting.OLS_fit(
                    signal, isosbestic, add_intercept=add_intercept
                )
            case 'IRLS':
                fitted_iso, params = reference_fitting.IRLS_fit(
                    signal, isosbestic, maxiter=maxiter, c=c, add_intercept=add_intercept
                )
            case _:
                raise ValueError(f'{fit_using} fitting method not recognized!')

        # cheack output
        if np.isnan(fitted_iso).any():
            raise ValueError(f'NaN values in fitted isosbestic.')
        r2_val = r2_score(signal, fitted_iso)
        return fitted_iso, r2_val, params

    def fit_photobleaching_curve(
            self,
            signal: np.ndarray,
            window_dur_sec: float =  5,
            ) -> tuple[np.ndarray, float, list[float]]:
        """Fit a curve with a negative bi-exponential photobleaching model.

        The fit uses soft-L1 least squares on a sliding-window median
        downsampled signal.

        Parameters
        ----------
        signal : np.ndarray
            Signal array to fit.
        window_dur_sec : float, default=5
            Sliding-window duration in seconds.

        Returns
        -------
        fitted_curve : np.ndarray
            Fitted photobleaching curve.
        r2_val : float
            Coefficient of determination for the fit.
        params : list[float]
            Fitted model parameters: ``a1``, ``tau1``, ``a2``, ``tau2``, and
            ``c``.

        Raises
        ------
        ValueError
            If the fitted curve contains NaN values.
        """
        window_len = int(window_dur_sec * self.frequency)
        fitted_curve, params = reference_fitting.fit_photobleaching(signal, self.time, window=window_len)
        r2_val = r2_score(signal, fitted_curve)

        if np.isnan(fitted_curve).any():
            raise ValueError(f'Fitted photobleaching curve produced NaNs.')

        return fitted_curve, r2_val, params
    
    def detrend_signal(
            self,
            signal: np.ndarray,
            window_dur_sec: float = 5,
            ) -> tuple[np.ndarray, float, list[float]]:
        bleaching, r2_val, params = self.fit_photobleaching_curve(
            signal=signal,
            window_dur_sec=window_dur_sec,
        )

        detrended = (signal - bleaching) / np.maximum(bleaching, np.finfo(np.float32).eps)
        return detrended, r2_val, params

    # --- hidden signal preprocess helpers ---
    def _apply_correction_method(
            self,
            correction_method: CorrectionMethod,
            signal: np.ndarray,
            fitted_ref: np.ndarray,
            ) -> np.ndarray:
        """Apply the configured reference-correction method."""

        match correction_method:
            case func if callable(func):
                out: np.ndarray = correction_method(signal, fitted_ref)
            case 'dF' | 'dB':
                out = (signal - fitted_ref)
            case 'dF/F' | 'dB/B':
                out = (signal - fitted_ref) / np.maximum(fitted_ref, np.finfo(np.float32).eps)
            case 'none':
                out = signal
            case _:
                raise ValueError(f'{correction_method} isosbestic correction method not recognized!')
        return out

    def _apply_signal_normalization(
            self,
            signal_normalization: ExpNormMethod,
            signal : np.ndarray,
            ) -> np.ndarray:
        """Apply the configured whole-signal normalization method."""

        match signal_normalization:
            case func if callable(func):
                out: np.ndarray = signal_normalization(signal)
            case 'zscore':
                out = (signal - np.mean(signal)) / np.std(signal)
            case 'nullZ':
                out = signal / np.sqrt(np.mean(signal**2))
            case 'none':
                out = signal
            case _:
                raise ValueError(f'{signal_normalization} whole signal normalization method not recognized!')
        return out

    def _detect_artifacts(
            self,
            detector: ArtifactDetector,
            signal: np.ndarray,
            reference: np.ndarray,
            time: np.ndarray,
            ) -> ArtifactResult:
        """Run the configured artifact detector."""

        artifacts = detector.detect(
            signal=signal,
            reference=reference,
            time=time,
        )
        return artifacts

    def _correct_artifacts(
            self,
            corrector: ArtifactCorrector,
            artifacts: ArtifactResult,
            signal: np.ndarray,
            time: np.ndarray,
            ) -> np.ndarray:
        """Run the configured artifact corrector."""

        corrected = corrector.correct(
            signal=signal,
            time=time,
            artifacts=artifacts,
        )
        return corrected

    #endregion

    ##################################
    #region --- EXTRACTION HELPERS ---
    ##################################
    def _apply_trial_normalization(
            self,
            trial_normalization: TrialNormMethod,
            trial_windows: WindowResult,
            baseline_windows: WindowResult | None,
            ) -> np.ndarray:
        """Apply the configured per-trial normalization method."""
        # unpack
        raw_trial_signals = trial_windows.signals
        baseline_signals = None if baseline_windows is None else baseline_windows.signals

        # execute
        match trial_normalization:
            case func if callable(func):
                trial_signals = trial_normalization(raw_trial_signals, baseline_signals)
            case 'zscore':
                trial_signals = operations.zscore_signal(raw_trial_signals, baseline_signals)
            case 'zero':
                trial_signals = operations.center_signal(raw_trial_signals, baseline_signals)
            case 'mad':
                trial_signals = operations.mad_norm_signal(raw_trial_signals, baseline_signals)
            case 'amp':
                trial_signals = operations.amp_norm_signal(raw_trial_signals, baseline_signals)
            case 'none':
                trial_signals = raw_trial_signals.copy()
            case _:
                raise ValueError(f'{trial_normalization} trial-wise normalization method not recognized!')
        return trial_signals

    def _find_interval_bounds(self, series: np.ndarray, centers: np.ndarray, bounds: tuple[int, int]) -> np.ndarray:
        """Compute index bounds for intervals around centers."""
        low, high = bounds
        left_idxs = np.searchsorted(series, centers + low, side='left')
        right_idxs = np.searchsorted(series, centers + high, side='right')
        return np.c_[left_idxs, right_idxs]

    def _find_timestamp_in_intervals(
            self,
            label: str,
            timestamps: np.ndarray,
            time_intervals: np.ndarray,
            logic: EventSelectionLogic = 'first',
            ) -> dict[str, np.ndarray]:
        """Find timestamps within each time interval using customizable logic."""
        timestamps = np.sort(timestamps)

        # tests which events are in valid intervals
        valid_intervals = time_intervals[:, 0] <= time_intervals[:, 1]
        lo_idx = np.searchsorted(timestamps, time_intervals[:, 0], side="left")
        hi_idx = np.searchsorted(timestamps, time_intervals[:, 1], side="right")
        in_interval = valid_intervals & (lo_idx < hi_idx)

        # pre allocate
        n_intervals = len(time_intervals)
        interval_timestamps = np.full(n_intervals, np.nan, float)

        # perform choice logic
        match logic:

            case 'first':
                interval_timestamps[in_interval] = timestamps[lo_idx[in_interval]]
                return {label : interval_timestamps}

            case 'last':
                interval_timestamps[in_interval] = timestamps[hi_idx[in_interval] - 1]
                return {label : interval_timestamps}

            case 'mean':
                counts = hi_idx - lo_idx
                in_interval = counts > 0

                csum = np.concatenate(([0.0], np.cumsum(timestamps)))
                sums = csum[hi_idx] - csum[lo_idx]

                interval_timestamps[in_interval] = sums[in_interval] / counts[in_interval]
                return {label : interval_timestamps}

            case 'all':
                # find max occurence count
                counts = np.where(in_interval, hi_idx - lo_idx, 0)
                max_count = int(counts.max(initial=0))

                # zero case
                if max_count == 0:
                    return {label : interval_timestamps}

                # pack values into 2D array
                values = np.full((n_intervals, max_count), np.nan, dtype=float)

                occurrence_offsets = np.arange(max_count)
                candidate_idxs = lo_idx[:, None] + occurrence_offsets[None, :] # canidate timestamp idxs
                # only add if that trial acutal had that many occurences
                valid_occurrences = occurrence_offsets[None, :] < counts[:, None]
                values[valid_occurrences] = timestamps[candidate_idxs[valid_occurrences]]

                # package into dict, with + _n# naming scheme
                result = {label: values[:, 0]}
                for occurrence_idx in range(2, max_count + 1):
                    result[f"{label}_occurrence_{occurrence_idx}"] = values[:, occurrence_idx - 1]

                return result

            case _:
                raise ValueError(f'Multiple event selection logic, {logic}, is not recognized')

    def _resolve_alignment(
            self,
            align_to: str | Sequence[str] | float | Sequence[float],
            ) -> tuple[str, np.ndarray, pd.DataFrame]:
        """Resolve event labels or timestamps into trial alignments."""
        # set default manual alignment label
        ALIGNMENT_LABEL = 'ALIGNMENTS'

        # single event
        if isinstance(align_to, str):
            if align_to not in self.events:
                raise KeyError(
                    f"align_to '{align_to}' not found in events. "
                    f"Available events: {list(self.events)}"
                )

            align_events = self.events[align_to].copy()
            align_obs = pd.DataFrame(index=np.arange(align_events.size))
            align_label = align_to

        # single timestamp
        elif isinstance(align_to, Real) and not isinstance(align_to, bool):
            align_events = np.asarray([align_to])
            align_obs = pd.DataFrame(index=np.arange(align_events.size))
            align_label = ALIGNMENT_LABEL

        # multiple events
        elif isinstance(align_to, Sequence) and all(isinstance(v, str) for v in align_to):
            if len(align_to) == 0:
                raise ValueError("align_to must contain at least one event label.")

            missing = [label for label in align_to if label not in self.events]
            if missing:
                raise KeyError(
                    f"align_to labels not found in events: {missing}. "
                    f"Available events: {list(self.events)}"
                )

            times = np.concatenate([self.events[label] for label in align_to])
            sources = np.concatenate([
                np.repeat(label, len(self.events[label]))
                for label in align_to
            ])

            order = np.argsort(times, kind="stable")

            align_events = times[order]
            align_obs = pd.DataFrame(
                {"align_event": sources[order]},
                index=np.arange(align_events.size),
            )
            align_label = ALIGNMENT_LABEL

        # multiple timestamps
        elif isinstance(align_to, Sequence) and all(isinstance(v, (float, int)) for v in align_to):
            if len(align_to) == 0:
                raise ValueError("align_to must contain at least one timestamp.")

            align_events = np.asarray(align_to, dtype=float)
            align_obs = pd.DataFrame(index=np.arange(align_events.size))
            align_label = ALIGNMENT_LABEL

        else:
            raise ValueError(f'Invalid input type for align_to: {type(align_to)}.')

        # validate
        if align_events.size == 0:
            raise ValueError(f"No align_to ({align_to}) events found.")

        if isinstance(align_events, np.ndarray) and (align_events.ndim != 1):
            raise ValueError(f'align_events is not 1D-array, type: {type(align_events)}, ndim: {align_events.ndim}.')

        return align_label, align_events, align_obs

    def _coerce_tolerances(
            self,
            event_tolerences: dict[str, tuple[float, float] | None] | None,
            center_on: list[str],
            trial_bounds: tuple[float, float],
            all_events: bool,
            ) -> dict[str, tuple[float, float]]:
        """Normalize event tolerance inputs and fill defaults."""

        event_tolerences = dict(event_tolerences or {})

        # if all_events ensure all events are present
        if all_events:
            for label in self.event_labels:
                event_tolerences.setdefault(label, trial_bounds)
                
        # else ensure all center_on events present
        else:
            for label in center_on:
                event_tolerences.setdefault(label, trial_bounds)

        # change None to maximum tolerance
        for label, tolerance in event_tolerences.items():
            if tolerance is None:
                event_tolerences[label] = trial_bounds

        new_tolerences: dict[str, tuple[float, float]] = event_tolerences
        return new_tolerences

    def _coerce_centering(
            self,
            center_on: str | Sequence[str] | None,
            ) -> list[str]:
        """Normalize center event labels to a validated list."""
        if center_on is None:
            center_out = []

        elif isinstance(center_on, str):
            if center_on not in self.events:
                raise KeyError(
                f"center_on label ({center_on}) not found in events. "
                f"Available events: {list(self.events)}"
            )
            center_out = [center_on]

        elif isinstance(center_on, Sequence):
            missing = [label for label in center_on if label not in self.events]
            if missing:
                raise KeyError(
                f"center_on labels ({missing}) not found in events. "
                f"Available events: {list(self.events)}"
            )
            center_out = list(center_on)
        return center_out

    def _annotate_intervals(
            self,
            align_label: str,
            series: np.ndarray,
            centers: np.ndarray,
            events: dict[str, np.ndarray],
            tolorences: dict[str, np.ndarray],
            logic: EventSelectionLogic = 'first',
            ) -> dict[str, np.ndarray]:
        """Annotate intervals around centers with event timestamps."""
        out = {align_label: centers.copy()}
        tmin, tmax = series[0], series[-1]

        for label, bounds in tolorences.items():
            # validation
            low, high = map(float, bounds)
            if low > high:
                raise ValueError(f"Invalid bounds for {label}: {bounds}")

            timestamps = events.get(label)
            if timestamps is None:
                out[label] = np.full_like(centers, np.nan, dtype=float)
                continue

            timestamps = np.asarray(timestamps, dtype=float)
            if timestamps.ndim != 1:
                raise ValueError(f"Event timestamps for {label} must be 1D")

            # build and clip intervals to time series bounds
            time_intervals = np.column_stack((centers + low, centers + high))
            time_intervals[:, 0] = np.maximum(time_intervals[:, 0], tmin)
            time_intervals[:, 1] = np.minimum(time_intervals[:, 1], tmax)

            # pass to helper to actually find timestamps in intervals
            out.update(
                self._find_timestamp_in_intervals(
                    label=label,
                    timestamps=timestamps,
                    time_intervals=time_intervals,
                    logic=logic
                )
            )
        return out

    def _create_windows(
            self,
            signal: np.ndarray,
            time: np.ndarray,
            events: dict[str, np.ndarray],
            centers: np.ndarray,
            bounds: tuple[float, float],
            strategy: WindowMethod = 'nearest',
            ) -> WindowResult:
        """Create one-dimensional windows using the requested strategy."""
        # execute windowing
        match strategy:
            case 'nearest':
                return window.create_windows_nearest_1D(signal, time, events, centers, bounds, self.frequency)
            case 'interp':
                return window.create_windows_interp_1D(signal, time, events, centers, bounds, self.frequency)
            case _:
                raise ValueError(f'Window alignment strategy {strategy} not recognized.')

    def _handle_invalid_windows(
            self,
            trial_windows: WindowResult,
            baseline_windows: WindowResult | None,
            policy: InvalidWindowPolicy = 'drop'
            ) -> np.ndarray:
        """Apply invalid-window policy to trial and baseline windows."""
        # find valid trials
        if baseline_windows is None:
            invalid_mask = trial_windows.invalid_mask
        else:
            invalid_mask = trial_windows.invalid_mask | baseline_windows.invalid_mask
        valid_mask = ~invalid_mask

        # apply policy
        if invalid_mask.any():
            invalid_idxs = np.flatnonzero(invalid_mask).tolist()
            self.metadata['invalid_windows'] = invalid_idxs

            match policy:
                case 'drop':
                    trial_windows.apply_mask(valid_mask)
                    if baseline_windows is not None:
                        baseline_windows.apply_mask(valid_mask)

                case 'error':
                    raise ValueError(
                        f'Invalid trial windows that extend outside signal range at trial indicies {invalid_idxs}'
                    )
                case _:
                    raise ValueError(f'Invalid window policy {policy} not recognized.')
        else:
            self.metadata['invalid_windows'] = None

        # confirm no invalid windows left
        assert not trial_windows.invalid_mask.any()

        # confirm some trials remain
        if trial_windows.signals.size == 0:
            raise ValueError(f'No trials remain after dropping invalid windows.')

        return valid_mask

    def _find_window_centers(
            self,
            center_on: list[str],
            align_events: np.ndarray,
            events: dict[str, np.ndarray],
            check_overlap: bool = True
            ) -> np.ndarray:
        """Determine window centers based on ``center_on`` events."""
        # center_on events should be non-overlaping
        # if no center_on events present, center on align_on
        centers = align_events.copy()
        present_count = np.zeros_like(centers, dtype=int)

        if len(center_on) == 0:
            return centers

        for label in center_on:
            arr = events[label]
            event_not_nan = ~np.isnan(arr)
            centers[event_not_nan] = arr[event_not_nan]
            present_count += event_not_nan.astype(int)

        overlap = present_count > 1

        if check_overlap and overlap.any():
            at_idxs = np.where(overlap)
            culprits = {k : events[k][at_idxs] for k in center_on}
            culprits_in_og_events = {k : self.events[k][np.searchsorted(self.events[k], culprits[k])] for k in center_on}
            raise ValueError(f'Center_on events over lap in trials {np.where(overlap)}, with culprits: {culprits_in_og_events}')
        return centers

    #endregion

    #############################
    #region --- EXTRA UTILITY ---
    #############################
    def _nearest_timestamp_mask(self, times: np.ndarray, timestamps: np.ndarray):
        """Get boolean mask of time, with nearest times to timestamp as True."""
        mask = np.zeros(times.shape, dtype=bool)

        if len(times) == 0 or len(timestamps) == 0:
            return mask

        if len(times) == 1:
            mask[0] = True
            return mask

        idx = np.searchsorted(times, timestamps)

        idx_right = np.clip(idx, 0, len(times) - 1)
        idx_left = np.clip(idx - 1, 0, len(times) - 1)

        dist_left = np.abs(timestamps - times[idx_left])
        dist_right = np.abs(timestamps - times[idx_right])

        nearest_idx = np.where(dist_right < dist_left, idx_right, idx_left)

        mask[nearest_idx] = True
        return mask

    def _apply_time_mask(self, mask: np.ndarray) -> None:
        """Filter time and all present timeseries by boolean mask."""
        mask = np.asarray(mask, dtype=bool)

        # validate
        if mask.shape != self.time.shape:
            raise ValueError('mask must match time shape')

        if not mask.any():
            raise ValueError('Trim would remove all timepoints')

        # apply mask
        old_time = self.time
        new_time = old_time[mask]

        for attr in [
            'raw_signal',
            'raw_isosbestic',
            'signal',
            'filt_sig',
            'filt_iso',
            'fitted_ref',
        ]:
            arr = getattr(self, attr, None)
            if arr is not None:
                setattr(self, attr, arr[mask])

        self.time = new_time

        lower = new_time[0]
        upper = new_time[-1]
        self.events = {
            label: np.asarray(timestamps)[
                (np.asarray(timestamps) >= lower) & (np.asarray(timestamps) <= upper)
            ]
            for label, timestamps in self.events.items()
        }

    def trim_times_by_index(
            self,
            start_idx: int | None = None,
            stop_idx: int | None = None,
            ) -> None:
        """Trim all time-series by start and stop indexes.

        WARNING: It is not recommended you do this after
        trial extraction or artifact detection as the time series
        will no longer fully align.

        Parameters
        ----------
        start_idx : int or None, default=None
            Inclusive starting index. If ``None``, no lower index bound is
            applied.
        stop_idx : int or None, default=None
            Exclusive stopping index. If ``None``, no upper index bound is
            applied.

        Raises
        ------
        ValueError
            If indexes are outside the time range or do not define a non-empty
            interval.
        """
        if (start_idx is None) and (stop_idx is None): return

        start = 0 if start_idx is None else start_idx
        stop = self.n_times if stop_idx is None else stop_idx

        if start < 0 or stop > self.n_times:
            raise ValueError('Trim indexes are outside time range')
        if start >= stop:
            raise ValueError('start_idx must be less than stop_idx')

        mask = np.zeros(self.n_times, dtype=bool)
        mask[start:stop] = True
        self._apply_time_mask(mask)

    def trim_times_by_values(
            self,
            lower: float | None = None,
            upper: float | None = None,
            ) -> None:
        """Trim all time-series by time values.

        WARNING: It is not recommended you do this after
        trial extraction or artifact detection as the time series
        will no longer fully align.

        Parameters
        ----------
        lower : float or None, default=None
            Lower time bound. If ``None``, no lower bound is applied.
        upper : float or None, default=None
            Upper time bound. If ``None``, no upper bound is applied.

        Raises
        ------
        ValueError
            If ``lower`` is greater than ``upper``.
        """
        if (lower is None) and (upper is None): return

        lower = -np.inf if lower is None else lower
        upper = np.inf if upper is None else upper

        if lower > upper:
            raise ValueError(f'Lower bound ({lower}) must be <= upper bound ({upper})')

        start = np.searchsorted(self.time, lower, side='left')
        stop = np.searchsorted(self.time, upper, side='right')

        self.trim_times_by_index(start, stop)

    def merge_events(
            self,
            to_merge: list[str],
            new_label: str,
            drop_merged: bool = True,
            ) -> None:
        """Merge multiple event labels into one.

        Parameters
        ----------
        to_merge : list[str]
            The labels of the events to merge together.
        new_label : str
            The new label for the merged event.
        drop_merged : bool, default=True
            Whether to drop the events in ``to_merge`` after merging.
            Excludes ``new_label`` if ``new_label`` in ``to_merge``.

        Raises
        ------
        KeyError
            If any in ``to_merge`` are not in ``self.event_labels``.
        """
        # validate inputs
        missing = [label for label in to_merge if label not in self.event_labels]
        if missing:
            raise KeyError(f'Event labels {missing} not found in events.')
        
        # merge timestamps and add
        merged_timestamps = np.concat([self.events[label] for label in to_merge])
        self.events[new_label] = np.sort(merged_timestamps)

        # optionally drop merged events
        if drop_merged:
            self.events = {
                label : arr for label, arr in self.events.items()
                if (label not in to_merge) or (label == new_label)
            }


    #endregion

    ########################
    #region --- PLOTTING ---
    ########################
    def plot_dashboard(
            self,
            save: str | None = None,
            raw: bool | Literal['auto'] = 'auto',
            downsample: int | None = None,
            line_kwargs: dict = {},
            theme_kwargs: dict = {},
            ) -> ggplot | Compose | None:
        """Plot or save a continuous-recording dashboard.

        Parameters
        ----------
        save : str or None, default=None
            Output path passed to ``plotnine`` when saving. If ``None``, the
            plot object is returned.
        raw : bool or {'auto'}, default='auto'
            If ``True``, plot raw traces only. If ``False``, plot raw traces,
            fitted reference, and processed signal. ``'auto'`` plots raw traces
            until preprocessing has run.
        downsample : int or None, default=None
            Downsampling factor used for plotting.
        line_kwargs : dict, default={}
            Keyword arguments forwarded to line geoms.
        theme_kwargs : dict, default={}
            Keyword arguments forwarded to the plot theme helper.

        Returns
        -------
        ggplot, plotnine.composition.Compose, or None
            Plot object when ``save`` is ``None``; otherwise ``None``.

        Raises
        ------
        ValueError
            If processed plotting is requested before preprocessing has run.
        """
        # defaults
        color_values = {
            'raw_signal' : 'tab:blue',
            'raw_isosbestic' : 'dimgrey',
            'filtered_signal' : 'tab:blue',
            'filtered_isosbestic' : 'dimgrey',
            'fitted_reference' : 'tab:orange',
            'processed_signal' : 'tab:blue'
        }

        raw_line_kwargs = line_kwargs | dict(alpha=0.8)
        process_line_kwargs = line_kwargs | dict(alpha=1.0)

        common_args = dict(
            label_col='source',
            color_values=color_values,
            theme_kwargs=theme_kwargs,
        )

        # resolve raw or not
        if raw == 'auto':
            raw = not self.has_ran_preprocess
        elif (not raw) and (not self.has_ran_preprocess):
            raise ValueError('preprocess_signal must be run before a dashboard can be made with raw == False.')

        # handle title
        title = None if self.id is None else f'Dashboard for {self.id}'

        # get long df
        long_df = self.to_long_dataframe(downsample=downsample)

        if raw:
            p: ggplot = graphing.plot_experiment_traces(
                df=long_df,
                only_traces=['raw_signal', 'raw_isosbestic'],
                title=title,
                x_label='Time (s)',
                y_label='Fluorescence intensity (a.u.)',
                legend_label='Raw',
                line_kwargs=raw_line_kwargs,
                **common_args
            )
        else:
            p_top = graphing.plot_experiment_traces(
                df=long_df,
                only_traces=['filtered_signal', 'filtered_isosbestic', 'fitted_reference'],
                title=title,
                x_label='Time (s)',
                y_label='Fluorescence intensity (a.u.)',
                legend_label='Raw',
                line_kwargs=raw_line_kwargs,
                **common_args
            )
            p_bottom = graphing.plot_experiment_traces(
                df=long_df,
                only_traces=['processed_signal'],
                title='',
                x_label='Time (s)',
                y_label=f'{self.metadata.get("correction_method", "Signal (unknown)")}',
                legend_label='Processed',
                line_kwargs=process_line_kwargs,
                **common_args
            )
            p = p_top / p_bottom

        if save is not None:
            p.save(save)
        else:
            return p

    #endregion
