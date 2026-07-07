"""Trial-wise photometry data containers and analysis helpers."""

from __future__ import annotations
from typing import Any, Callable, Literal, Self, cast
from anndata.experimental import concat_on_disk
from plotnine import ggplot

import matplotlib.pyplot as plt
import matplotlib.axes
import anndata as ad
import numpy as np
import pandas as pd
import pingouin as pg
import os

from ..analysis.peaks import PeakResult, RollingThresholdDetector, StaticThresholdDetector
from ..utils import graphing, window
from ..utils.io import clean_axis_dtype
from ..utils.operations import downsample_signal, downsample_time

ad.settings.allow_write_nullable_strings = True

class PhotometryData:
    """Handle and analyze trial-wise photometry time-series data."""

    adata: ad.AnnData

    #############################
    #region --- CONSTRUCTORS ---
    #############################
    def __init__(self, adata: ad.AnnData):
        """Initialize a `PhotometryData` object from an AnnData object.

        Parameters
        ----------
        adata : ad.AnnData
            AnnData object containing trial-wise time-series data.
        """
        assert isinstance(adata, ad.AnnData)
        self.adata = adata

    @classmethod
    def from_arrays(
            cls,
            obs: pd.DataFrame,
            data: np.ndarray,
            time_points: np.ndarray,
            layers: dict[str, np.ndarray] | None = None,
            metadata: dict[str, Any] | None = None,
            ) -> Self:
        """Construct `PhotometryData` from arrays and observation metadata.

        Parameters
        ----------
        obs : pd.DataFrame
            Per-trial observation metadata.
        data : np.ndarray
            Time-series data with shape ``(n_trials, n_times)``.
        time_points : np.ndarray
            Time axis with shape ``(n_times,)``.
        layers : dict[str, np.ndarray] or None, default=None
            Optional named layers. Each layer must match ``data.shape``.
        metadata : dict[str, Any] or None, default=None
            Optional unstructured metadata stored in ``.uns``.

        Returns
        -------
        PhotometryData
            Constructed photometry data object.
        """
        obs = obs.copy()
        obs.reset_index(drop=True, inplace=True)
        obs.index = obs.index.astype(str)
        var = pd.DataFrame({"t": time_points})
        var.index = var.index.astype(str)

        A = ad.AnnData(X=data, obs=obs, var=var, uns=metadata)
        for k, v in (layers or {}).items():
            assert v.shape == data.shape
            A.layers[k] = v
        return cls(A)

    #endregion

    #######################
    #region --- DUNDERS ---
    #######################
    def __str__(self) -> str:
        """Return a compact dataset summary."""
        out = (
            f"Photometry dataset with {self.n_trials} trials, "
            f"{self.n_times} timepoints, and {self.obs.columns.size} observations."
        )
        return out

    def __repr__(self) -> str:
        """Return the interactive representation."""
        return self.__str__()

    #endregion

    ####################################
    #region --- PROPERTIES & SETTERS ---
    ####################################
    # props
    @property
    def X(self) -> np.ndarray:
        """Primary trial-by-time signal matrix."""
        return cast(np.ndarray, self.adata.X)
    @property
    def ts(self) -> np.ndarray:
        """Time points for the signal columns."""
        return self.adata.var["t"].to_numpy()
    @property
    def obs(self) -> pd.DataFrame:
        """Per-trial observation metadata."""
        return cast(pd.DataFrame, self.adata.obs)
    @property
    def var(self) -> pd.DataFrame:
        """Per-time-point variable metadata."""
        return cast(pd.DataFrame, self.adata.var)
    @property
    def uns(self) -> dict:
        """Unstructured metadata dictionary."""
        return cast(dict, self.adata.uns)
    @property
    def n_trials(self) -> int:
        """Number of trials."""
        return self.adata.n_obs
    @property
    def n_times(self) -> int:
        """Number of time points per trial."""
        return self.adata.n_vars
    @property
    def freq(self) -> float:
        """Estimated sampling frequency for trial time points."""
        return (self.n_times - 1) / (self.ts[-1] - self.ts[0])
    @property
    def dt(self) -> float:
        """Estimated sampling interval for trial time points."""
        return 1 / self.freq

    # setters
    @X.setter
    def X(self, value: np.ndarray) -> None:
        """Set the primary signal matrix."""
        self.adata.X = value
    @obs.setter
    def obs(self, value: pd.DataFrame) -> None:
        """Set per-trial observation metadata."""
        self.adata.obs = value
    @var.setter
    def var(self, value: pd.DataFrame) -> None:
        """Set per-time-point variable metadata."""
        self.adata.var = value
    @uns.setter
    def uns(self, value: dict) -> None:
        """Set unstructured metadata."""
        self.adata.uns = value

    #endregion

    ###################
    #region --- I/O ---
    ###################
    @classmethod
    def read_h5ad(cls, path: str) -> Self:
        """Read a `PhotometryData` object from an `.h5ad` file.

        Parameters
        ----------
        path : str
            Path to the ``.h5ad`` file.

        Returns
        -------
        PhotometryData
            Loaded photometry data object.
        """
        return cls(ad.read_h5ad(path))

    @classmethod
    def read_zarr(cls, path: str) -> Self:
        """Read a `PhotometryData` object from zarr storage.

        Parameters
        ----------
        path : str
            Path to the zarr storage.

        Returns
        -------
        PhotometryData
            Loaded photometry data object.
        """
        return cls(ad.read_zarr(path))

    def write_h5ad(self, path: str, **kwargs) -> None:
        """Write the underlying AnnData to an `.h5ad` file.

        Cleans dtypes of ``.obs`` and ``.var`` before saving.

        Parameters
        ----------
        path : str
            Path to the output ``.h5ad`` file.
        kwargs :
            Keyword arguments passed to ``anndata.write_h5ad``.
        """
        convert_strings_to_categoricals = kwargs.get(
            "convert_strings_to_categoricals",
            True,
        )
        adata = self.adata.copy()
        adata.obs = clean_axis_dtype(
            adata.obs,
            convert_strings_to_categoricals=convert_strings_to_categoricals,
        )
        adata.var = clean_axis_dtype(
            adata.var,
            convert_strings_to_categoricals=convert_strings_to_categoricals,
        )
        adata.write_h5ad(path, **kwargs)

    def write_zarr(self, path: str) -> None:
        """Write the underlying AnnData to zarr storage.

        Parameters
        ----------
        path : str
            Path to the output zarr storage.
        """
        self.adata.write_zarr(path)

    def append_on_disk_h5ad(
            self, 
            path: str, 
            join: str = 'inner', 
            merge: str = 'same', 
            uns_merge: str = 'first'
            ) -> None:
        """Append this object's data to an existing `.h5ad` file on disk.

        Creates the file if it does not already exist.

        Parameters
        ----------
        path : str
            Path to the target ``.h5ad`` file.
        join : str, default='inner'
            Join strategy passed to ``anndata.experimental.concat_on_disk``.
        merge : str, default='same'
            Merge strategy for elements not aligned to the concatenated axis.
        uns_merge : str, default='first'
            Merge strategy for unstructured metadata.

        Raises
        ------
        Exception
            Re-raised with context if on-disk concatenation fails.
        """
        if not os.path.exists(path):
            self.write_h5ad(path)
            return

        # create tmp files and rename
        base, ext = os.path.splitext(path)
        tmp_path_new = base + '_new_tmp' + ext
        self.write_h5ad(tmp_path_new)

        tmp_path_old = base + '_base_tmp' + ext
        os.rename(path, tmp_path_old)

        try:
            concat_on_disk(
                in_files=[tmp_path_old, tmp_path_new],
                out_file=path,
                axis='obs',
                join=join,
                merge=merge,
                uns_merge=uns_merge,
                index_unique='-',
            )
        except Exception as e:
            os.rename(tmp_path_old, path)
            os.remove(tmp_path_new)
            raise Exception(f'In core.PhotometryData.append_on_disk_h5ad() concat_on_disk: {e}')

        os.remove(tmp_path_new)
        os.remove(tmp_path_old)
        return

    #endregion

    ##########################
    #region --- OPERATIONS ---
    ##########################
    def copy(self) -> Self:
        """Return a deep copy of this object.

        Returns
        -------
        PhotometryData
            Copied photometry data object.
        """
        return type(self)(self.adata.copy())

    def pipe(
            self,
            func: Callable[..., Self],
            *args: Any,
            **kwargs: Any,
            ) -> Self:
        """Apply a function to this object for method chaining.

        Parameters
        ----------
        func : Callable[..., PhotometryData]
            Function that accepts this object as its first argument.
        *args : Any
            Additional positional arguments passed to ``func``.
        **kwargs : Any
            Additional keyword arguments passed to ``func``.

        Returns
        -------
        PhotometryData
            Output returned by ``func``.
        """
        return func(self, *args, **kwargs)

    def combine_obj(
            self,
            to_append: "PhotometryData" | list["PhotometryData"],
            inplace: bool = False,
            join: str = 'inner',
            merge: str ='same',
            uns_merge: str = 'same',
            check_time: bool = False,
            time_tol: float = 1e-6,
            ) -> None | Self:
        """Concatenate this object with one or more `PhotometryData` objects.

        Parameters
        ----------
        to_append : PhotometryData or list[PhotometryData]
            Object or objects to append along the trial axis.
        inplace : bool, default=False
            If ``True``, replace this object's AnnData with the merged result.
            If ``False``, return a new object.
        join : str, default='inner'
            Join strategy passed to ``anndata.concat``.
        merge : str, default='same'
            Merge strategy for elements not aligned to the concatenated axis.
        uns_merge : str, default='same'
            Merge strategy for unstructured metadata.
        check_time : bool, default=False
            Whether to check if all times are close. If ``True``, an error 
            will be thrown if the times of ``to_append`` are outside of ``time_tol``
            when compared to ``self.ts``.
        time_tol : float, default=1e-6
            The absolute tolerance for close time series. The combined object will
            inherit ``self.ts``.

        Returns
        -------
        PhotometryData or None
            Combined object when ``inplace`` is ``False``; otherwise ``None``.
        """
        if not isinstance(to_append, list):
            to_append = [to_append]

        adatas = [self.adata.copy()] + [obj.adata.copy() for obj in to_append]

        offset = 0
        for adata in adatas:
            adata.obs = adata.obs.infer_objects()
            adata.obs_names = pd.Index(
                np.arange(offset, offset + adata.n_obs, dtype=int).astype(str)
            )
            adata.var_names = adata.var_names.astype(str)
            offset += adata.n_obs

            # check for time misalignment
            if check_time and ('t' in adata.var):
                if not np.allclose(self.ts, adata.var['t'].to_numpy(), atol=time_tol, rtol=0):
                    misalignment = np.max(self.ts - adata.var['t'])
                    raise ValueError(f'Time-series misalignment ({misalignment}) is greater than time_tol ({time_tol})')

        merged_adata = ad.concat(
            adatas=adatas,
            axis="obs",
            join=join,
            merge=merge,
            uns_merge=uns_merge,
            index_unique=None,
        )

        # convert to string index
        merged_adata.obs_names = pd.Index(
            np.arange(merged_adata.n_obs, dtype=int).astype(str)
        )

        # inherit time series
        merged_adata.var['t'] = self.var['t']

        if inplace:
            self.adata = merged_adata
            return
        else:
            return type(self)(merged_adata)

    def downsample(
            self,
            factor: int | None,
            method: Literal['mean', 'resample'] = 'mean',
            upsample: int = 1,
            window: str | tuple | np.ndarray = ('kaiser', 5),
            padtype: str = 'line',
            cval: float | None = None
            ) -> Self:
        """Downsample the time dimension of the dataset.

        Parameters
        ----------
        factor : int or None
            Downsampling factor applied along the time axis. ``None``, ``0``,
            and ``1`` return ``self``.
        method : {'mean', 'resample'}, default='mean'
            Downsampling method passed to the signal and time helpers.
        upsample : int, default=1
            Upsampling factor used by the resampling method.
        window : str, tuple, or np.ndarray, default=('kaiser', 5)
            Resampling window passed to the signal helper.
        padtype : str, default='line'
            Padding mode passed to the signal helper.
        cval : float or None, default=None
            Constant value used by padding modes that require one.

        Returns
        -------
        PhotometryData
            Downsampled photometry data object, or ``self`` when no
            downsampling is requested.
        """

        if factor in (None, 0, 1):
            return self

        # downsample signals
        X_new = downsample_signal(
            self.adata.X,
            factor=factor,
            axis=1,
            method=method,
            upsample=upsample,
            window=window,
            padtype=padtype,
            cval=cval,
        )
        layers_new = {
            k : downsample_signal(
                v,
                factor=factor,
                axis=1,
                method=method,
                upsample=upsample,
                window=window,
                padtype=padtype,
                cval=cval,
            )
            for k, v in self.adata.layers.items()
        }

        # downsample time
        t_new = downsample_time(
            self.var['t'].to_numpy(),
            factor=factor,
            method=method,
            upsample=upsample,
        )

        # construct and return new object with inherited metadata
        metadata = self.uns.copy()
        if "frequency" in metadata:
            metadata["frequency"] = metadata["frequency"] * upsample / factor
        return type(self).from_arrays(
            obs=self.obs,
            data=X_new,
            time_points=t_new,
            layers=layers_new,
            metadata=metadata,
        )

    def _agg(
            self,
            method: Callable[..., np.ndarray],
            group_on: list[str] | None,
            data_cols: list[str],
            collapse_cols: list[str] | None = None,
            count_col: str | None = None
            ) -> tuple[pd.DataFrame, np.ndarray]:
        """Aggregate signals and observation columns over groups."""
        X = np.asarray(self.adata.X)
        obs: pd.DataFrame = self.adata.obs

        if isinstance(group_on, str):
            group_on = [group_on]

        if group_on is None or len(group_on) == 0:
            new_cols = data_cols if count_col is None else [count_col] + data_cols

            X_agg = method(X, axis=0)[np.newaxis, :]
            obs_agg = pd.DataFrame(columns=new_cols, index=[0])
            obs_agg.loc[0, data_cols] = method(obs.loc[:, data_cols], axis=0)
            if count_col is not None: obs_agg.loc[0, count_col] = obs.index.size
            if collapse_cols is not None:
                for col in collapse_cols:
                    obs_agg.loc[0, col] = obs[col].to_list()

        else:
            groups = obs.groupby(group_on, sort=False, observed=True).indices # observed needs to be True
            n_groups = len(groups)
            new_cols = group_on + data_cols if count_col is None else group_on + [count_col] + data_cols
            if collapse_cols is not None:
                new_cols = new_cols + collapse_cols

            X_agg = np.empty((n_groups, X.shape[1]), dtype=X.dtype)
            obs_agg = pd.DataFrame(columns=new_cols, index=np.arange(n_groups))

            for i, (gkey, idxs) in enumerate(groups.items()):
                X_agg[i] = method(X[idxs], axis=0)
                obs_agg.loc[i, group_on] = gkey
                obs_agg.loc[i, data_cols] = method(obs.iloc[idxs][data_cols], axis=0)
                if count_col is not None: obs_agg.loc[i, count_col] = len(idxs)
                if collapse_cols is not None:
                    for col in collapse_cols:
                        obs_agg.loc[i, col] = obs.iloc[idxs][col].to_list()


        # clean dtypes
        obs_agg = obs_agg.infer_objects()

        return obs_agg, X_agg

    def collapse(
            self,
            group_on: str | list[str] | None = None,
            method: Callable = np.nanmean,
            metrics: dict[str, Callable] = {'std':np.std},
            data_cols: list[str] = [],
            collapse_cols: list[str] | None = None,
            count_col: str | None = 'n'
            ) -> Self:
        """Collapse trials by grouping `obs` and aggregating data.

        Parameters
        ----------
        group_on : str, list[str], or None, default=None
            Observation columns used to define groups. If ``None`` or empty,
            all trials are collapsed into one row.
        method : Callable, default=np.nanmean
            Aggregation function for the primary ``X`` matrix.
        metrics : dict[str, Callable], default={'std': np.std}
            Additional aggregation functions stored as layers. Metric keys are
            also appended to aggregated ``data_cols`` column names.
        data_cols : list[str], default=[]
            Observation columns to aggregate with ``method`` and ``metrics``.
        collapse_cols : list[str] or None, default=None
            Observation columns to collapse into lists.
        count_col : str or None, default='n'
            Optional output column containing group counts.

        Returns
        -------
        PhotometryData
            Collapsed photometry data object.
        """
        obs_agg, X_agg = self._agg(method=method, group_on=group_on, data_cols=data_cols, count_col=count_col, collapse_cols=collapse_cols)

        layers = {}
        for key, func in metrics.items():
            obs_lay, X_lay = self._agg(method=func, group_on=group_on, data_cols=data_cols, count_col=count_col, collapse_cols=None)
            layers[key] = X_lay
            obs_agg = obs_agg.join(obs_lay[data_cols], rsuffix='_' + str(key))

        new_obj = type(self).from_arrays(
            obs=obs_agg,
            data=X_agg,
            time_points=self.adata.var['t'],
            layers=layers,
            metadata=self.adata.uns
        )
        return new_obj

    def window(
            self,
            centers: np.ndarray | float | str,
            bounds: tuple[float, float],
            event_cols: list[str] | None = None,
            strategy: Literal['interp', 'nearest'] = 'nearest',
            invalid_window_policy: Literal['drop', 'error'] = 'error',
            verbose: bool = False,
            ) -> Self:
        """Return a new `PhotometryData` object with windowed time series.

        Parameters
        ----------
        centers : np.ndarray, float, or str
            Window centers. Scalars are broadcast to all trials; strings are
            interpreted as numeric columns in ``obs``; arrays must align with
            trial rows.
        bounds : tuple[float, float]
            Lower and upper bounds relative to each center.
        event_cols : list[str] or None, default=None
            Observation columns containing event times to recenter relative to
            ``centers`` in the output object.
        strategy : {'interp', 'nearest'}, default='nearest'
            Windowing strategy. ``'nearest'`` rounds to sampled time points;
            ``'interp'`` interpolates signals onto an exact centered grid.
        invalid_window_policy : {'drop', 'error'}, default='error'
            Policy for windows that extend outside the available time range.
        verbose : bool, default=False
            If ``True``, print indexes dropped by ``invalid_window_policy='drop'``.

        Returns
        -------
        PhotometryData
            Windowed photometry data object.

        Raises
        ------
        ValueError
            If center column or strategy is unknown, or if invalid windows are
            present under ``invalid_window_policy='error'``.
        """
        # coerce centers input
        if isinstance(centers, (float, int)):
            centers = np.full(shape=self.X.shape[0], fill_value=centers, dtype=float)

        elif isinstance(centers, str):
            if centers not in self.obs: raise ValueError(f"Centers {centers} not a column in self.obs.")
            centers = self.obs[centers].to_numpy().astype(float)

        centers = np.asarray(centers, dtype=float)

        # convert layers and events
        layers = {key: self.get_layer(key).copy() for key in self.adata.layers.keys()}
        events = None if event_cols is None else {col: self.obs[col].to_numpy() for col in event_cols}

        # execute windowing
        match strategy:
            case 'nearest':
                windows = window.create_windows_nearest_2D(
                    signal=self.X,
                    time=self.ts,
                    centers=centers,
                    bounds=bounds,
                    freq=self.freq,
                    events=events,
                    layers=layers,
                )
            case 'interp':
                windows = window.create_windows_interp_2D(
                    signal=self.X,
                    time=self.ts,
                    centers=centers,
                    bounds=bounds,
                    freq=self.freq,
                    events=events,
                    layers=layers,
                )
            case _:
                raise ValueError(f'Window strategy {strategy} not recognized.')

        # create new obs
        new_obs = self.obs.copy()
        if event_cols is not None:
            new_obs = new_obs.assign(**windows.events)

        # handle invalid windows
        if windows.invalid_mask.any():
            invalid_idxs = np.flatnonzero(windows.invalid_mask).tolist()
            keep_mask = ~windows.invalid_mask

            if invalid_window_policy == 'error':
                raise ValueError(f'Invalid trial windows that extend outside of time range at {invalid_idxs}.')

            elif invalid_window_policy == 'drop':
                # importantly apply mask to windows AND new obs
                new_obs = new_obs.loc[keep_mask, :]
                windows.apply_mask(keep_mask)
                if verbose: print(f'Dropped trials at indexes {invalid_idxs} with invalid windows.')

        # package result
        out = type(self).from_arrays(
            obs=new_obs,
            data=windows.signals,
            time_points=windows.time_grid,
            layers=windows.layers,
            metadata=self.uns.copy(),
        )

        return out

    #endregion

    ###############################
    #region --- SIGNAL ANALYSIS ---
    ###############################
    def difference(
            self,
            n: int = 1,
            ) -> np.ndarray:
        """Calculate the n-th discrete difference of each trial.

        Parameters
        ----------
        n : int, default=1
            Number of differencing passes to apply along the time axis.

        Returns
        -------
        np.ndarray
            Difference array with shape ``(n_trials, n_times)``.
        """
        arr = self.X.copy()
        for i in range(n):
            arr = np.diff(arr, axis=1, prepend=arr[:, 0][:, np.newaxis])
        return arr

    def area_under_curve(
            self,
            centers: np.ndarray | float | None = None,
            bounds: tuple[float, float] | None = None,
            transformation: Callable | None = None
            ) -> np.ndarray:
        """Calculate area under the curve for each trial.

        Optionally calculate the area only on specific windows.

        Parameters
        ----------
        centers : np.ndarray, float, or None, default=None
            Window centers. If ``None`` or if ``bounds`` is ``None``, the full
            trial duration is integrated.
        bounds : tuple[float, float] or None, default=None
            Window bounds relative to ``centers``. If ``None`` or if
            ``centers`` is ``None``, the full trial duration is integrated.
        transformation : Callable or None, default=None
            Optional transformation applied to the signal before integration.
            The callable must return an array with the same shape as the input.

        Returns
        -------
        np.ndarray
            Area values with one value per trial.
        """
        # window signals if necessary
        if centers is None or bounds is None:
            y = self.X
            x = self.ts
        else:
            windows = self.window(centers=centers, bounds=bounds)
            y = windows.X
            x = windows.ts

        # transform signals if specified
        if transformation is not None:
            y = transformation(y)

        return np.trapezoid(y=y, x=x, axis=1)

    def detect_peaks_static_threshold(
            self,
            center_method: Literal['median', 'mean', 'zeros'] | Callable = 'median',
            scale_method: Literal['mad', 'std', 'ones'] | Callable = 'mad',
            test_magnitude: float = 3.0,
            baselines: Self | np.ndarray | None = None,
            min_distance_sec: float | None = None,
            min_duration_sec: float | None = None,
            max_duration_sec: float | None = None,
            direction: Literal['positive', 'negative', 'both'] = 'both',
            detailed: bool = False,
            ) -> PeakResult:
        """Detect peaks using a static per-trial threshold.

        Parameters
        ----------
        center_method : {'median', 'mean', 'zeros'} or Callable, default='median'
            Method used by the detector to estimate baseline center.
        scale_method : {'mad', 'std', 'ones'} or Callable, default='mad'
            Method used by the detector to estimate baseline scale.
        test_magnitude : float, default=3.0
            Threshold multiplier applied to the scale estimate.
        baselines : PhotometryData, np.ndarray, or None, default=None
            Optional baseline signals. A single-row array is tiled across
            trials; otherwise the first dimension must match ``n_trials``.
        min_distance_sec : float or None, default=None
            Minimum distance between peaks in seconds.
        min_duration_sec : float or None, default=None
            Minimum peak duration in seconds.
        max_duration_sec : float or None, default=None
            Maximum peak duration in seconds.
        direction : {'positive', 'negative', 'both'}, default='both'
            Peak direction to detect.
        detailed : bool, default=False
            Whether to return detailed detector output.

        Returns
        -------
        PeakResult
            Detected peaks.

        Raises
        ------
        ValueError
            If provided baselines do not have one row or ``n_trials`` rows.
        """
        # coerce baselines input
        if baselines is not None:
            if isinstance(baselines, PhotometryData):
                baselines = baselines.X

            if baselines.shape[0] == 1:
                baselines = np.tile(baselines, (self.n_trials, 1))

            if baselines.shape[0] != self.n_trials:
                raise ValueError(
                    f'Baselines must have same number of rows as self.n_trials ({self.n_trials}) or 1, is {baselines.shape[0]}'
                )

        # initialize detector
        detector = StaticThresholdDetector(
            center_method=center_method,
            scale_method=scale_method,
            test_magnitude=test_magnitude,
        )

        # execute
        result = detector.detect(
            signals=self.X,
            time=self.ts,
            frequency=self.freq,
            baselines=baselines,
            min_distance_sec=min_distance_sec,
            min_duration_sec=min_duration_sec,
            max_duration_sec=max_duration_sec,
            direction=direction,
            detailed=detailed,
        )

        # check for empty, format, and return
        if result.empty:
            print('No peaks detected!')

        return result

    def detect_peaks_rolling_threshold(
            self,
            window_width_sec: float = 5.0,
            center_method: Literal['median', 'mean', 'zeros'] | Callable = 'median',
            scale_method: Literal['mad', 'std', 'ones'] | Callable = 'mad',
            test_magnitude: float = 3.0,
            min_distance_sec: float | None = None,
            min_duration_sec: float | None = None,
            max_duration_sec: float | None = None,
            direction: Literal['positive', 'negative', 'both'] = 'both',
            detailed: bool = False,
            ) -> PeakResult:
        """Detect peaks using a rolling threshold.

        Parameters
        ----------
        window_width_sec : float, default=5.0
            Width of the rolling threshold window in seconds.
        center_method : {'median', 'mean', 'zeros'} or Callable, default='median'
            Method used by the detector to estimate rolling center.
        scale_method : {'mad', 'std', 'ones'} or Callable, default='mad'
            Method used by the detector to estimate rolling scale.
        test_magnitude : float, default=3.0
            Threshold multiplier applied to the rolling scale estimate.
        min_distance_sec : float or None, default=None
            Minimum distance between peaks in seconds.
        min_duration_sec : float or None, default=None
            Minimum peak duration in seconds.
        max_duration_sec : float or None, default=None
            Maximum peak duration in seconds.
        direction : {'positive', 'negative', 'both'}, default='both'
            Peak direction to detect.
        detailed : bool, default=False
            Whether to return detailed detector output.

        Returns
        -------
        PeakResult
            Detected peaks.
        """
        # initialize detector
        detector = RollingThresholdDetector(
            window_width_sec=window_width_sec,
            center_method=center_method,
            scale_method=scale_method,
            test_magnitude=test_magnitude,
        )

        # execute
        result = detector.detect(
            signals=self.X,
            time=self.ts,
            frequency=self.freq,
            baselines=None,
            min_distance_sec=min_distance_sec,
            min_duration_sec=min_duration_sec,
            max_duration_sec=max_duration_sec,
            direction=direction,
            detailed=detailed,
        )

        # check for empty, format, and return
        if result.empty:
            print('No peaks detected!')

        return result

    #endregion

    #################################
    #region --- METADATA ANALYSIS ---
    #################################
    def ANOVA(
            self,
            dependent_var: str,
            between: str | list[str],
            ss_type: Literal[1, 2, 3] = 2,
            detailed: bool = True,
            effsize: Literal['np2', 'n2'] = 'np2',
            ) -> pd.DataFrame:
        """One-way and *N*-way ANOVA.

        Parameters
        ----------
        dependent_var : str
            Observation column containing the dependent variable.
        between : str or list[str]
            Observation column or columns containing between-subject factors.
        ss_type : {1, 2, 3}, default=2
            Sums-of-squares type for unbalanced designs with two or more
            factors.
        detailed : bool, default=True
            If ``True``, return a detailed ANOVA table.
        effsize : {'np2', 'n2'}, default='np2'
            Effect-size column requested from ``pingouin.anova``.

        Returns
        -------
        pd.DataFrame
            ANOVA summary returned by ``pingouin.anova``.
        """
        aov = pg.anova(
            self.obs,
            dv=dependent_var,
            between=between,
            ss_type=ss_type,
            detailed=detailed,
            effsize=effsize,
        )
        return aov

    def ANOVA_rm(
            self,
            dependent_var: str,
            within: str | list[str],
            subject: str,
            correction: bool | Literal['auto'] = 'auto',
            detailed: bool = True,
            effsize: Literal['np2', 'n2', 'ng2'] = 'ng2',
            ) -> pd.DataFrame:
        """One-way and two-way repeated measures ANOVA.

        Parameters
        ----------
        dependent_var : str
            Observation column containing the dependent variable.
        within : str or list[str]
            Observation column or columns containing within-subject factors.
        subject : str
            Observation column containing subject identifiers.
        correction : bool or {'auto'}, default='auto'
            Greenhouse-Geisser correction option passed to
            ``pingouin.rm_anova``.
        detailed : bool, default=True
            If ``True``, return a full ANOVA table.
        effsize : {'np2', 'n2', 'ng2'}, default='ng2'
            Effect-size column requested from ``pingouin.rm_anova``.

        Returns
        -------
        pd.DataFrame
            Repeated-measures ANOVA summary returned by ``pingouin.rm_anova``.
        """
        aov = pg.rm_anova(
            self.obs,
            dv=dependent_var,
            within=within,
            subject=subject,
            correction=correction,
            detailed=detailed,
            effsize=effsize,
        )
        return aov

    def ANOVA_mixed(
            self,
            dependent_var: str,
            between: str,
            within: str,
            subject: str,
            correction: bool | Literal['auto'] = 'auto',
            effsize: Literal['np2', 'n2', 'ng2'] = 'np2',
            ) -> pd.DataFrame:
        """Mixed-design ANOVA.

        Parameters
        ----------
        dependent_var : str
            Observation column containing the dependent variable.
        between : str
            Observation column containing the between-subject factor.
        within : str
            Observation column containing the within-subject factor.
        subject : str
            Observation column containing subject identifiers.
        correction : bool or {'auto'}, default='auto'
            Greenhouse-Geisser correction option passed to
            ``pingouin.mixed_anova``.
        effsize : {'np2', 'n2', 'ng2'}, default='np2'
            Effect-size column requested from ``pingouin.mixed_anova``.

        Returns
        -------
        pd.DataFrame
            Mixed-design ANOVA summary returned by ``pingouin.mixed_anova``.
        """
        aov = pg.mixed_anova(
            self.obs,
            dv=dependent_var,
            within=within,
            subject=subject,
            between=between,
            correction=correction,
            effsize=effsize,
        )
        return aov
    #endregion

    ############################
    #region --- CONVIENIENCE ---
    ############################
    def print_info(self) -> None:
        """Print a short summary of the dataset."""
        print(
            f"Photometry dataset with {self.n_trials} trials, "
            f"{self.n_times} timepoints, and {self.obs.columns.size} observations"
        )

    def get_layer(self, key: str) -> np.ndarray:
        """Get a layer from the underlying AnnData object.

        Parameters
        ----------
        key : str
            Layer name.

        Returns
        -------
        np.ndarray
            Requested layer data.
        """
        return cast(np.ndarray, self.adata.layers[key])

    def filter_rows(self, sel: np.ndarray | int | None, inplace: bool = False) -> None | Self:
        """Filter rows (trials) using indexes or boolean mask.

        Parameters
        ----------
        sel : np.ndarray, int, or None
            Boolean mask, integer indexer, or ``None``. ``None`` leaves the
            object unchanged.
        inplace : bool, default=False
            If ``True``, modify this object. If ``False``, return a filtered
            copy.

        Returns
        -------
        PhotometryData or None
            Filtered object when ``inplace`` is ``False``; otherwise ``None``.
        """
        if inplace:
            if sel is None: return
            self.adata = self.adata[sel, :].copy()
        else:
            if sel is None: return self
            return type(self)(self.adata[sel, :].copy())

    def mutate_obs(self, **columns) -> Self:
        """Return a copy with mutated observation columns.

        Parameters
        ----------
        **columns : Any
            Column values or callables. Callables receive the copied
            `PhotometryData` object and should return the column values to
            assign. Values of ``None`` are skipped.

        Returns
        -------
        PhotometryData
            New object with updated ``obs`` columns.
        """
        out = self.copy()
        for name, value in columns.items():
            to_assign = value(out) if callable(value) else value
            if to_assign is None: continue
            out.obs[name] = to_assign

        return out

    def add_obs_columns(self, add_from: dict[str, Any], keys: list[str] | None = None) -> None:
        """Add columns to `obs` from a dictionary.

        Parameters
        ----------
        add_from : dict[str, Any]
            Mapping from column names to values.
        keys : list[str] or None, default=None
            Keys from ``add_from`` to add. If ``None``, all keys are used.
        """
        keys = list(add_from.keys()) if keys is None else keys
        for k in keys:
            self.adata.obs[k] = add_from.get(k, None)

    def add_metadata(self, add_from: dict[str, Any], keys: list[str] | None = None) -> None:
        """Add entries to the `.uns` metadata dictionary.

        Parameters
        ----------
        add_from : dict[str, Any]
            Mapping from metadata keys to values.
        keys : list[str] or None, default=None
            Keys from ``add_from`` to add. If ``None``, all keys are used.
        """
        keys = list(add_from.keys()) if keys is None else keys
        for k in keys:
            self.adata.uns[k] = add_from.get(k, None)

    def drop_obs_columns(self, to_drop: list[str]) -> None:
        """Drop observation columns from `obs`.

        Parameters
        ----------
        to_drop : list[str]
            Column names to drop. Missing columns are ignored.
        """
        self.adata.obs = self.obs.drop(to_drop, errors='ignore', axis=1)

    def get_text_value_counts(self, col: str) -> str:
        """Get a string summary of value counts for a column in `obs`.

        Parameters
        ----------
        col : str
            Column name in ``obs``.

        Returns
        -------
        str
            Comma-separated summary of value counts.
        """
        vc = self.adata.obs[col].value_counts(dropna=False)
        return ", ".join(f"{k}: {v}" for k, v in vc.items())

    #endregion

    ######################
    #region --- EXPORT ---
    ######################
    def trials_to_long_df(
            self,
            layer: str | None = None,
            err_layer: str | None = None,
            obs_cols: list[str] | str | None = None,
            downsample: int | None = None,
            ) -> pd.DataFrame:
        """Translate trial data to long DataFrame format.

        Mostly useful for graphing.

        Parameters
        ----------
        layer : str or None, default=None
            Layer to export. If ``None``, ``self.X`` is used.
        err_layer : str or None, default=None
            Optional layer to include as an error column.
        obs_cols : list[str], str, or None, default=None
            Observation columns to repeat into the long dataframe.
        downsample : int or None, default=None
            Downsampling factor applied before export.

        Returns
        -------
        pd.DataFrame
            Long dataframe containing trial indexes, time indexes, signal
            values, time values, selected observation columns, and optional
            error values.
        """
        obj = self if downsample is None else self.downsample(downsample)
        X = obj.adata.layers[layer] if layer is not None else obj.adata.X

        if isinstance(obs_cols, str):
            obs_cols = [obs_cols]

        def _make_long_fast(arr: np.ndarray, obj: PhotometryData, obs_cols: list[str] | None = None, value_name: str = 'signal') -> pd.DataFrame:
                """Build a long dataframe from one trial-by-time array."""
                n_trials, n_time = arr.shape

                trial_ids = obj.adata.obs_names.to_numpy()

                out = {
                    "trial_idx": np.repeat(trial_ids, n_time),
                    "time_idx": np.tile(np.arange(n_time), n_trials),
                    value_name: np.asarray(arr).reshape(-1),
                }

                if obs_cols is not None:
                    obs_df = obj.adata.obs[obs_cols]
                    for col in obs_cols:
                        out[col] = np.repeat(obs_df[col].to_numpy(), n_time)

                return pd.DataFrame(out)

        long_signal = _make_long_fast(X, obj=obj, obs_cols=obs_cols)

        long_signal["time_idx"] = long_signal["time_idx"].astype(int)
        long_signal["time"] = obj.ts[long_signal["time_idx"]]

        if err_layer is not None:
            E = obj.adata.layers[err_layer]
            long_err = _make_long_fast(E, obj=obj, obs_cols=None, value_name=err_layer)
            long_signal = long_signal.join(long_err.filter([err_layer]), how='left')

        return long_signal

    def trials_to_wide_df(
            self,
            layer: str | None = None,
            obs_cols: list[str] | None = None,
            signal_prefix: str = 'photometry',
            downsample: int | None = None,
            downsample_kwargs: dict = {}
            ) -> pd.DataFrame:
        """Translate trial data to wide DataFrame format.

        Mostly useful for exporting to ``analysis.FMM`` module.

        Parameters
        ----------
        layer : str or None, default=None
            Layer to export. If ``None``, ``self.X`` is used.
        obs_cols : list[str] or None, default=None
            Observation columns to include. If ``None``, all observation
            columns are included.
        signal_prefix : str, default='photometry'
            Prefix used for signal columns.
        downsample : int or None, default=None
            Downsampling factor applied before export.
        downsample_kwargs : dict, default={}
            Additional keyword arguments passed to `downsample_signal`.

        Returns
        -------
        pd.DataFrame
            Wide dataframe containing observation columns followed by one
            signal column per exported time point.
        """
        if layer is None:
            signal = downsample_signal(self.X, downsample, axis=1, **downsample_kwargs)
        else:
            signal = downsample_signal(self.get_layer(layer), downsample, axis=1, **downsample_kwargs)

        sig_columns = signal_prefix + '.' + np.arange(1, signal.shape[1] + 1).astype(str)
        sig_df = pd.DataFrame(signal, columns=sig_columns)

        obs_df = self.obs.copy() if obs_cols is None else self.obs.filter(obs_cols).copy()

        sig_df.reset_index(drop=True, inplace=True)
        obs_df.reset_index(drop=True, inplace=True)

        return pd.concat([obs_df, sig_df], axis=1)

    #endregion

    ########################
    #region --- PLOTTING ---
    ########################
    def _make_labels(self, label_with: list[str] | str | None) -> pd.Series:
        """Build display labels from observation columns."""
        if label_with is None:
            return pd.Series([None] * self.n_trials, index=self.obs.index)

        if isinstance(label_with, str):
            label_with = [label_with]

        missing = [col for col in label_with if col not in self.obs.columns]
        if missing:
            raise KeyError(f"Columns not found in obs: {missing}")

        labels = pd.Series("", index=self.obs.index, dtype="object")

        for i, col in enumerate(label_with):
            part = col + ": " + self.obs[col].astype(str)
            labels = part if i == 0 else labels + ", " + part

        return labels

    def _make_group_labels(self, group_on: list[str] | str | None) -> pd.Series:
        """Build grouping labels from observation columns."""
        if group_on is None:
            return pd.Series([None] * self.n_trials, index=self.obs.index)

        if isinstance(group_on, str):
            group_on = [group_on]

        glabels = self.obs[group_on[0]].astype(str)
        for i in range(1, len(group_on)):
            glabels = glabels + '_' + self.obs[group_on[i]].astype(str)

        return glabels

    def plot_trials(
            self,
            sel: np.ndarray | int | None = None,
            label_with: list[str] | str | None = None,
            group_on: list[str] | str | None = None,
            err_layer: str | None = None,
            downsample: int | None = None,
            line_kwargs: dict = {},
            ribbon_kwargs: dict = {},
            theme_kwargs: dict = {},
            ) -> ggplot:
        """Plot trial traces as a `plotnine` object.

        Parameters
        ----------
        sel : np.ndarray, int, or None, default=None
            Row selector applied before plotting.
        label_with : list[str], str, or None, default=None
            Observation columns used to build legend labels.
        group_on : list[str], str, or None, default=None
            Observation columns used to define line groups.
        err_layer : str or None, default=None
            Optional layer plotted as an error ribbon.
        downsample : int or None, default=None
            Downsampling factor applied before plotting.
        line_kwargs : dict, default={}
            Keyword arguments forwarded to line geoms.
        ribbon_kwargs : dict, default={}
            Keyword arguments forwarded to ribbon geoms.
        theme_kwargs : dict, default={}
            Keyword arguments forwarded to the plot theme helper.

        Returns
        -------
        ggplot
            Plot object for the selected trials.
        """
        # handle None inputs
        label_col = None if label_with is None else 'label'
        group_col = None if group_on is None else 'group'
        obs_cols = [col for col in [label_col, group_col] if col is not None]

        # make long df grapher
        long_df = (
            self
            .filter_rows(sel, inplace=False)
            .mutate_obs(
                label = lambda data: data._make_labels(label_with),
                group = lambda data: data._make_group_labels(group_on),
            )
            .trials_to_long_df(err_layer=err_layer, downsample=downsample, obs_cols=obs_cols)
        )

        # make graph
        p = graphing.plot_photometry_data(
            long_df=long_df,
            label_col=label_col,
            group_col=group_col,
            err_layer=err_layer,
            line_kwargs=line_kwargs,
            ribbon_kwargs=ribbon_kwargs,
            theme_kwargs=theme_kwargs,
        )

        return p

    #endregion
