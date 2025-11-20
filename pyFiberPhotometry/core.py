from __future__ import annotations
from typing import Dict, Any, List, Callable, Tuple, Literal
from matplotlib.axes import Axes
from anndata.experimental import concat_on_disk
from scipy.optimize import curve_fit
from scipy.signal import butter, sosfiltfilt
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error

import matplotlib.pyplot as plt
import anndata as ad
import numpy as np
import pandas as pd
import tdt
import os

from .utils import *
from .io import *

class PhotometryData:
    """
    Wrap an AnnData object for trial-wise photometry time-series data.
    """
    adata: ad.AnnData

    # --- constructors ---
    def __init__(self, adata):
        """
        Initialize a PhotometryData wrapper from an AnnData object.
        Args:
            adata (anndata.AnnData): AnnData object containing time-series data.
        Returns:
            None
        """
        assert isinstance(adata, ad.AnnData)
        self.adata = adata
    
    @classmethod
    def from_arrays(
        cls,
        obs: pd.DataFrame, 
        data: np.ndarray, 
        time_points: np.ndarray,
        layers: Dict[str, np.ndarray] | None = None,
        metadata: Dict[str, Any] | None = None,
        ) -> "PhotometryData":
        """
        Construct PhotometryData from arrays and observation metadata.
        Args:
            obs (pd.DataFrame): Per-trial observation metadata.
            data (np.ndarray): Time-series data of shape (n_trials, n_time).
            time_points (np.ndarray): Time axis of shape (n_time,).
            layers (dict[str, np.ndarray] | None): Optional named layers matching shape of data.
            metadata (dict[str, Any] | None): Optional unstructured metadata stored in .uns.
        Returns:
            PhotometryData
        """
        obs = obs.copy()
        obs.index = obs.index.astype(str)
        var = pd.DataFrame({"t": time_points})
        var.index = var.index.astype(str)

        A = ad.AnnData(X=data, obs=obs, var=var, uns=metadata)
        for k, v in (layers or {}).items():
            assert v.shape == data.shape
            A.layers[k] = v
        return cls(A)
    
    # --- I/O ---
    @classmethod
    def read_h5ad(cls, path: str) -> "PhotometryData":
        """
        Read a PhotometryData object from an .h5ad file.
        Args:
            path (str): Path to the .h5ad file.
        Returns:
            PhotometryData: Loaded PhotometryData instance.
        """        
        return cls(ad.read_h5ad(path))
    @classmethod
    def read_zarr(cls, path: str) -> "PhotometryData":
        """
        Read a PhotometryData object from zarr storage.
        Args:
            path (str): Path to the zarr storage.
        Returns:
            PhotometryData: Loaded PhotometryData instance.
        """      
        return cls(ad.read_zarr(path))
    
    def write_h5ad(self, path: str) -> None:
        """
        Write the underlying AnnData to an .h5ad file.
        Args:
            path (str): Path to the output .h5ad file.
        Returns:
            None
        """        
        self.adata.write_h5ad(path)
    def write_zarr(self, path: str) -> None:
        """
        Write the underlying AnnData to zarr storage.
        Args:
            path (str): Path to the output zarr storage.
        Returns:
            None
        """     
        self.adata.write_zarr(path)

    def append_on_disk_h5ad(self, path: str) -> None:
        """
        Append this object's data to an existing .h5ad file on disk or create it if it does not exist.
        Args:
            path (str): Path to the target .h5ad file.
        Returns:
            None
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
                join='inner',
                merge='same',
                uns_merge='same',
            )
        except Exception as e:
            os.rename(tmp_path_old, path)
            os.remove(tmp_path_new)
            raise Exception(f'In core.PhotometryData.append_on_disk_h5ad() concat_on_disk: {e}')
        
        os.remove(tmp_path_new)
        os.remove(tmp_path_old)
        return

    # --- convenience views ---
    @property
    def X(self) -> np.ndarray: return self.adata.X
    @property
    def ts(self) -> np.ndarray: return self.adata.var["t"].to_numpy()
    @property
    def obs(self) -> pd.DataFrame: return self.adata.obs
    @property
    def var(self) -> pd.DataFrame: return self.adata.var
    @property
    def n_trials(self) -> int: return self.adata.n_obs
    @property
    def n_times(self) -> int: return self.adata.n_vars

    # --- hidden functions ---
    def _agg(
            self, 
            method: Callable[..., np.ndarray], 
            group_on: List[str], 
            data_cols: List[str], 
            count_col: str | None = None
            ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Aggregate X and selected obs columns over groups.
        Args:
            method (callable): Aggregation function applied along axis 0.
            group_on (list[str]): Columns in obs used to define groups.
            data_cols (list[str]): Observation columns to aggregate.
            count_col (str | None): Optional column name to store group counts.
        Returns:
            tuple[pd.DataFrame, np.ndarray]: Aggregated obs and X arrays.
        """
        X = np.asarray(self.adata.X)
        obs = self.adata.obs

        groups = obs.groupby(group_on, sort=False, observed=False).indices
        n_groups = len(groups)
        new_cols = group_on + data_cols if count_col is None else group_on + [count_col] + data_cols
        
        X_agg = np.empty((n_groups, X.shape[1]), dtype=X.dtype)
        obs_agg = pd.DataFrame(columns=new_cols, index=range(n_groups))

        for i, (gkey, idxs) in enumerate(groups.items()):
            X_agg[i] = method(X[idxs], axis=0)
            obs_agg.loc[i, group_on] = gkey
            obs_agg.loc[i, data_cols] = method(obs.iloc[idxs][data_cols], axis=0)
            if count_col is not None: obs_agg.loc[i, count_col] = len(idxs)

        # clean dtypes
        obs_agg = obs_agg.infer_objects()

        return obs_agg, X_agg

    # --- operations ---
    def combine_obj(self, to_append: "PhotometryData" | List["PhotometryData"], inplace: bool = False) -> None | PhotometryData:
        """
        Concatenate the rows of object with one or more PhotometryData objects.
        Args:
            to_append (PhotometryData | list[PhotometryData]): Object(s) to append.
            inplace (bool) : Whether to modify the original object or return new merged one.
        Returns:
            Combined PhotometryData or None depending on ``inplace``.
        """        
        if not isinstance(to_append, list):
            to_append = [to_append]
        adatas = [getattr(self, 'adata')] + [getattr(obj, 'adata') for obj in to_append]
        merged_adata = ad.concat(
            adatas=adatas,
            axis='obs',
            join='inner',
            merge='same',
            uns_merge='same',
            index_unique=''
        )
        merged_adata.obs.reset_index(drop=True, inplace=True)
        if inplace:
            self.adata = merged_adata
            return
        else:
            return PhotometryData(merged_adata)

    def collapse(
            self,
            group_on: List[str],
            method: function = np.nanmean,
            metrics: Dict[str : function] = {},
            data_cols: list[str] = [],
            count_col: str | None = 'n'
            ) -> "PhotometryData":
        """
        Collapse trials by grouping obs and aggregating X and selected obs columns.
        Args:
            group_on (list[str]): Columns in obs used to define groups.
            method (callable): Aggregation function for the main X matrix.
            metrics (dict[str, callable]): Additional named aggregation functions stored as layers.
            data_cols (list[str]): Observation columns to aggregate with each method.
            count_col (str | None): Optional column name to store group counts.
        Returns:
            PhotometryData.
        """        
        obs_agg, X_agg = self._agg(method=method, group_on=group_on, data_cols=data_cols, count_col=count_col)

        layers = {}
        for key, func in metrics.items():
            obs_lay, X_lay = self._agg(method=func, group_on=group_on, data_cols=data_cols, count_col=count_col)
            layers[key] = X_lay
            obs_agg = obs_agg.join(obs_lay[data_cols], rsuffix='_' + str(key))
        
        new_obj = PhotometryData.from_arrays(
            obs=obs_agg,
            data=X_agg,
            time_points=self.adata.var['t'],
            layers=layers,
            metadata=self.adata.uns
        )
        return new_obj
    
    # --- plotting ---
    def plot_line(
            self, i: int, 
            ax: Axes | None = None, 
            label_with: List[str] | None = None, 
            err_layer: str | None = None,
            **kwargs
            ) -> None:
        """
        Plot a single trial time-series with optional error band.
        Args:
            i (int): Trial index to plot.
            ax (matplotlib.axes.Axes | None): Axis to plot on. Creates a new one if None.
            label_with (list[str] | None): Obs columns used to build the legend label.
            err_layer (str | None): Optional layer name providing per-timepoint error.
            **kwargs: Additional keyword arguments passed to plt.plot.
        Returns:
            None
        """        
        if ax is None: fig, ax = plt.subplots()
        
        x = self.adata.var['t'].to_numpy()
        y = self.adata.X[i, :]
        row = self.adata.obs.iloc[i]

        if label_with is not None:
            vals = row[label_with].astype(str).to_list()
            name_val_pairs = [k + ': ' + v for k, v in zip(label_with, vals)]
            label = ', '.join(name_val_pairs)
        else:
            label = None

        ax.plot(x, y, label=label)
        if err_layer is not None:
            yerr = self.adata.layers[err_layer][i]
            ax.fill_between(x, y + yerr, y - yerr, alpha=0.3)

    def plot_set(
            self, 
            subset: List[bool | int], 
            ax: Axes = None,
            title: str = None, 
            label_with: List[str] = None, 
            err_layer: str = None, 
            **kwargs
            ) -> None:
        """
        Plot a set of trials given a boolean or index subset.
        Args:
            subset (list[bool | int]): Boolean mask or indices selecting trials.
            ax (matplotlib.axes.Axes | None): Axis to plot on. Creates a new one if None.
            title (str | None): Optional title for the axis.
            label_with (list[str] | None): Obs columns used to build legend labels.
            err_layer (str | None): Optional layer name providing per-timepoint error.
            **kwargs: Additional keyword arguments passed to plot_line.
        Returns:
            None
        """        
        idxs = np.arange(self.n_trials)[subset]
        if ax is None: fig, ax = plt.subplots()
        for i in idxs:
            self.plot_line(i, ax=ax, label_with=label_with, err_layer=err_layer)
        if label_with is not None: plt.legend()
        if title is not None: ax.set_title(title)

    def plot_groups(
            self, 
            group_on: List[str], 
            label_with: List[str] | None = None, 
            err_layer: str | None = None, 
            save_dir: str | None = None,
            save_ext: str = '.png',
            save_dpi: int = 140, 
            **kwargs):
        """
        Plot trials grouped by observation columns.
        Args:
            group_on (list[str]): Obs columns used to define groups.
            label_with (list[str] | None): Obs columns used to build legend labels.
            err_layer (str | None): Optional layer name providing per-timepoint error.
            save_dir (str | None): Optional output directory to save figures to.
            save_ext (str): What format to save images in.
            save_dpi (int): What dpi to save images at.
            **kwargs: Additional keyword arguments passed to plot_set.
        Returns:
            None
        """        
        groups = self.adata.obs.groupby(group_on).indices
        for gkey, idxs in groups.items():
            if not isinstance(gkey, tuple): gkey=[gkey]
            title = ', '.join([f'{name}: {val}' for name, val in zip(group_on, gkey)])
            self.plot_set(subset=idxs, label_with=label_with, title=title, err_layer=err_layer, **kwargs)
            if save_dir is not None:
                file_name = '_'.join([f'{name}-{val}' for name, val in zip(group_on, gkey)]) + save_ext
                plt.savefig(os.path.join(save_dir, file_name), dpi=save_dpi)
            plt.show()

    # --- convienience ---
    def filter_rows(self, mask: np.ndarray[bool], inplace: bool = False) -> None | "PhotometryData":
        """
        Filter rows (trials) using a boolean mask.
        Args:
            mask (np.ndarray[bool]): Boolean array of length n_trials.
            inplace (bool): If True, modify in place. If False, return a new object.
        Returns:
            None | PhotometryData: New filtered object if inplace is False, else None.
        """        
        if inplace:
            self.adata = self.adata[mask, :].copy()
        else:
            return PhotometryData(self.adata[mask, :].copy())

    def add_obs_columns(self, add_from: Dict[str, Any], keys: List[str] | None = None) -> None:
        """
        Add columns to obs from a dictionary.
        Args:
            add_from (dict[str, Any]): Mapping from column names to values.
            keys (list[str] | None): Keys from add_from to add. Defaults to all keys.
        Returns:
            None
        """        
        keys = list(add_from.keys()) if keys is None else keys
        for k in keys:
            self.adata.obs[k] = add_from[k]

    def add_metadata(self, add_from: Dict[str, Any], keys: List[str] | None = None) -> None:
        """
        Add entries to the .uns metadata dictionary.
        Args:
            add_from (dict[str, Any]): Mapping from keys to metadata values.
            keys (list[str] | None): Keys from add_from to add. Defaults to all keys.
        Returns:
            None
        """        
        keys = list(add_from.keys()) if keys is None else keys
        for k in keys:
            self.adata.uns[k] = add_from[k]

    def drop_obs_columns(self, to_drop: List[str]) -> None:
        """
        Drop observation columns from obs.
        Args:
            to_drop (list[str]): Column names to drop.
        Returns:
            None
        """        
        self.adata.obs = self.adata.obs.drop(to_drop, errors='ignore')

    def get_text_value_counts(self, col: str) -> str:
        """
        Get a string summary of value counts for a column in obs.
        Args:
            col (str): Column name in obs.
        Returns:
            str: Comma-separated summary of value counts.
        """
        vc = self.adata.obs[col].value_counts(dropna=False)
        return ", ".join(f"{k}: {v}" for k, v in vc.items())

class PhotometryExperiment:
    """
    Handle extraction and preprocessing of raw photometry data from TDT folders.
    """
    def __init__(
        self,
        data_folder: str,
        box: str,
        event_labels: list[str],
        signal_label: str,
        isosbestic_label: str,
        notes_filename: str,
    ):
        """
        Initialize a PhotometryExperiment with TDT configuration.
        Args:
            data_folder (str): Path to the TDT block folder.
            box (str): TDT box identifier used in stream and epoc labels.
            event_labels (list[str]): Event labels to extract from epocs.
            signal_label (str): Base label for the signal channel.
            isosbestic_label (str): Base label for the isosbestic channel.
            notes_filename (str): Name of the notes file associated with this session.
        Returns:
            None
        """        
        self.data_folder = data_folder
        self.notes_filename = notes_filename
        self.box = box
        self.signal_label = signal_label
        self.isosbestic_label = isosbestic_label
        self.event_labels = list(event_labels)

        self.metadata = {}
        self.events = {}
        self.raw_signal = None
        self.raw_isosbestic = None
        self.frequency = None
        self.time = None
        self.trial_data = None
                
    # --- pipeline API ---
    def run_pipeline(self) -> None:
        """
        Run the full processing pipeline for 1 experiment (to be implemented in child classes).
        Args:
            None
        Returns:
            None
        """
        return

    # --- data extraction ---
    def extract_data(self, downsample: int = 10) -> None:
        """
        Read a TDT block, extract streams and events, and downsample signals.
        Args:
            downsample (int): Integer downsampling factor for the raw streams.
        Returns:
            None
        """
        tdt_obj = tdt.read_block(self.data_folder)

        # rip data out of TDT object
        sig = tdt_obj.streams[self.signal_label + self.box].data
        iso = tdt_obj.streams[self.isosbestic_label + self.box].data
        fs = tdt_obj.streams[self.signal_label + self.box].fs

        self.raw_signal = downsample_1d(np.asarray(sig, dtype=np.float32), factor=downsample)
        self.raw_isosbestic = downsample_1d(np.asarray(iso, dtype=np.float32), factor=downsample)
        self.frequency = float(fs) / downsample

        n = self.raw_signal.size
        self.time = np.arange(n, dtype=float) / self.frequency

        # extract event timestamps for requested labels
        self.events = {}
        self.metadata['missing_events'] = []
        for label in self.event_labels:
            # some sessions may lack a label entirely if no events are recorded
            if hasattr(tdt_obj.epocs, self.box + label):
                ep = tdt_obj.epocs[self.box + label]
                self.events[label] = np.asarray(ep.onset)
            else:
                self.events[label] = np.array([], dtype=float)
                self.metadata['missing_events'].append(label)
        del tdt_obj

    # --- signal processing ---
    def preprocess_signal(self,
                          cutoff_frequency: float = 30.0, 
                          order: int = 4,
                          method: str = 'dF/F'
                          ) -> None:
        """
        Low-pass filter and preprocess the signal using isosbestic fitting.
        Args:
            cutoff_frequency (float): Low-pass cutoff frequency in Hz.
            order (int): Butterworth filter order.
            method (str): Preprocessing method, either 'dF/F' or 'dF'.
        Returns:
            None
        """
        filt_sig = self.low_frequency_pass_butter(self.raw_signal, self.frequency,
                                                    cutoff_frequency=cutoff_frequency,
                                                    order=order)
        filt_iso = self.low_frequency_pass_butter(self.raw_isosbestic, self.frequency,
                                                    cutoff_frequency=cutoff_frequency,
                                                    order=order)
        
        min_len = min(filt_sig.size, filt_iso.size)
        filt_sig = filt_sig[:min_len]
        filt_iso = filt_iso[:min_len]

        fitted_iso, r2_val, coeff = self.fit_isosbestic_to_signal_IRLS(filt_sig, filt_iso)
        self.metadata['signal_processing_method'] = method 
        self.metadata['isosbestic_fit'] = {'r2_val' : r2_val,
                                           'coeffs' : coeff}
        self.fitted_isosbestic = fitted_iso

        avaliable_methods = ['dF/F', 'dF']
        match method:
            case 'dF':
              self.signal = (filt_sig - fitted_iso)
            case 'dF/F':
                self.signal = (filt_sig - fitted_iso) / np.maximum(fitted_iso, np.finfo(np.float32).eps)
            case _:
                raise ValueError(f'Preprocessing method {method} not avaliable\n'
                                 f'Avalible methods are {avaliable_methods}')

        self.signal = self.signal.astype(np.float32, copy=False)
        return
    
    def low_frequency_pass_butter(
            self,
            signal: np.ndarray, 
            sample_frequency: float, 
            cutoff_frequency: float = 30.0, 
            order: int = 4
        ) -> np.ndarray:
        """
        Apply a low-pass Butterworth filter to a 1D signal. Usually already done by photometry machine at 20 or 30 Hz.
        Args:
            signal (np.ndarray): Input signal array.
            sample_frequency (float): Sampling frequency in Hz.
            cutoff_frequency (float): Low-pass cutoff frequency in Hz.
            order (int): Butterworth filter order.
        Returns:
            np.ndarray: Filtered signal.
        """        
        normalized_frequency = cutoff_frequency / (sample_frequency / 2)
        sos = butter(order, normalized_frequency, btype='low', output='sos') 
        return sosfiltfilt(sos, signal, axis=0, padtype='odd', padlen=None)
    
    def fit_isosbestic_to_signal_IRLS(self, signal: np.ndarray, isosbestic: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Fit the isosbestic channel to the signal using IRLS.
        Args:
            signal (np.ndarray): Filtered signal trace.
            isosbestic (np.ndarray): Filtered isosbestic trace.
        Returns:
            tuple[np.ndarray, float, np.ndarry]: Fitted isosbestic, R² value, and fit coefficients.
        """
        # Ensure 1D float64 for statsmodels
        y = np.asarray(signal, dtype=np.float64, copy=False).ravel()
        x = np.asarray(isosbestic, dtype=np.float64, copy=False).ravel()
        # add intercept
        X = sm.add_constant(x)  

        model = sm.RLM(endog=y, exog=X, M=sm.robust.norms.TukeyBiweight())
        res = model.fit()
        fitted_isosbestic = res.fittedvalues.astype(np.float32, copy=False)
        r2_val = r2_score(signal, fitted_isosbestic)
        return fitted_isosbestic, r2_val, res.params
    
    def fit_photobleaching_curve(
            self, 
            downsample_factor: int = 100, 
            skip: int = 200, 
            n_boost: int = 500000, 
            boost_factor: int = 2, 
            return_curve: bool = False
            ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Fit a negative bi-exponential to model photobleaching in the raw signal.
        Args:
            skip (int): Number of initial samples to skip before fitting.
            n_boost (int): Number of early samples to down-weight.
            boost_factor (int): Factor by which early samples are down-weighted.
            return_curve (bool): If True, also return the fitted curve.
        Returns:
            tuple[pd.DataFrame, np.ndarray]: DataFrame of parameters and metrics, and fitted curve or None.
        """
        x = downsample_1d(self.time[skip:], factor=downsample_factor)
        y = downsample_1d(self.raw_signal[skip:], factor=downsample_factor)

        # Initial guess and bounds for the parameters
        initial_guess = [y.max()-y[-1], 0.001, y.max()-y[-1], 0.001, y[-1]]
        bounds = ([0, 0, 0, 0, -np.inf], 
                  [np.inf, np.inf, np.inf, np.inf, np.inf])
        
        # Reweighted data points to emphasize early times
        weights = np.ones_like(y)
        weights[:n_boost] /= boost_factor

        try:
            params, _ = curve_fit(neg_bi_exponential_5, x, y, p0=initial_guess, bounds=bounds, maxfev=20000, sigma=weights)
            fitted_curve = neg_bi_exponential_5(x, *params)
            r2_val = r2_score(y, fitted_curve)
            mse_val = mean_squared_error(y, fitted_curve)
        except RuntimeError:
            # If the fit fails, return NaNs
            fitted_curve = np.full_like(x, np.nan)
            r2_val = np.nan
            mse_val = np.nan
            params = np.full(len(initial_guess), np.nan)

        row = {'id': self.id, 
               'a1': params[0], 'b1': params[1], 'a2': params[2], 'b2': params[3], 'c': params[4],
               'r2_val': r2_val, 'mse_val': mse_val}
        
        self.fitted_params = params
        fitted_curve = fitted_curve.astype(np.float32, copy=False) if return_curve else None

        return pd.DataFrame([row]), fitted_curve
    
    # --- extraction of trial-wise data ---
    def extract_trial_data(
        self,
        align_to: str,
        center_on: list[str],
        trial_bounds: list[float, float],
        baseline_bounds: list[float, float],
        event_tolerences: Dict[str, Tuple[float, float]],
        normalization: Literal['zscore', 'zero', 'none'] = 'zscore',
        check_overlap: bool = True,
        time_error_threshold: float = 0.01,
    ) -> None:
        """
        Build trial-wise windows, normalize, and store trial data.
        Args:
            align_to (str): Event label used to align and identify trial, should be one per trial.
            center_on (list[str]): Event labels to center trial windows on.
            trial_bounds (list[float, float]): Trial window bounds relative to ``center_on`` events.
            baseline_bounds (list[float, float]): Baseline window bounds relative to ``align_to`` event.
            event_tolerences (dict[str, tuple[float, float]]): Time tolerances for event annotation, relative to ``align_to``.
            normalization (Literal['zscore', 'zero']): Normalization method for trial signals based on baselines.
            check_overlap (bool): Whether to throw an error multiple ``center_on`` events are found in the same trial.
            time_error_threshold (float): Maximum allowed mean timing error.
        Returns:
            None
        """        
        # build trials around align_to event
        align_events = self.events[align_to].copy()

        # validate inputs
        if align_to not in self.events: raise KeyError(f"align_to '{align_to}' not found in events: {list(self.events)}")
        if align_events.size == 0: raise ValueError(f"No '{align_to}' events found.")
        missing = [lab for lab in center_on if lab not in self.events]
        if missing: raise KeyError(f"center_on labels not found in events: {missing}")
        
        # annotate events based on tolerences
        events_selected = self.annotate_intervals(
            align_to=align_to,
            series=self.time, 
            centers=align_events,
            events=self.events, 
            tolorences=event_tolerences
            )
        
        # find window centers using the selected events
        trial_window_centers = self.find_window_centers(
            center_on=center_on, 
            align_on=align_to, 
            events=events_selected,
            check_overlap=check_overlap,
            )
        baseline_window_centers = align_events
        
        # construct trial and baseline windows
        raw_trial_signals, trial_times, trial_events = self.create_windows(
            signal=self.signal,
            time=self.time,
            events=events_selected,
            centers=trial_window_centers,
            bounds=trial_bounds
        )
        baseline_signals, baseline_times, baseline_events = self.create_windows(
            signal=self.signal,
            time=self.time,
            events=events_selected,
            centers=baseline_window_centers,
            bounds=baseline_bounds
        )

        # apply trial-wise normalization method
        match normalization:
            case 'zscore':
                trial_signals = zscore_signal(raw_trial_signals, baseline_signals)
            case 'zero':
                trial_signals = center_signal(raw_trial_signals, baseline_signals)
            case 'none':
                trial_signals = raw_trial_signals
            case _:
                raise ValueError(f'Invalid normalization method ({normalization})')
        
        # reconstruct times for consistency
        trial_time_points = reconstruct_time_points(trial_bounds, self.frequency)
        baseline_time_points = reconstruct_time_points(baseline_bounds, self.frequency)

        # save trial data
        self.trial_signals = trial_signals
        self.trial_time_points = trial_time_points
        self.trial_events = trial_events
        self.raw_trial_signal = raw_trial_signals

        self.baseline_signals = baseline_signals
        self.baseline_time_points = baseline_time_points
        self.baseline_events = baseline_events

        # check error in times
        time_err = trial_times.std(axis=0).mean()
        if time_err > time_error_threshold:
            raise ValueError(f'Error in rounded times is too high: {time_err} > {time_error_threshold}')
        
    def find_interval_bounds(self, series: np.ndarray, centers: np.ndarray, bounds: Tuple[int, int]) -> np.ndarray:
        """
        Compute index bounds for windows around specified centers.
        Args:
            series (np.ndarray): Monotonic time-like series.
            centers (np.ndarray): Center times for each interval.
            bounds (tuple[int, int]): Relative lower and upper bounds in the same units as series.
        Returns:
            np.ndarray: Array of shape (n_intervals, 2) with left and right indices.
        """        
        low, high = bounds
        left_idxs = np.searchsorted(series, centers + low, side='left')
        right_idxs = np.searchsorted(series, centers + high, side='right')
        return np.c_[left_idxs, right_idxs]

    def first_timestamp_in_intervals(self, timestamps: np.ndarray, time_intervals: np.ndarray) -> np.ndarray:
        """
        Find the first timestamp that falls within each time interval.
        Args:
            timestamps (np.ndarray): Sorted 1D array of event timestamps.
            time_intervals (np.ndarray): Array of shape (n, 2) with [start, end] bounds.
        Returns:
            np.ndarray: Array of first timestamps per interval or NaN if none exist.
        """        
        # tests which events are in interval
        timestamps = np.sort(timestamps)
        lo_idx = np.searchsorted(timestamps, time_intervals[:, 0], side="left")
        hi_idx = np.searchsorted(timestamps, time_intervals[:, 1], side="right")
        in_interval = lo_idx < hi_idx
        # returns lowest event if within interval
        out = np.full(len(time_intervals), np.nan, float)
        out[in_interval] = timestamps[lo_idx[in_interval]]
        return out
    
    def annotate_intervals(self, align_to: str, series: np.ndarray, centers: np.ndarray, events: Dict[str, np.ndarray], tolorences: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Annotate intervals around centers with event timestamps within tolerances.
        Args:
            align_to (str): Label of the primary alignment event.
            series (np.ndarray): Monotonic time-like series.
            centers (np.ndarray): Center times for each trial.
            events (dict[str, np.ndarray]): Mapping of event labels to timestamps.
            tolorences (dict[str, np.ndarray]): Mapping of labels to time tolerances.
        Returns:
            dict[str, np.ndarray]: Mapping from labels to aligned event times per trial.
        """        
        out = {}
        out[align_to] = centers
        for label, bounds in tolorences.items():
            timestamps = events.get(label)
            interval_bounds = self.find_interval_bounds(series=series, centers=centers, bounds=bounds)
            time_intervals = series[interval_bounds]
            out[label] = self.first_timestamp_in_intervals(timestamps=timestamps, time_intervals=time_intervals)
        return out

    def create_windows(self, signal: np.ndarray, time: np.ndarray, events: Dict[str, np.ndarray], centers: np.ndarray, bounds: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Slice signal and time into fixed-length windows around centers.
        Args:
            signal (np.ndarray): Full preprocessed signal trace.
            time (np.ndarray): Associated time vector.
            events (dict[str, np.ndarray]): Per-label event times for each trial.
            centers (np.ndarray): Window center times for each trial.
            bounds (tuple[int, int]): Relative bounds [low, high] around centers.
        Returns:
            tuple[np.ndarray, np.ndarray, dict]: Signal windows, centered time windows, and centered events.
        """        
        # find target window bounds
        intervals = self.find_interval_bounds(series=time, centers=centers, bounds=bounds)
        window_idxs = [np.arange(left, right+1) for left, right in intervals]

        # calculate minimum window size to ensure stable sizes across experiments with the same freq and bounds
        target_len = np.floor((bounds[1] - bounds[0]) * self.frequency).astype(int)
        minimum_len = min(map(len, window_idxs))
        assert minimum_len >= target_len
        window_idxs = [arr[:target_len] for arr in window_idxs]

        # slice and stack + center time and events
        signal_windows = signal[window_idxs]
        time_windows = time[window_idxs] - centers[:, np.newaxis]
        events_centered = {k : v - centers for k, v in events.items()}
        return signal_windows, time_windows, events_centered

    def find_window_centers(self, center_on: str | List[str], align_on: str, events: Dict[str, np.ndarray], check_overlap: bool = True) -> np.ndarray:
        """
        Determine window centers based on center_on events or fallback to align_on.
        Args:
            center_on (str | list[str]): Event labels used as preferred centers.
            align_on (str): Fallback event label used when center_on is missing.
            events (dict[str, np.ndarray]): Mapping from labels to event times per trial.
            check_overlap (bool): Whether to throw an error multiple ``center_on`` events are found in the same trial.
        Returns:
            np.ndarray: Center times per trial.
        """        
        # center_on events should be non-overlaping
        # if no center_on events present, center on align_on
        if isinstance(center_on, str):
            center_on = [center_on]

        centers = events[align_on].copy()
        overlap = np.full_like(centers, True, dtype=bool)
        for label in center_on:
            arr = events[label]
            event_not_nan = ~np.isnan(arr)
            centers[event_not_nan] = arr[event_not_nan]
            overlap &= event_not_nan
        if check_overlap and overlap.any():
            at_idxs = np.where(overlap)
            culprits = {k : events[k][at_idxs] for k in center_on}
            culprits_in_og_events = {k : self.events[k][np.searchsorted(self.events[k], culprits[k])] for k in center_on}
            raise ValueError(f'Center_on events over lap in trials {np.where(overlap)}, with culprits: {culprits_in_og_events}')
        return centers
    
    # --- poor signal checks ---
    def median_centered_abs_max_check(self, trial_signal: np.ndarray, threshold: float = 0.075):
        median_centered = trial_signal - np.median(trial_signal, axis=1, keepdims=True)
        abs_max = np.abs(median_centered).max(axis=1)
        is_poor_signal = np.mean(abs_max) < threshold
        return is_poor_signal

    # --- graphing ---
    def dashboard(self, save: str | None = None) -> None:
        """
        Quickly plot the raw, fitted, and processed signal, isosbestic, and fitted photobleaching curve (if avaliable).
        Args:
            save (str, None): Path to save figure, if None figure does not save.
        Returns:
            None.
        """
        fig, (ax1, ax2) = plt.subplots(
            ncols=1, nrows=2, 
            sharex=True, figsize=(6, 6), dpi=140,
            gridspec_kw={'height_ratios': [3, 1]})
        fig.tight_layout()

        downsample_factor = 20
        x = downsample_1d(self.time, downsample_factor)
        raw_sig = downsample_1d(self.raw_signal, downsample_factor)
        raw_iso = downsample_1d(self.raw_isosbestic, downsample_factor)
        fit_iso = downsample_1d(self.fitted_isosbestic, downsample_factor)
        fit_params = getattr(self, 'fitted_params', None)
        final_sig = downsample_1d(self.signal, downsample_factor)
        
        # raw and fitted signals
        ax1: plt.Axes
        ax1.plot(x, raw_sig, label='Raw Signal', c='#1f77b4')
        ax1.plot(x, raw_iso, label='Raw Iso.', c="#4B4B4B", alpha=0.9)
        ax1.plot(x, fit_iso, label='Fitted Iso.', c='#ff7f0e', alpha=0.9)
        if fit_params is not None:
            ax1.plot(x, neg_bi_exponential_5(x, *fit_params), label='Fitted Curve', c="#920000")
        ax1.legend()

        # processed signal
        ax2: plt.Axes
        y_pad_factor = 2.5
        middle_third = np.array_split(final_sig, 3)[1]
        y_high = np.max(middle_third)
        y_low = np.min(middle_third)
        print(y_high)
        ax2.plot(x, final_sig, label='Processed Signal', c='#1f77b4')
        ax2.set_ylim(bottom=y_low*y_pad_factor, top=y_high*y_pad_factor)
        ax2.legend()

        # annotate
        ax1.set_title(
        f"Dashboard for {getattr(self, 'id', 'Unnamed')}\n"
        f"Origin: {self.data_folder.split('/')[-1]}"
        )
        ax1.set_ylabel('Signal amplitude (a.u.)')
        ax2.set_ylabel(f"{self.metadata.get('signal_processing_method', 'NOT FOUND')}")
        ax2.set_xlabel('Time (s)')

        if save is not None:
            plt.savefig(save, bbox_inches='tight')