from __future__ import annotations
from typing import Dict, Sequence, Tuple, Union, Literal, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import anndata as ad
import re
import os
import logging

from ..core import PhotometryData, PhotometryExperiment
from ..utils import *
from ..io import *

ArrayLike = Union[np.ndarray, Sequence[float]]

class RDT_PhotometryData(PhotometryData):
    adata: ad.AnnData
    """
    RDT-specific implementation of PhotometryData with block/trial labels and QC.
    """
    def __init__(self, adata):  
        """
        Initialize an RDT_PhotometryData wrapper from an AnnData object.
        Args:
            adata (anndata.AnnData): AnnData object containing trial-wise data.
        Returns:
            RDT_PhotometryData
        """        
        super().__init__(adata=adata)

    # --- labeling ---
    def label_blocks(
        self,
        n_blocks: int = 3,
        free_trial_slices: Sequence[Sequence[int]] = ((8, 28), (36, 56), (64, 84)),
        block_labels: Sequence[str] = ("0%", "25%", "75%"),
    ) -> None:
        """
        Label trials into blocks and forced/free for the RDT task.
        Args:
            n_blocks (int): Number of blocks.
            free_trial_slices (Sequence[Sequence[int]]): (start, end) indices for free trials per block.
            block_labels (Sequence[str]): Block labels, e.g. reward probabilities.
        Returns:
            None
        """
        if len(free_trial_slices) != n_blocks or len(block_labels) != n_blocks:
            raise ValueError("length of free_trial_slices and block_labels must match n_blocks")

        # implicit contiguous forced blocks that precede each free slice
        block_ends = [0] + [hi for (_, hi) in free_trial_slices]
        block_slices = [(block_ends[i], block_ends[i + 1]) for i in range(len(block_ends) - 1)]
        
        obs = self.adata.obs
        obs["block"] = pd.Series(index=obs.index, dtype="object")
        obs["block_num"] = pd.Series(index=obs.index, dtype="int64")
        obs["forced"] = True

        for (b_lo, b_hi), b_num, b_lab, (f_lo, f_hi) in zip(
            block_slices, range(1, n_blocks + 1), block_labels, free_trial_slices
        ):
            obs.loc[obs.index[b_lo:b_hi], 'block_num'] = int(b_num)
            obs.loc[obs.index[b_lo:b_hi], 'block'] = b_lab
            obs.loc[obs.index[f_lo:f_hi], 'forced'] = False

        self.adata.obs = obs

    def label_trials(self) -> None:
        """
        Label trials as Sml, Pun, or UnPun based on event timestamps. Conflicting or missing timestamps result in NaN.
        Args:
            None
        Returns:
            None
        """        
        obs = self.adata.obs
        obs["trial_label"] = pd.Series(index=obs.index, dtype="object")

        is_sml = obs.get("Sml", pd.Series(index=obs.index)).notna()
        is_lrg = obs.get("Lrg", pd.Series(index=obs.index)).notna()
        is_zap = obs.get("Zap", pd.Series(index=obs.index)).notna()

        obs.loc[is_sml, "trial_label"] = "Sml"
        obs.loc[is_lrg & is_zap, "trial_label"] = "Pun"
        obs.loc[is_lrg & ~is_zap, "trial_label"] = "UnPun"
        
        # ensure all Sml and Lrg do not overlap and Sml trials have no Zap timestamp
        obs.loc[is_lrg & is_sml, "trial_label"] = pd.NA
        obs.loc[is_sml & is_zap, "Zap"] = pd.NA

    # --- quality control ---
    def quality_control(self, drop: bool = False) -> None:
        """
        Run simple QC checks on trials and optionally drop bad trials.
        Args:
            drop (bool): If True, drop trials that fail QC.
        Returns:
            None
        """        
        self.qc = {}

        self.qc['before'] = self.info()

        good_trials = np.full(self.n_trials, True, dtype=bool)
        if 'Hsl' in self.obs:
            good_trials &= ~self.obs['Hsl'].isna()
        
        if ('Lrg' in self.obs) and ('Sml' in self.obs):
            xor_lever = np.logical_xor(self.obs['Lrg'].isna().to_numpy(), self.obs['Sml'].isna().to_numpy())
            good_trials &= xor_lever
        
        if 'forced' in self.obs:
            good_trials &= ~self.obs['forced']

        if 'poorSignalFlag' in self.obs:
            good_trials &= ~(self.obs['poorSignalFlag'])

        self.filter_rows(good_trials, inplace=drop)
        self.qc['after'] = self.info()

    # --- convienice ---
    def trial_type_summary(self) -> Dict[str, Any]:
        """
        Summarize counts of trial types and blocks.
        Args:
            None
        Returns:
            dict[str, Any]: Summary of trial counts and labels.
        """        
        summary = {
            'n_trials' : self.n_trials,
            'forced_trials' : self.get_text_value_counts('forced') if 'forced' in self.adata.obs else 'NA',
            'blocks' : self.get_text_value_counts('block') if 'block' in self.adata.obs else 'NA',
            'trial_label' : self.get_text_value_counts('trial_label') if 'trial_label' in self.adata.obs else 'NA'
        }
        return summary

    def info(self, info_on: list[str] = ['rat', 'trial_label', 'block', 'current', 'forced', 'poorSignalFlag']) -> None:
        """
        Build a simple string summary of key obs columns.
        Args:
            info_on (list[str]): Observation columns to summarize.
        Returns:
            str: Multi-line string with counts and unique values per column.
        """        
        info_str = "\n"
        for col in info_on:
            if col in self.obs:
                info_str += f'\t{col} - {self.get_text_value_counts(col)}; {self.obs[col].unique().size} unique\n'
        return info_str

class RDT_PhotometryExperiment(PhotometryExperiment):
    """
    RDT-specific implementation of PhotometryExperiment for TDT-based RDT sessions.
    """
    def __init__(
        self,
        data_folder: str,
        box: str = "A",
        event_labels: list[str] = ("Lrg", "Sml", "Hsl", "Zap"),
        signal_label: str = "_465",
        isosbestic_label: str = "_405",
        notes_filename: str = "Notes.txt",
    ) -> "RDT_PhotometryExperiment":
        """
        Initialize an RDT_PhotometryExperiment with TDT and RDT settings.
        Args:
            data_folder (str): Path to the TDT block folder.
            box (str): TDT box identifier.
            event_labels (list[str]): Event labels to extract for RDT.
            signal_label (str): Base label for the signal stream.
            isosbestic_label (str): Base label for the isosbestic stream.
            notes_filename (str): File name of the notes file in the data folder.
        Returns:
            RDT_PhotometryExperiment
        """
        super().__init__(data_folder, box, event_labels, signal_label, isosbestic_label, notes_filename)

        self.parse_notes_file(self.notes_filename)
        self.metadata["Data Folder"] = self.data_folder
        self.id = (
            f"{self.metadata.get('rat', 'UnknownRat')}_"
            f"{self.metadata.get('current', 'UnknownCurrent')}uA_"
            f"Box{self.box}_"
            f"{self.metadata.get('stripped_date', 'UnknownDate')}"
        )
        f"{self.metadata.get('rat', 'UnknownRat')}_{self.metadata.get('current', 'UnknownCurrent')}uA_{self.metadata.get('stripped_date', 'UnknownDate')}"
                
    # --- pipeline API ---
    def run_pipeline(
        self,
        logger: logging.Logger | None = None,
        downsample: int = 10,
        cutoff_frequency: float = 30.0,
        order: int = 4,
        preprocess_method: str = 'dF/F',
        
        align_to: str = 'Hsl',
        center_on: list[str] = ['Lrg', 'Sml'],
        trial_bounds: Tuple[float, float] = (-10.0, 5.0),
        baseline_bounds: Tuple[float, float] = (-5, -1),
        event_tolerences: Dict[str, Tuple[float, float]] = {'Lrg' : (5, 18), 'Sml' : (5, 18), 'Zap': (4.5, 18.5)},
        normalization: Literal['zscore', 'zero'] = 'zscore',
        check_overlap: bool = False,
        poor_signal_threshold: float = 0.075,
        time_error_threshold: float = 0.01,

        n_blocks: int = 3,
        free_trial_slices: list[list[int]] = ((8, 28), (36, 56), (64, 84)),
        block_labels: list[str] = ("0%", "25%", "75%"),
        qc_drop: bool = False,
        to_trim: list[str] = ['Lrg', 'Sml'],
    ) -> None:
        """
        Run full RDT pipeline: extract, preprocess, window trials, label, QC, and optionally save.
        Args:
            logger (logging.Logger | None): Logger for status messages.
            downsample (int): Downsampling factor for raw TDT streams.
            cutoff_frequency (float): Low-pass filter cutoff in Hz.
            order (int): Butterworth filter order.
            preprocess_method (str): Preprocessing method, e.g. 'dF/F' or 'dF'.
            align_to (str): Primary event label used to align trials.
            center_on (list[str]): Events used to refine trial centers.
            trial_bounds (tuple[float, float]): Trial window bounds relative to center.
            baseline_bounds (tuple[float, float]): Baseline window bounds relative to center.
            event_tolerences (dict[str, tuple[float, float]]): Time tolerances for event annotation.
            normalization (Literal['zscore', 'zero']): Trial normalization method.
            check_overlap (bool): Whether to throw an error multiple ``center_on`` events are found in the same trial.
            poor_signal_threshold (float): Threshold for poor signal checking.
            time_error_threshold (float): Threshold on timing error for sanity check.
            n_blocks (int): Number of RDT blocks.
            free_trial_slices (list[list[int]]): Free trial index ranges per block.
            block_labels (list[str]): Labels for each block.
            qc_drop (bool): If True, drop QC-failed trials.
            to_trim (list[str]): Obs columns to drop from final dataset.
        Returns:
            None
        """
        # step 0: logging
        log = logger or logging.getLogger(__name__)
        log.info(f"Starting Pipeline {self.id}...")
        
        # step 1: extract raw data from TDT
        log.info(f"Extracting raw data from TDT block, downsampling x{downsample}...")
        self.extract_data(downsample=downsample)
        if len(self.metadata['missing_events']) != 0:
            log.warning(f"Requested events {self.metadata['missing_events']} are missing")
        if align_to in self.metadata['missing_events']:
            raise ValueError(f'There are no {align_to} events present in data!')

        # step 2: preprocess signal with lowpass filter and dF/F strategy
        log.info(f"Preprocessing signal...")
        self.preprocess_signal(cutoff_frequency=cutoff_frequency, order=order, method=preprocess_method)
        log.info(f"Done. Fitted isosbestic R2 = {self.metadata['isosbestic_fit']['r2_val']:.4f}")

        # step 3: extract per-trial data
        log.info(f"Extracting per-trial data...")
        self.trials, self.baselines = self.extract_trial_data(
            align_to=align_to,
            center_on=center_on,
            trial_bounds=trial_bounds,
            baseline_bounds=baseline_bounds,
            event_tolerences=event_tolerences,
            normalization=normalization,
            time_error_threshold=time_error_threshold,
            poor_signal_threshold=poor_signal_threshold,
            check_overlap=check_overlap,
        )
        log.info(f"Done. Extracted {self.trials.n_trials} trials of {self.trials.n_times} size each.")

        # step 4: annotate and clean per-trial data
        log.info(f"Annotating and cleaning trial data...")
        self.trials.label_trials()
        self.trials.label_blocks(
            n_blocks=n_blocks, free_trial_slices=free_trial_slices, block_labels=block_labels
        )
        self.trials.quality_control(drop=qc_drop)

        # step 5: pass down important metadata
        self.trials.add_obs_columns(self.metadata, keys=['rat', 'box', 'current', 'date'])
        self.trials.drop_obs_columns(to_drop=to_trim)
        log.info(f"Done. {self.trials.n_trials} trials remain.")

        if self.poorSignalFlag:
            log.warning(f"Warning: Poor signal quality detected in {self.id}.")

        log.info(f"Pipeline complete.")

    # --- trial windowing ---
    def extract_trial_data(
        self,
        align_to: str = 'Hsl',
        center_on: list[str] = ['Lrg', 'Sml'],
        trial_bounds: Tuple[float, float] = (-10.0, 5.0),
        baseline_bounds: Tuple[float, float] = (-5, -1),
        event_tolerences: Dict[str, Tuple[float, float]] = {'Lrg' : (5, 18), 'Sml' : (5, 18), 'Zap': (4.5, 18.5)},
        normalization: Literal['zscore', 'zero'] = 'zscore',
        check_overlap: bool = False,
        poor_signal_threshold: float = 0.075,
        time_error_threshold: float = 0.01,
    ) -> Tuple["RDT_PhotometryData", "RDT_PhotometryData"]:
        """
        Extract trial and baseline windows for RDT and return PhotometryData objects.
        Args:
            align_to (str): Event label used to align trials.
            center_on (list[str]): Event labels used to refine trial centers.
            trial_bounds (tuple[float, float]): Trial window bounds relative to center.
            baseline_bounds (tuple[float, float]): Baseline window bounds relative to center.
            event_tolerences (dict[str, tuple[float, float]]): Time tolerances for event annotation.
            normalization (Literal['zscore', 'zero']): Trial normalization method.
            check_overlap (bool): Whether to throw an error multiple ``center_on`` events are found in the same trial.
            poor_signal_threshold (float): Threshold for poor signal checking.
            time_error_threshold (float): Threshold on timing error for sanity check.
        Returns:
            tuple[RDT_PhotometryData, RDT_PhotometryData]: Trials and baselines as RDT_PhotometryData objects.
        """        
        # trim last Hsl event (I believe it is a dumby event)
        self.events[align_to] = self.events[align_to][:-1]
        
        # feed into parent fucntion
        super().extract_trial_data(
            align_to=align_to,
            center_on=center_on,
            trial_bounds=trial_bounds,
            baseline_bounds=baseline_bounds,
            event_tolerences=event_tolerences,
            normalization=normalization,
            check_overlap=check_overlap,
            time_error_threshold=time_error_threshold,
        )

        self.poorSignalFlag = self.median_centered_abs_max_check(
            self.raw_trial_signal, 
            threshold=poor_signal_threshold
        )

        n_trials = len(self.events[align_to])
        trial_num = np.arange(n_trials) + 1

        trial_obs = pd.DataFrame(self.trial_events)
        trial_obs['trial_num'] = trial_num
        trial_obs['poorSignalFlag'] = self.poorSignalFlag
        baseline_obs = pd.DataFrame(self.baseline_events)
        baseline_obs['trial_num'] = trial_num

        trial_metadata = {
            'aligned' : align_to,
            'lower_bound' : trial_bounds[0],
            'upper_bound' : trial_bounds[1],
            'frquency' : self.frequency,
            'normalization' : normalization,
        }
        baseline_metadata = {
            'aligned' : align_to,
            'lower_bound' : baseline_bounds[0],
            'upper_bound' : baseline_bounds[1],
            'frquency' : self.frequency,
            'normalization' : normalization,
        }

        # package results as RDT_PhotometryData objects
        trials: "RDT_PhotometryData" = RDT_PhotometryData.from_arrays(
            obs=trial_obs,
            data=self.trial_signals,
            time_points=self.trial_time_points,
            metadata=trial_metadata,
        )
        baselines: "RDT_PhotometryData" = RDT_PhotometryData.from_arrays(
            obs=baseline_obs,
            data=self.baseline_signals,
            time_points=self.baseline_time_points,
            metadata=baseline_metadata,
        )

        return trials, baselines
        
    def parse_notes_file(self, filename: str = "Notes.txt") -> None:
        """
        Parse the TDT notes file and populate self.metadata.
        Args:
            filename (str): Notes file name relative to the data folder.
        Returns:
            None
        """
        path = os.path.join(self.data_folder, filename)
        if not os.path.exists(path):
            self.metadata = {"box": self.box}
            return

        meta = {"box": self.box}
        with open(path, "r") as f:
            for line in f:
                if line.startswith("Experiment:"):
                    meta["experiment"] = line.split(":", 1)[1].strip()
                elif line.startswith("Subject:"):
                    meta["subject"] = line.split(":", 1)[1].strip()
                elif line.startswith("User:"):
                    meta["user"] = line.split(":", 1)[1].strip()
                elif line.startswith("Start:"):
                    m = re.match(r"Start:\s*([\d:apmAPM]+)\s+(\d{1,2}/\d{1,2}/\d{4})", line)
                    if m:
                        meta["start_time"] = m.group(1)
                        date = m.group(2)
                        meta["date"] = date
                        meta['stripped_date'] = date[0:2] + date[3:5] + date[8:10]
                elif line.startswith("Stop:"):
                    m = re.match(r"Stop:\s*([\d:apmAPM]+)\s+(\d{1,2}/\d{1,2}/\d{4})", line)
                    if m:
                        meta["stop_time"] = m.group(1)
                        meta["stop_date"] = m.group(2)
                # Box-specific animal/current line
                elif re.search(r"[Nn]ote", line):
                    m = re.search(rf'(?:[Bb]ox|\s*)\s*{self.box}\s*[: ;]\s*(\w+|\w+\s*\d+)\s*RDT\s*(\d+)(?:\s*uA)?', line)
                    if m:
                        meta["rat"] = m.group(1).replace(' ', '')
                        meta["current"] = int(m.group(2))
                    else:
                        raise Warning('Rat and current information not found in notes file.')
        
        self.metadata = meta
    
def RDT_process_whole_directory(
    data_dir: str,
    output_dir: str,
    log_file: str | None = None,
    fit_photobleaching: bool = True,
    save_baselines: bool = True,
    save_dashboards: bool = False,

    trial_data_file: str = 'trials.h5ad',
    baseline_data_file: str = 'baselines.h5ad',
    photobleach_curve_file: str = 'fit_photobleaching_curve.csv',
    dashboard_folder: str = 'dashboard_plots',

    boxes: List[str] = ['A', 'B'],
    event_labels: list[str] = ('Lrg', 'Sml', 'Hsl', 'Zap'),
    signal_label: str = '_465',
    isosbestic_label: str = '_405',
    notes_filename: str = 'Notes.txt',

    pipeline_kwargs: Dict[str, Any] = {},
    photobleaching_fit_kwargs: Dict[str, Any] = {}
    ) -> "RDT_PhotometryData":
    """
    Runs full RDT pipeline on all folders in a directory and combines the results.
    Args:
        data_dir (str): Directory containing the TDT data folders.
        output_dir (str): Directory to save processed data in.
        log_file (str) : Path to log file.
        fit_photobleaching (bool) : Whether to fit a photobleaching curve to raw signal.
        save_baselines (bool) : Whether to save trial-wise baseline data.
        save_dashboards (bool) : Whether to save graphical dashboard for each experiment.

        trial_data_file (str) : Name for trial data output file.
        baseline_data_file (str) : Name for baseline data output file.
        photobleach_curve_file (str) : Name for photobleach curve data output file.
        dashboard_folder (str) : Name for folder that dashboards will be saved to in ``output_dir``.

        boxes (str): TDT box identifiers to extract data from.
        event_labels (list[str]): Event labels to extract for RDT.
        signal_label (str): Base label for the signal stream.
        isosbestic_label (str): Base label for the isosbestic stream.
        notes_filename (str): File name of the notes file in the data folder.

        pipeline_kwargs (dict[str, Any]): Arguments to be passed to ``run_pipeline()``, see function for more details.
        photobleaching_fit_kwargs (dict[str, Any]): Arguments to be passed to ``fit_photobleaching_curve()``, see function for more details.
    Returns:
        RDT_PhotometryData object containing all trials extracted
    """
    # set up logger
    log = logging.getLogger(__name__)
    if log_file is not None:
        logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO)

    log.info("Starting data ripping process...")

    # create list of tdt data folders, ignore non-directories
    tdt_folders_list = [os.path.join(data_dir, foldername) for foldername in os.listdir(data_dir)]
    tdt_folders_list = [folderpath for folderpath in tdt_folders_list if not os.path.isfile(folderpath)]

    # concat savefiles
    trial_data_path = os.path.join(output_dir, trial_data_file)
    baseline_data_path = os.path.join(output_dir, baseline_data_file)
    photobleach_curve_path = os.path.join(output_dir, photobleach_curve_file)
    dashboard_folder_abs = os.path.join(output_dir, dashboard_folder)

    # delete previous files (only if they are not folders)
    for path in [trial_data_path, baseline_data_path, photobleach_curve_path]:
        if os.path.isfile(path) and os.path.exists(path): os.remove(path)

    # create dashboard folder if needed
    if save_dashboards and (not os.path.exists(dashboard_folder_abs)):
        os.mkdir(dashboard_folder_abs)
    
    # init tracking metrics
    n_experiments = int(len(boxes)*len(tdt_folders_list))
    n_poorSignal = 0
    n_errors = 0
    i = 1    

    # loop through every TDT folder add box
    for tdt_folder in tdt_folders_list:
        for box in boxes:
            log.info(f"Processing {tdt_folder}, box {box} ({i} / {n_experiments})...")
            i += 1
            try: 
                exp = RDT_PhotometryExperiment(
                    tdt_folder, 
                    box=box,
                    event_labels=event_labels,
                    signal_label=signal_label,
                    isosbestic_label=isosbestic_label,
                    notes_filename=notes_filename
                    )
                exp.run_pipeline(
                    logger=log,
                    **pipeline_kwargs
                    )

                if exp.poorSignalFlag:
                    n_poorSignal += 1
                    
                log.info(f"Saving trial data...")
                exp.trials.append_on_disk_h5ad(path=str(trial_data_path))

                if save_baselines:
                    log.info(f"Saving baseline data...")
                    exp.baselines.append_on_disk_h5ad(path=str(baseline_data_path))

                if fit_photobleaching:
                    log.info(f"Fitting & saving photobleaching curve...")
                    photobleach_fit, _ = exp.fit_photobleaching_curve(return_curve=False, **photobleaching_fit_kwargs)
                    append_to_csv(str(photobleach_curve_path), photobleach_fit)
                
                if save_dashboards:
                    log.info(f"Plotting and saving dashboard...")
                    save_path = os.path.join(dashboard_folder_abs, getattr(exp, 'id', 'Unnamed') + '.svg')
                    exp.dashboard(save=save_path)

                log.info(f"Experiment info for {exp.id}")
                log.info(exp.trials.info())
                
                del exp

            except Exception as e:
                n_errors += 1
                log.error(f"Error processing {tdt_folder}, box {box}: \n\t {e}\n")
                continue
    
    log.info(f'Data processing complete with {n_errors} errors '
             f'and {n_poorSignal} poor signals out of '
             f'{n_experiments} experiments')

    log.info(f'Reseting index of trial data...')
    trial_data = RDT_PhotometryData.read_h5ad(trial_data_path)
    trial_data.adata.obs.reset_index(drop=True, inplace=True)
    trial_data.adata.obs.index = trial_data.adata.obs.index.astype(str)
    trial_data.write_h5ad(trial_data_path)
    log.info(f'All done!')
    return trial_data