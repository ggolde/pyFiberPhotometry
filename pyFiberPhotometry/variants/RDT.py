from __future__ import annotations
from typing import Dict, Sequence, Tuple, Union, Literal, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import logging

from ..core import PhotometryData, PhotometryExperiment
from ..utils import *

ArrayLike = Union[np.ndarray, Sequence[float]]

class RDT_PhotometryData(PhotometryData):
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
        free_trial_slices: Sequence[Sequence[int]] = ((9, 28), (37, 56), (65, 84)),
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
        obs["block"] = None
        obs["block_num"] = None
        obs["forced"] = True

        for (b_lo, b_hi), b_num, b_lab, (f_lo, f_hi) in zip(
            block_slices, range(1, n_blocks + 1), block_labels, free_trial_slices
        ):
            obs.loc[obs.index[b_lo:b_hi], 'block_num'] = b_num
            obs.loc[obs.index[b_lo:b_hi], 'block'] = b_lab
            obs.loc[obs.index[f_lo:f_hi], 'forced'] = False

        self.adata.obs = obs

    def label_trials(self) -> None:
        """
        Label trials as Sml, Pun, or UnPun based on event timestamps.
        Args:
            None
        Returns:
            None
        """        
        obs = self.adata.obs
        obs["trial_label"] = None

        is_sml = obs.get("Sml", pd.Series(index=obs.index)).notna()
        is_lrg = obs.get("Lrg", pd.Series(index=obs.index)).notna()
        is_zap = obs.get("Zap", pd.Series(index=obs.index)).notna()

        obs.loc[is_sml, "trial_label"] = "Sml"
        obs.loc[is_lrg & is_zap, "trial_label"] = "Pun"
        obs.loc[is_lrg & ~is_zap, "trial_label"] = "UnPun"
        obs.loc[is_lrg & is_sml, "trial_label"] = None
        # ensure all Sml trials have no Zap timestamp
        obs.loc[is_sml & is_zap, "Zap"] = None

    # --- quality control ---
    def quality_control(self, abs_zscore_threshold: float = 0.1, drop: bool = False) -> None:
        """
        Run simple QC checks on trials and optionally drop bad trials.
        Args:
            abs_zscore_threshold (float): Threshold on mean |z| for poor signal flag.
            drop (bool): If True, drop trials that fail QC.
        Returns:
            None
        """        
        qc = {}
        self.poorSignalFlag = (np.mean(np.abs(self.X)) < abs_zscore_threshold)

        qc['before'] = self.trial_type_summary()
        qc['report'] = qc['before']

        mask = np.full(self.n_trials, True, dtype=bool)
        if 'Hsl' in self.obs:
            missing_Hsl = self.obs['Hsl'].isna()
            qc['missing_Hsl'] = int(missing_Hsl.sum())
            mask |= missing_Hsl
        
        if ('Lrg' in self.obs) and ('Sml' in self.obs):
            missing_Lrg = self.obs['Lrg'].isna()
            missing_Sml = self.obs['Sml'].isna()
            missing_lever = missing_Lrg & missing_Sml
            qc['missing_lever'] = int(missing_lever.sum())
            mask |= missing_lever
        
        if 'forced' in self.obs:
            forced_trials = self.obs['forced']
            qc['forced_trials'] = int(forced_trials.sum())
            mask |= forced_trials

        if drop:
            self.filter_rows(self, mask, inplace=drop)
            qc['after'] = self.trial_type_summary()
            qc['report'] = qc['after']

        self.qc = qc

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

    def info(self, info_on: list[str] = ['rat', 'trial_label', 'block', 'current', 'experiment', 'free']) -> None:
        """
        Build a simple string summary of key obs columns.
        Args:
            info_on (list[str]): Observation columns to summarize.
        Returns:
            str: Multi-line string with counts and unique values per column.
        """        
        info_str = "\n"
        for col in info_on:
            info_str += f'\t{col}: {self.get_text_value_counts(col) if col in self.df else "NA"}, {self.df[col].unique().size if col in self.df else "NA"} unique\n'
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
        self.id = f"{self.metadata.get('Rat', 'UnknownRat')}_{self.metadata.get('Current', 'UnknownCurrent')}uA_{self.metadata.get('Stripped_date', 'UnknownDate')}"
                
    # pipeline API
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
        time_error_threshold: float = 0.01,

        n_blocks: int = 3,
        free_trial_slices: list[list[int]] = ((9, 28), (37, 56), (65, 84)),
        block_labels: list[str] = ("0%", "25%", "75%"),
        qc_drop: bool = False,
        abs_zscore_threshold: float = 0.1,
        to_trim: list[str] = ['Lrg', 'Sml'],
        save_as: str = None, 
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
            time_error_threshold (float): Threshold on timing error for sanity check.
            n_blocks (int): Number of RDT blocks.
            free_trial_slices (list[list[int]]): Free trial index ranges per block.
            block_labels (list[str]): Labels for each block.
            qc_drop (bool): If True, drop QC-failed trials.
            abs_zscore_threshold (float): Threshold on mean |z| for poor signal flag.
            to_trim (list[str]): Obs columns to drop from final dataset.
            save_as (str | None): Directory to save output to; if None, do not save.
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
        log.info(f"Done. Fitted isosbestic R2 = {self.metadata['Isosbestic_fit']['r2_val']:.4f}")

        # step 3: extract per-trial data
        log.info(f"Extracting per-trial data...")
        self.trials, self.baselines = self.extract_trial_data(
            align_to=align_to,
            center_on=center_on,
            trial_bounds=trial_bounds,
            baseline_bounds=baseline_bounds,
            event_tolerences=event_tolerences,
            normalization=normalization,
            time_error_threshold=time_error_threshold
        )
        self.trials: "RDT_PhotometryData"
        log.info(f"Done. Extracted {self.trials.n_trials} trials of {self.trials.n_times} size each.")

        # step 4: annotate and clean per-trial data
        log.info(f"Annotating and cleaning trial data...")
        self.trials.label_trials()
        self.trials.label_blocks(
            n_blocks=n_blocks, free_trial_slices=free_trial_slices, block_labels=block_labels
        )
        self.trials.quality_control(
            drop=qc_drop, abs_zscore_threshold=abs_zscore_threshold
        )

        # step 5: pass down important metadata
        self.trials.add_obs_columns(self.metadata, keys=['rat', 'box', 'current', 'date'])
        self.trials.drop_obs_columns(to_drop=to_trim)
        log.info(f"Done. {self.trials.n_trials} trials remain.")

        if self.trials.poorSignalFlag:
            log.warning(f"Warning: Poor signal quality detected in {self.id} (mean z-score {np.mean(np.abs(self.trials.X))} < threshold {abs_zscore_threshold}).")

        if save_as is not None:
            log.info(f"Saving to {save_as} as {self.id}...")
            basepath = os.path.join(save_as, self.id)
            self.trials.write(basepath=basepath)
            write_dict_to_txt(self.trials.qc, basepath + '.qc')

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
        time_error_threshold: float = 0.01,
    ) -> None:
        """
        Extract trial and baseline windows for RDT and return PhotometryData objects.
        Args:
            align_to (str): Event label used to align trials.
            center_on (list[str]): Event labels used to refine trial centers.
            trial_bounds (tuple[float, float]): Trial window bounds relative to center.
            baseline_bounds (tuple[float, float]): Baseline window bounds relative to center.
            event_tolerences (dict[str, tuple[float, float]]): Time tolerances for event annotation.
            normalization (Literal['zscore', 'zero']): Trial normalization method.
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
            time_error_threshold=time_error_threshold
        )

        n_trials = len(self.events[align_to])
        trial_num = np.arange(n_trials) + 1

        trial_obs = pd.DataFrame(self.trial_events)
        trial_obs['trial_num'] = trial_num
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
        
        self.metadata = meta
    
    def sanity_check(self, basepath=None):
        """
        Plot raw signal and isosbestic traces for a quick visual sanity check.
        Args:
            basepath (str | None): If provided, save the plot to basepath + '.png'.
        Returns:
            matplotlib.figure.Figure: Figure containing the sanity check plot.
        """        
        title = self.metadata['experiment'] + '_' + self.metadata['rat'] + '_' + self.metadata['box']
        fig = plt.figure(figsize=(10,8))
        plt.plot(self.time, self.raw_signal, c='tab:blue', label='signal')
        plt.plot(self.time, self.raw_isosbestic, c='black', alpha=0.8, label='isobestic')
        plt.title(title)
        plt.legend()
        if basepath is not None:
            plt.savefig(basepath + '.png')
        return fig