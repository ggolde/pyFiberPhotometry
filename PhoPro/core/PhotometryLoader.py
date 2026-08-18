"""Load continuous photometry recordings into experiment objects."""

from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Callable, Literal

from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import h5py
import json
import yaml
import tdt
import os

from .PhotometryExperiment import PhotometryExperiment
from ..utils.operations import downsample_signal, downsample_time
from ..types import EventEncoding

AnnotationHandler = Callable[[str, "PhotometryLoader"], dict[str, Any]]

#region --- EVENT DECODING ---

class EventDecoder:
    """Convert storage-independent event arrays into timestamp arrays."""

    @classmethod
    def decode(
            cls,
            events: Mapping[str, ArrayLike],
            encoding: EventEncoding,
            *,
            time: ArrayLike | None = None,
            ) -> dict[str, np.ndarray]:
        """Decode raw event streams using a built-in or custom strategy."""
        decoded_time = None if time is None else cls._as_1d_array(time, name='time')

        match encoding:
            case func if callable(func):
                decoded = func(events, decoded_time)
                if not isinstance(decoded, Mapping):
                    raise TypeError('A custom event decoder must return a mapping.')
                return cls._normalize_timestamps(decoded)
            case 'binary':
                return cls._decode_binary(events, decoded_time)
            case 'timestamp':
                return cls._normalize_timestamps(events)
            case 'long':
                return cls._decode_long(events, decoded_time)
            case _:
                raise ValueError(f'Event encoding {encoding!r} not recognized.')

    @staticmethod
    def _as_1d_array(values: ArrayLike, *, name: str) -> np.ndarray:
        array = np.asarray(values)
        if array.ndim != 1:
            raise ValueError(f'{name} must be one-dimensional; got shape {array.shape}.')
        return array

    @classmethod
    def _decode_binary(
            cls,
            events: Mapping[str, ArrayLike],
            time: np.ndarray | None,
            ) -> dict[str, np.ndarray]:
        if time is None:
            raise ValueError('Time must be present when decoding binary events.')

        decoded = {}
        for label, values in events.items():
            raw = cls._as_1d_array(values, name=f'event {label!r}')
            if raw.size != time.size:
                raise ValueError(
                    f'Binary event {label!r} has {raw.size} values but time has '
                    f'{time.size} values.'
                )
            present = np.asarray(pd.Series(raw).fillna(False), dtype=bool)
            decoded[str(label)] = np.asarray(time[present], dtype=np.float64)
        return decoded

    @classmethod
    def _normalize_timestamps(
            cls,
            events: Mapping[str, ArrayLike],
            ) -> dict[str, np.ndarray]:
        decoded = {}
        for label, values in events.items():
            raw = cls._as_1d_array(values, name=f'event {label!r}')
            timestamps = np.asarray(raw[~pd.isna(raw)], dtype=np.float64)
            decoded[str(label)] = timestamps
        return decoded

    @classmethod
    def _decode_long(
            cls,
            events: Mapping[str, ArrayLike],
            time: np.ndarray | None,
            ) -> dict[str, np.ndarray]:
        if time is None:
            raise ValueError('Time must be present when decoding long events.')
        if len(events) != 1:
            raise ValueError('Long event encoding requires exactly one label stream.')

        source_name, values = next(iter(events.items()))
        labels = cls._as_1d_array(values, name=f'event {source_name!r}')
        if labels.size != time.size:
            raise ValueError(
                f'Long event label stream has {labels.size} values but time has '
                f'{time.size} values.'
            )

        grouped: dict[str, list[float]] = {}
        for label, timestamp in zip(labels, time):
            if pd.isna(label) or pd.isna(timestamp):
                continue
            if isinstance(label, bytes):
                label = label.decode()
            grouped.setdefault(str(label), []).append(float(timestamp))

        return {
            label: np.asarray(timestamps, dtype=np.float64)
            for label, timestamps in grouped.items()
        }

##############################
#region --- ABSTRACT CLASS ---
##############################
class PhotometryLoader(ABC):
    """Abstract base class for photometry data loaders."""

    def load(self, exp_cls: type[PhotometryExperiment] = PhotometryExperiment) -> PhotometryExperiment:
        """Load data and return a `PhotometryExperiment` instance.

        Parameters
        ----------
        exp_cls : type[PhotometryExperiment], default=PhotometryExperiment
            Experiment class used to construct the loaded object.

        Returns
        -------
        PhotometryExperiment
            Loaded experiment object.
        """
        data = self.extract_data()
        return exp_cls(**data)

    @abstractmethod
    def extract_data(self) -> dict[str, Any]:
        """Extract loader-specific data for `PhotometryExperiment` construction.

        Returns
        -------
        dict[str, Any]
            Keyword arguments accepted by `PhotometryExperiment`.
        """
        pass

    def read_annotation(
            self,
            file: str | None,
            handler: Literal['json', 'yaml'] | AnnotationHandler,
            parent_key: str | None = None
            ) -> dict:
        """Load experiment metadata from annotation file.

        Parameters
        ----------
        file : str or None
            Path to the annotation file. If ``None``, an empty dictionary is
            returned.
        handler : {'json', 'yaml'} or AnnotationHandler
            Built-in annotation format name or a callable that accepts the file
            path and this loader and returns a dictionary.
        parent_key : str or None, default=None
            Optional top-level key to select from the loaded annotation
            dictionary.

        Returns
        -------
        dict
            Loaded annotation metadata.

        Raises
        ------
        ValueError
            If the file is missing, the handler is unknown, ``parent_key`` is
            absent, or the loaded annotations are not a dictionary.
        """
        # check input
        if file is None:
            return {}
        elif not os.path.exists(file):
            raise ValueError(f'Annotation file {file} does not exsit.')

        # use handler to load annotations
        match handler:
            case func if callable(func):
                annots = handler(file, self)
            case 'json':
                with open(file, 'r') as f:
                    annots = json.load(f)
            case 'yaml':
                with open(file, 'r') as f:
                    annots = yaml.load(f, Loader=yaml.SafeLoader)
            case _:
                raise ValueError(f'Annotation handler {handler} not recognized!')

        # use parent key if specified
        if parent_key is not None:
            if parent_key not in annots:
                raise ValueError(f'{parent_key} is not a key in the full annotation file.')

            annots = annots[parent_key]

        # validate handler output
        if not isinstance(annots, dict):
            raise ValueError(f'Loaded annotations are not a dictionary. It is type {type(annots)}')

        return annots

#endregion

###################
#region --- TDT ---
###################
class TDTLoader(PhotometryLoader):
    """Extract photometry data from TDT folder format."""

    def __init__(
            self,
            data_folder: str,
            parent_key: str | None,
            event_labels: list[str],
            signal_label: str,
            isosbestic_label: str | None,
            downsample: int | None = None,
            downsample_kwargs: dict = {},
            annotation_file: str | None = None,
            annotation_handler: Literal['json', 'yaml'] | AnnotationHandler = 'json',
            ):
        """Initialize a TDT photometry loader.

        Parameters
        ----------
        data_folder : str
            Path to the TDT block folder.
        parent_key : str or None
            TDT sub-storage identifier appended to stream and epoc labels. If ``None``,
            no parent key is used.
        event_labels : list[str]
            Event labels to extract from epocs.
        signal_label : str
            Base label for the signal stream.
        isosbestic_label : str
            Base label for the isosbestic stream. If ``None``, no isosbestic
            signal is extracted.
        downsample : int or None, default=None
            Downsampling factor for raw streams. If ``None``, no downsampling
            is performed.
        downsample_kwargs : dict, default={}
            Additional keyword arguments passed to the downsampling helpers.
        annotation_file : str or None, default=None
            Annotation filename inside ``data_folder``. The loaded annotation
            dictionary is indexed by ``parent_key`` before metadata are added.
        annotation_handler : {'json', 'yaml'} or AnnotationHandler, default='json'
            Built-in annotation format name or custom annotation reader.
        """
        self.data_folder = data_folder

        # handle parent key
        self.parent_key = parent_key
        self.parent_key_str = '' if parent_key is None else parent_key

        self.signal_label = signal_label
        self.isosbestic_label = isosbestic_label
        self.event_labels = list(event_labels)
        self.downsample = 1 if downsample is None else downsample
        self.downsample_kwargs = downsample_kwargs

        self.annotation_file = annotation_file
        self.annotation_handler = annotation_handler

        self.metadata = {'source' : str(self.data_folder), 'parent_key' : self.parent_key_str}

    # --- data extraction ---
    def extract_data(self) -> dict[str, Any]:
        """Load data from TDT and extract streams and events.

        Downsamples the signal and isosbestic streams before packaging them
        into the experiment input dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary containing raw signal, raw isosbestic signal, time
            vector, sampling frequency, events, and metadata.
        """
        tdt_obj = tdt.read_block(self.data_folder, verbose=False)

        # rip data out of TDT object
        sig = tdt_obj.streams[self.signal_label + self.parent_key_str].data
        fs = tdt_obj.streams[self.signal_label + self.parent_key_str].fs
        start_time = tdt_obj.streams[self.signal_label + self.parent_key_str].start_time

        # downsample raw signal
        raw_signal: np.ndarray = downsample_signal(np.asarray(sig, dtype=np.float64), factor=self.downsample, **self.downsample_kwargs)

        # handle isosbestic if present
        if self.isosbestic_label is not None:
            iso = tdt_obj.streams[self.isosbestic_label + self.parent_key_str].data
            raw_isosbestic: np.ndarray = downsample_signal(np.asarray(iso, dtype=np.float64), factor=self.downsample, **self.downsample_kwargs)
        else:
            raw_isosbestic = None

        # contruct time
        n_times = sig.size
        raw_time = start_time + (np.arange(n_times, dtype=float) / float(fs))
        time = downsample_time(np.asarray(raw_time, dtype=np.float64), factor=self.downsample, **self.downsample_kwargs)
        frequency = (time.size - 1) / (time[-1] - time[0])

        # extract events
        events = self.extract_events(tdt_obj)
        del tdt_obj

        # load annotations
        if self.annotation_file is not None:
            annotation_fpath = os.path.join(self.data_folder, self.annotation_file)
            annots = self.read_annotation(file=annotation_fpath, handler=self.annotation_handler, parent_key=self.parent_key)
            self.metadata.update(annots)

        data = dict(
            raw_signal=raw_signal,
            raw_isosbestic=raw_isosbestic,
            time=time,
            frequency=frequency,
            events=events,
            metadata=self.metadata
        )
        return data

    # --- event extraction ---
    def extract_events(self, tdt_obj) -> dict:
        """Extract event timestamps from a TDT block object.

        Parameters
        ----------
        tdt_obj
            Object returned by `tdt.read_block`.

        Returns
        -------
        dict
            Mapping from requested event labels to onset timestamp arrays.
        """
        # extract event timestamps for requested labels
        events = {}
        self.metadata['missing_events'] = []
        for label in self.event_labels:
            # some sessions may lack a label entirely if no events are recorded
            if hasattr(tdt_obj.epocs, self.parent_key_str + label):
                ep = tdt_obj.epocs[self.parent_key_str + label]
                events[label] = np.asarray(ep.onset)
            else:
                events[label] = np.array([], dtype=float)
                self.metadata['missing_events'].append(label)
        return events

#endregion

###################
#region --- CSV ---
###################
class CSVLoader(PhotometryLoader):
    """Extract photometry data from CSV-based inputs."""

    def __init__(
            self,
            csv: str,
            time_col: str = 'time',
            signal_col: str = 'signal',
            isosbestic_col: str | None = 'isosbestic',
            event_file: str | None = None,
            event_cols: str | list[str] | None = None,
            event_encoding: EventEncoding | None = None,
            downsample: int | None = None,
            downsample_kwargs: dict = {},
            annotation_file: str | None = None,
            annotation_handler: Literal['json', 'yaml'] | AnnotationHandler = 'json',
            ) -> None:
        """Initialize a CSV photometry loader.

        Parameters
        ----------
        csv : str
            Path to the CSV file containing photometry data.
        time_col : str, default='time'
            Column containing time values.
        signal_col : str, default='signal'
            Column containing signal values.
        isosbestic_col : str or None, default='isosbestic'
            Column containing isosbestic values. If absent from the CSV, the
            loaded experiment is single-channel.
        event_file : str or None, default=None
            A path to a seperate CSV file containing events as columns and
            timestamps as row values. If None, the event columns are assumed to
            be in the same file as the signal values and encoded in binary-like
            values.
        event_cols : str, list[str], or None, default=None
            If ``event_file`` is None: column or columns containing truth-like
            values where events occur in the same file as the signal values. If
            ``event_file`` is not None: the column names in the event files whose
            values are timestamps.
        event_encoding : EventEncoding or None, default=None
            Encoding of events in a CSV file.

            - ``binary``: true-false like values indicating whether the event
            occured at the timestamp of an aligned columns containing sampling times
            - ``timestamps``: timestamp values of when the event occured relative to
            the experiments time series with columns serving as event names
            - ``long``: like ``timestamps`` but in long format, where the time column serves
            as the timestamps and a single event column contains event labels
            - A custom function that accepts a mapping of event labels to raw
            arrays and an optional time array, and returns an event-label to
            timestamps mapping.
            - If ``None``, binary encoding is used for events in ``csv`` and
            timestamp encoding is used when ``event_file`` is provided.
        downsample : int or None, default=None
            Downsampling factor for the raw arrays. If ``None``, no
            downsampling is performed.
        downsample_kwargs : dict, default={}
            Additional keyword arguments passed to the downsampling helpers.
        annotation_file : str or None, default=None
            Path to an annotation file to merge into metadata.
        annotation_handler : {'json', 'yaml'} or AnnotationHandler, default='json'
            Built-in annotation format name or custom annotation reader.
        """
        # save fpaths and params
        self.csv = csv
        self.time_col = time_col
        self.sig_col = signal_col
        self.iso_col = isosbestic_col

        if isinstance(event_cols, str):
            event_cols = [event_cols]
        self.event_cols = event_cols
        self.event_file = event_file
        if event_encoding is None:
            event_encoding = 'binary' if event_file is None else 'timestamp'
        self.event_encoding: EventEncoding = event_encoding

        self.downsample = downsample
        self.downsample_kwargs = downsample_kwargs

        self.annotation_file = annotation_file
        self.annotation_handler = annotation_handler

        self.metadata = {'source' : str(self.csv)}

    def extract_data(self) -> dict[str, Any]:
        """Load signal, time, and event data from CSV and JSON files.

        Returns
        -------
        dict[str, Any]
            Dictionary containing raw signal, optional raw isosbestic signal,
            time vector, events, and metadata.

        Raises
        ------
        ValueError
            If the configured signal column is absent.
        KeyError
            If the configured time column or event columns are absent.
        """
        # load csv
        df = pd.read_csv(self.csv)

        # load required
        if self.sig_col in df:
            raw_signal = downsample_signal(df[self.sig_col].to_numpy(), factor=self.downsample, **self.downsample_kwargs)
        else:
            raise ValueError(f"Column for signal timepoints ({self.sig_col}) is not in {self.csv}")

        if self.time_col in df:
            time = downsample_time(df[self.time_col].to_numpy(), factor=self.downsample, **self.downsample_kwargs)
        else:
            raise KeyError(f"Column for timepoints ({self.time_col}) not found in CSV.")

        # load optional
        if self.iso_col in df:
            raw_isosbestic = downsample_signal(df[self.iso_col].to_numpy(), factor=self.downsample, **self.downsample_kwargs)
        else:
            raw_isosbestic = None

        # extract events
        events = self._extract_events(df=df)

        # load annotations
        if self.annotation_file is not None:
            annots = self.read_annotation(file=self.annotation_file, handler=self.annotation_handler, parent_key=None)
            self.metadata.update(annots)

        # package results
        data = dict(
            raw_signal = raw_signal,
            raw_isosbestic = raw_isosbestic,
            time = time,
            frequency = None,
            events = events,
            metadata = self.metadata,
        )
        return data

    def _extract_events(
            self,
            df: pd.DataFrame,
            ) -> dict[str, np.ndarray]:
        event_df = df if self.event_file is None else pd.read_csv(self.event_file)

        # Resolve CSV-specific column selection before handing neutral arrays to
        # the shared decoder.
        exclude_cols = [self.time_col, self.sig_col, self.iso_col]
        if self.event_cols is None:
            event_cols = [col for col in event_df.columns if col not in exclude_cols]
        else:
            event_cols = self.event_cols
            missing = [col for col in event_cols if col not in event_df.columns]
            if missing:
                raise KeyError(f'Event columns ({missing}) not found in CSV containing events.')

        needs_time = isinstance(self.event_encoding, str) and self.event_encoding in ('binary', 'long')
        if needs_time and self.time_col not in event_df:
            raise KeyError(
                f'Column for timepoints ({self.time_col}) not found in CSV '
                'containing events.'
            )

        raw_events = {col: event_df[col].to_numpy() for col in event_cols}
        event_time = event_df[self.time_col].to_numpy() if self.time_col in event_df else None
        return EventDecoder.decode(
            raw_events,
            self.event_encoding,
            time=event_time,
        )

###########################
#region --- DoricLoader ---
###########################

class H5Loader(PhotometryLoader):
    def __init__(
            self,
            file: str,
            time_path: str,
            signal_path: str,
            isosbestic_path: str | None = None,
            event_paths: dict[str, str] | None = None,
            event_encoding: EventEncoding | None = None,
            downsample: int | None = None,
            downsample_kwargs: dict = {},
            annotation_file: str | None = None,
            annotation_handler: Literal['json', 'yaml'] | AnnotationHandler = 'json',
            ) -> None:

        self.file = Path(file)

        self.time_path = str(time_path)
        self.signal_path = str(signal_path)
        self.isosbestic_path = None if isosbestic_path is None else str(isosbestic_path)

        self.event_paths = {} if event_paths is None else event_paths
        self.event_encoding = event_encoding
        if self.event_paths and event_encoding is None:
            raise ValueError(
                f'event_encoding must be specified if event_paths not None.'
            )

        self.downsample = downsample
        self.downsample_kwargs = downsample_kwargs

        self.annotation_file = annotation_file
        self.annotation_handler = annotation_handler

        self.metadata = {'source' : str(self.file)}

    def extract_data(self) -> dict[str, Any]:

        if not self.file.exists():
            raise ValueError(f'File {self.file} does not exist.')

        with h5py.File(self.file, mode='r') as f:
            # extract signal
            if self.signal_path in f:
                raw_signal = downsample_signal(
                    np.asarray(f[self.signal_path], dtype=np.float64),
                    factor=self.downsample,
                    **self.downsample_kwargs
                )
            else:
                raise KeyError(
                    f'Signal path {self.signal_path} not in file {self.file}.'
                )

            # extract isosbestic
            if self.isosbestic_path is None:
                raw_isosbestic = None
            elif self.isosbestic_path in f:
                raw_isosbestic = downsample_signal(
                    np.asarray(f[self.isosbestic_path], dtype=np.float64),
                    factor=self.downsample,
                    **self.downsample_kwargs
                )
            else:
                raise KeyError(
                    f'Isosbestic path {self.isosbestic_path} not in file {self.file}.'
                )

            # extract time
            if self.time_path in f:
                raw_time = np.asarray(f[self.time_path], dtype=np.float64)
                time = downsample_time(
                    raw_time,
                    factor=self.downsample,
                    **self.downsample_kwargs
                )
            else:
                raise KeyError(
                    f'Time path {self.time_path} not in file {self.file}.'
                )

            # check for missing events
            missing = [
                event_path for event_path in self.event_paths.values()
                if event_path not in f
            ]
            if missing:
                raise KeyError(
                    f'Event paths {missing} are missing from file {self.file}'
                )

            # Read storage-specific datasets, then use the same decoder as CSV.
            raw_events = {
                label: np.asarray(f[event_path])
                for label, event_path in self.event_paths.items()
            }
            events = (
                {}
                if self.event_encoding is None
                else EventDecoder.decode(
                    raw_events,
                    self.event_encoding,
                    time=raw_time,
                )
            )

        # load annotations
        if self.annotation_file is not None:
            annots = self.read_annotation(file=self.annotation_file, handler=self.annotation_handler, parent_key=None)
            self.metadata.update(annots)

        # package results
        data = dict(
            raw_signal = raw_signal,
            raw_isosbestic = raw_isosbestic,
            time = time,
            frequency = None,
            events = events,
            metadata = self.metadata,
        )

        return data
