"""Utilities for generating and loading simulated photometry libraries."""

from __future__ import annotations
from typing import Any, Callable, Self

import itertools
import importlib
import inspect
import logging
import json

import numpy as np
import pandas as pd

from pathlib import Path

from . import SimulatedPhotometry
from ..core.PhotometryLoader import PhotometryLoader

#######################
#region --- HELPERS ---
#######################
def _accepted_args_for_SimPho() -> set:
    sig = inspect.signature(SimulatedPhotometry.from_parameters)
    params = sig.parameters
    ACCEPTED_ARGS = {
        name for name, p in params.items()
        if p.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
        and name not in ()
    }
    return ACCEPTED_ARGS

def _callable_to_import_path(func: Callable) -> str:
    module = getattr(func, '__module__', None)
    qualname = getattr(func, '__qualname__', None)

    if module is None or qualname is None:
        return str(func)
    return f'{module}.{qualname}'

def _import_callable(path: str) -> Callable:
    module_name, _, qualname = path.rpartition('.')
    if module_name == '':
        raise ValueError(f'Cannot import callable from unqualified path: {path}')

    obj: Any = importlib.import_module(module_name)
    for attr in qualname.split('.'):
        obj = getattr(obj, attr)

    if not callable(obj):
        raise TypeError(f'Imported object is not callable: {path}')
    return obj

def _stringify_callables(value: Any) -> Any:
    if inspect.isfunction(value) or inspect.ismethod(value):
        return _callable_to_import_path(value)
    if isinstance(value, dict):
        return {k: _stringify_callables(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_stringify_callables(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_stringify_callables(v) for v in value)
    return value

def safe_SimPho_from_params(params: dict) -> SimulatedPhotometry:
    ACCEPTED_ARGS = _accepted_args_for_SimPho()
    params = {k : v for k, v in params.items() if k in ACCEPTED_ARGS}
    if isinstance(params.get('event_kernel'), str):
        params['event_kernel'] = _import_callable(params['event_kernel'])
    return SimulatedPhotometry.from_parameters(**params)

#endregion


#############################
#region --- LIBRARY CLASS ---
#############################
class SimulatedLibrary:
    """Store and generate libraries of simulated photometry parameters.

    Parameters
    ----------
    params : dict[str, dict[str, Any]]
        Parameter dictionary for each simulated library member. Keys are
        normalized to strings for JSON round-tripping and loader lookup.

    Attributes
    ----------
    params : dict[str, dict[str, Any]]
        Normalized parameter dictionary keyed by library identifier.
    """

    params: dict[str, dict[str, Any]]

    def __init__(
            self,
            params: dict[str, dict[str, Any]]
            ) -> None:
        """Initialize a simulated library from parameter dictionaries."""
        self.params = self._normalize_param_input(params)

    # --- PROPERTIES ---
    @property
    def n_samples(self) -> int:
        """Number of parameter sets in the library."""
        return len(self.params)
    
    # --- EXPORT ---
    def to_table(self) -> pd.DataFrame:
        def _normalize_dict_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
            normed_res = pd.json_normalize(df[col])
            normed_res.columns = [f'{col}:{s}' for s in normed_res.columns]
            return normed_res

        out = (
            pd.DataFrame(_stringify_callables(self.params)).T
            .reset_index(names='LIB_ID')
            .infer_objects()
        )
        to_norm = out.select_dtypes(['object']).columns

        for col in to_norm:
            normed = _normalize_dict_col(out, col)
            if not normed.empty:
                out.drop(columns=col, inplace=True)
                out[normed.columns] = normed

        first_cols = out.columns[out.columns.str.endswith('_ID')].to_list()
        other_cols = [col for col in out.columns if col not in first_cols]
        out = out[first_cols + other_cols]
        return out

    # --- I/O ---
    def to_json(
            self,
            fpath: str | Path
            ) -> None:
        """Write library parameters to a JSON file.

        Parameters
        ----------
        fpath : str or Path
            Destination path for the JSON parameter table.
        """
        with open(fpath, 'w') as f:
            json.dump(_stringify_callables(self.params), f, indent=4)

    @classmethod
    def from_json(
            cls,
            file: str | Path,
            ) -> Self:
        """Read library parameters from a JSON file.

        Parameters
        ----------
        file : str or Path
            Path to a JSON parameter table.

        Returns
        -------
        SimulatedLibrary
            Library initialized from the saved parameter table.
        """
        with open(file, 'r') as f:
            params = json.load(f)
        return cls(params)

    # --- HELPERS ---
    def _normalize_param_input(self, params: dict) -> dict:
        """Return parameter dictionaries keyed by strings."""
        return {str(k) : v for k, v in params.items()}

    @staticmethod
    def _validate_kwargs(
            func: Callable,
            kwargs: dict[str, Any],
            ) -> None:
        """Validate keyword arguments against a callable signature."""
        sig = inspect.signature(func)
        params = sig.parameters
        skip = {'self', 'cls'}

        accepts_var_kw = any(
            p.kind is inspect.Parameter.VAR_KEYWORD
            for p in params.values()
        )

        accepted = {
            name for name, p in params.items()
            if p.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
            and name not in skip
        }

        required = {
            name for name, p in params.items()
            if p.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
            and p.default is inspect.Parameter.empty
            and name not in skip
        }

        missing = required - kwargs.keys()
        unexpected = set() if accepts_var_kw else (kwargs.keys() - accepted)

        if missing:
            raise TypeError(f"Missing required kwargs for {func.__qualname__}(): {sorted(missing)}")
        if unexpected:
            raise TypeError(f"Unexpected kwargs for {func.__qualname__}(): {sorted(unexpected)}")

    # --- PARAM GENERATION ---
    @classmethod
    def from_permutations(
            cls,
            contanst_kwargs: dict[str, Any],
            to_permute_kwargs: dict[str, list[Any]],
            across_kwargs: dict[str, dict[str, Any]] | None = None,
            replicates: int = 1,
            seed: int | None = None,
            ) -> Self:
        """Create a library from the permutation of parameter values.

        Parameters
        ----------
        contanst_kwargs : dict[str, Any]
            Keyword arguments passed unchanged to
            ``SimulatedPhotometry.from_parameters()`` for every simulation.
        to_permute_kwargs : dict[str, list[Any]]
            Keyword arguments whose values are fully crossed to create
            parameter permutations. Each value must be a list of candidates.
        across_kwargs : dict[str, dict[str, Any]] | None or None, default=None
            Labeled dictionary of keyword arguments to iterate through making a new combination
            for each element in the list.
        replicates : int, default=1
            Number of replicate simulations generated for each permutation.
            Each replicate gets its own seed.
        seed : int or None, default=None
            Optional seed used to shuffle the generated per-simulation seeds.

        Returns
        -------
        SimulatedLibrary
            Library containing one parameter dictionary per generated sample.
        """
        # pre-validate kwargs
        cls._validate_kwargs(SimulatedPhotometry.from_parameters, contanst_kwargs)
        if to_permute_kwargs is not None:
            cls._validate_kwargs(SimulatedPhotometry.from_parameters, to_permute_kwargs)
        if across_kwargs is not None:
            for label, kwargs in across_kwargs.items():
                cls._validate_kwargs(SimulatedPhotometry.from_parameters, kwargs)

        generator = ParamPermutationGenerator(
            contanst_kwargs=contanst_kwargs,
            to_permute_kwargs=to_permute_kwargs,
            across_kwargs=across_kwargs,
            replicates=replicates,
            seed=seed,
        )

        params = generator.build()
        return cls(params)

    # --- OPERATIONS ---
    def update_params(self, to_update: dict[str, Any]) -> None:
        """Update every parameter dictionary in the library.

        Parameters
        ----------
        to_update : dict[str, Any]
            Keyword-value pairs to merge into each built parameter dictionary.
        """
        for i, args in self.params.items():
            self.params[i] = args | to_update

    def mutate_params(self, to_update: dict[str, Any]) -> Self:
        """Mutate every parameter dictionary a copy of the library.

        Parameters
        ----------
        to_update : dict[str, Any]
            Keyword-value pairs to merge into each built parameter dictionary.
        """
        params = self.params.copy()
        for i, args in params.items():
            params[i] = args | to_update
        return type(self)(params)

    # --- BUILD LIBRARY ---
    def generate_library(
            self,
            output_dir: Path,
            to_trials_kwargs: dict[str, Any],
            prefix: str = 'simlib',
            param_file: str = '_params.json',
            true_file: str = '_true_trials.h5ad',
            exp_folder: str | None = None,
            log_file: str | None = None,
            operation: Callable[[SimulatedPhotometry], None] | None = None,
            ) -> None:
        """Generate simulated experiments and combined true-signal trial data.

        Parameters
        ----------
        output_dir : Path
            Directory where generated files will be written.
        to_trials_kwargs : dict[str, Any]
            Keyword arguments passed to
            ``SimulatedPhotometry.to_PhotometryData()`` when building the true
            signal trial object.
        prefix : str, default='simlib'
            Prefix for generated per-experiment CSV files.
        param_file : str, default='_params.json'
            File name for the saved parameter table.
        true_file : str, default='_true_trials.h5ad'
            File name for the combined true-signal ``PhotometryData`` object.
        exp_folder : str or None, default=None
            The subfolder within ``output_dir`` in which per-experiment CSVs
            are saved. If ``None``, experiment CSVs are not saved.
        log_file : str or None, default=None
            Optional log file path relative to ``output_dir``.
        operation : Callable[[SimulatedPhotometry], None] or None, default=None
            Optional function applied to each generated
            ``SimulatedPhotometry`` object before CSV export and trial
            extraction. This can be used to add events or otherwise modify
            generated data.
        """

        # coerce inputs
        output_dir = Path(output_dir)
        if not output_dir.exists():
            Path(output_dir).mkdir(exist_ok=True)
        if (exp_folder is not None) and not (output_dir / exp_folder).exists():
            Path(output_dir / exp_folder).mkdir(exist_ok=True)

        # --- set up logger ---
        logger = logging.getLogger(__name__)
        if log_file is not None:
            logging.basicConfig(filename=output_dir / log_file, filemode='w', level=logging.INFO, force=True)
        logger.info(
            f'Beginning generation of {self.n_samples} simulated datasets...'
        )

        # save param file
        param_file_path = output_dir / param_file
        logger.info(f'Saving parameters to {str(param_file_path)}...\n')
        self.to_json(param_file_path)

        # iterate over all params
        i = 0
        logger.info(f'Iterating over {self.n_samples} parameters...\n')

        for lib_id, params in self.params.items():
            logger.info(f'Generating experiment {lib_id} ({i+1}/{self.n_samples})...')
            sim = safe_SimPho_from_params(params)

            if operation is not None:
                logger.info(f'Executing custom operation...')
                operation(sim)

            if exp_folder is not None:
                save_path = output_dir / exp_folder / f'{prefix}_{lib_id}.csv'
                logger.info(f'Saving generated data to {save_path}...')
                sim.to_experiment_csv(save_path)

            logger.info(f'Extracting trial data...')
            if i == 0:
               logger.info(f'Creating first trial data instance...')
               data = (
                   sim
                   .to_PhotometryData(**to_trials_kwargs)
                   .mutate_obs(LIB_ID = lib_id)
               )
               trials = data.copy()
            else:
                logger.info(f'Accumulating result...')
                data = (
                    sim
                    .to_PhotometryData(**to_trials_kwargs)
                    .mutate_obs(LIB_ID = lib_id)
                )
                trials.combine_obj(data, inplace=True) #type: ignore

            logger.info('Completed job.\n')

            # iterate
            i += 1

        # save true signal trials
        trials.write_h5ad(output_dir / true_file) #type: ignore

        # report success
        logger.info(f'Library generation complete!')

    # --- UTILITY ---
    def create_loader_args(
            self,
            operation: Callable[[SimulatedPhotometry], None] | None = None,
            as_single_channel: bool = False,
            ) -> list[dict[str, Any]]:
        """Create loader argument dictionaries for all generated parameters.

        Parameters
        ----------
        operation : Callable[[SimulatedPhotometry], None] or None, default=None
            Optional function applied after simulation in
            ``SimulatedParamLoader``.
        as_single_channel : bool, default=False
            Whether loaders should omit the isosbestic channel.

        Returns
        -------
        list[dict[str, Any]]
            Loader keyword dictionaries covering all generated parameters.
        """
        return [
            dict(key=str(key), operation=operation, as_single_channel=as_single_channel)
            for key in self.params.keys()
        ]

#endregion


##########################
#region --- GENERATORS ---
##########################
class ParamPermutationGenerator:
    """Build simulation parameter dictionaries from crossed values.

    Parameters
    ----------
    contanst_kwargs : dict[str, Any]
        Keyword arguments passed unchanged to
        ``SimulatedPhotometry.from_parameters()`` for every simulation.
    to_permute_kwargs : dict[str, list[Any]]
        Keyword arguments whose values are fully crossed to create parameter
        permutations.
    across_kwargs : dict[str, dict[str, Any]] | None or None, default=None
        Labeled dictionary of keyword arguments to iterate through making a new combination
        for each element in the list.
    replicates : int, default=1
        Number of replicate simulations generated for each permutation.
    seed : int or None, default=None
        Optional seed used to shuffle generated per-simulation seeds.

    Attributes
    ----------
    constants : dict[str, Any]
        Keyword arguments reused for every generated sample.
    to_permute : dict[str, list[Any]]
        Keyword arguments crossed to create unique parameter permutations.
    across : list[dict[str, Any]]
        List of keyword arguements to iterate through to make combinations
        within each permutation.
    replicates : int
        Number of generated samples per permutation.
    seed : int or None
        Seed for assigning reproducible per-sample seeds.
    """

    def __init__(
            self,
            contanst_kwargs: dict[str, Any],
            to_permute_kwargs: dict[str, list[Any]] | None = None,
            across_kwargs: dict[str, dict[str, Any]] | None = None,
            replicates: int = 1,
            seed: int | None = None,
            ) -> None:
        """Initialize the parameter permutation generator."""
        # assign attrs
        self.constants = contanst_kwargs
        self.to_permute = {} if to_permute_kwargs is None else to_permute_kwargs
        self.across = [{}] if across_kwargs is None else across_kwargs
        self.has_across = across_kwargs is not None
        self.replicates = replicates
        self.seed = seed

        # validate
        self._validate_across_and_permutation_kwargs()

    # --- VALIDATE ---
    def _validate_across_and_permutation_kwargs(self) -> None:
        for label, across_kwargs in self.across.items():
            key_overlap = set(across_kwargs.keys()) & set(self.to_permute.keys())
            if len(key_overlap) != 0:
                raise ValueError(
                    f'Keys cannot be shared between to_permute_kwargs and across_kwargs. '
                    f'Keys {", ".join(list(key_overlap))} are shared.'
                )

    # --- PERMUTAION ---
    def _permutation_generator(self):
        """Yield dictionaries for each crossed parameter permutation."""
        for inp in itertools.product(*self.to_permute.values()):
            yield dict(zip(self.to_permute.keys(), inp))

    def build(self) -> dict:
        """Generate parameter dictionaries with unique simulation seeds.

        Returns
        -------
        dict[int, dict[str, Any]]
            Parameter dictionaries keyed by generated sample index.
        """
        # generate permutations
        permutations = [p for p in self._permutation_generator()]
        self.n_permutations = len(permutations)
        self.n_across = len(self.across)
        self.n_samples = self.n_permutations * self.n_across * self.replicates

        # set up unique seeds
        rng = np.random.default_rng(self.seed)
        seed_bank = np.arange(1, self.n_samples + 1).astype(int)
        rng.shuffle(seed_bank)

        # set up iteration
        i = 0
        combo_id = 0
        params = {}

        for perm_id, perm in enumerate(permutations):
            for across_id, (across_label, across) in enumerate(self.across.items()):
                combo = self.constants | perm | across

                for rep_id in range(self.replicates):
                    params[i] = (
                        combo | {
                            'seed' : int(seed_bank[i]),
                            'COMBO_ID' : combo_id,
                            'ACROSS_ID' : across_id,
                            'PERM_ID' : perm_id,
                            'REP_ID' : rep_id,
                            'CONDITION_ID' : across_label
                        }
                    )
                    i += 1

                combo_id += 1

        return params

#endregion


######################
#region --- LOADER ---
######################
class SimulatedParamLoader(PhotometryLoader):
    """Load one simulated experiment from a saved parameter table.

    Parameters
    ----------
    json : str
        Path to the JSON parameter table written by ``SimulatedLibrary``.
    key : str
        Identifier of the parameter set to load.
    operation : Callable[[SimulatedPhotometry], None] or None, default=None
        Optional function applied to the generated ``SimulatedPhotometry``
        object before extracting loader data.
    as_single_channel : bool, default=False
        Whether to omit the isosbestic channel from extracted data.
    """

    def __init__(
            self,
            json: str,
            key: str,
            operation: Callable[[SimulatedPhotometry], None] | None = None,
            as_single_channel: bool = False,
            ) -> None:
        """Initialize the simulated-parameter loader.

        Parameters
        ----------
        json : str
            Path to the JSON parameter table.
        key : str
            Identifier of the parameter set to generate.
        operation : Callable[[SimulatedPhotometry], None] or None, default=None
            Optional function applied to the generated simulator before data
            extraction.
        as_single_channel : bool, default=False
            Whether to omit the isosbestic channel.
        """
        self.json = json
        self.key = key
        self.operation = operation
        self.as_single_channel = as_single_channel

    def extract_data(self) -> dict[str, Any]:
        """Generate and return raw photometry data for the selected parameter set.

        Returns
        -------
        dict[str, Any]
            Dictionary containing raw signal, optional raw isosbestic signal,
            time values, event onsets, and metadata.
        """
        with open(self.json, 'r') as f:
            params: dict = json.load(f)
        params = params[self.key]

        sim = safe_SimPho_from_params(params)

        metadata: dict = params | {
            'source' : str(self.json),
            'LIB_ID' : str(self.key),
        }

        normed = {}
        for k, v in metadata.items():
            if isinstance(v, dict):
                for nk, nv in v.items():
                    normed[str(f'{k}:{nk}')] = nv

        metadata.update(normed)

        if self.operation is not None:
            self.operation(sim)

        data = dict(
            raw_signal = sim.F_exp,
            raw_isosbestic = None if self.as_single_channel else sim.F_iso,
            time = sim.time,
            events = sim.event_layer.onsets_to_dict(),
            metadata = metadata,
        )

        return data

#endregion
