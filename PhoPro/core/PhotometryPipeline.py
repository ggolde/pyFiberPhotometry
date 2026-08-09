"""Batch processing pipelines for photometry experiments."""

from __future__ import annotations

import os
import inspect
import logging
import itertools
import json

from pathlib import Path
from typing import Any, Callable, Iterable, Literal

from .PhotometryExperiment import PhotometryExperiment
from .PhotometryData import PhotometryData
from .PhotometryLoader import PhotometryLoader

LoaderKwargs = dict[str, Any] | list[dict[str, Any]]
LoaderKwargsResolver = Callable[[Path], LoaderKwargs]
LoaderKwargsInput = LoaderKwargs | LoaderKwargsResolver

MultiPreprocessKwargsInput = dict[str, dict[str, Any]]


def _build_file_logger(name: str, path: str | Path) -> tuple[logging.Logger, logging.Handler]:
    """Create an isolated logger which overwrites its output file."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    handler = logging.FileHandler(path, mode='w')
    handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s'
    ))
    logger.addHandler(handler)
    return logger, handler


def _close_file_logger(logger: logging.Logger, handler: logging.Handler) -> None:
    """Detach and close a file handler owned by one pipeline run."""
    logger.removeHandler(handler)
    handler.close()


#############################
#region --- BASE PIPELINE ---
#############################
class PhotometryPipeline:
    """Run a loader, preprocessing, and trial-extraction workflow over inputs."""

    ###########################
    #region --- CONSTRUCTOR ---
    ###########################
    def __init__(
            self,
            data_directory: str | Path,
            target_type: Literal['file', 'folder'],
            loader_cls: type[PhotometryLoader],
            experiment_cls: type[PhotometryExperiment] = PhotometryExperiment,
            data_cls: type[PhotometryData] = PhotometryData,
            recursive: bool = False,
            pattern: str | None = None,
            ) -> None:
        """Initialize a generic directory-level photometry pipeline.

        Parameters
        ----------
        data_directory : str or Path
            Root directory containing candidate input files or folders.
        target_type : {'file', 'folder'}
            Path kind to treat as a pipeline input.
        loader_cls : type[PhotometryLoader]
            Loader class used for each discovered input.
        experiment_cls : type[PhotometryExperiment], default=PhotometryExperiment
            Experiment class constructed by the loader.
        data_cls : type[PhotometryData], default=PhotometryData
            Trial-data class used when reloading accumulated data from disk.
        recursive : bool, default=False
            Whether to recursively search under ``data_directory``.
        pattern : str or None, default=None
            Optional glob pattern used to filter discovered inputs.

        Raises
        ------
        ValueError
            If ``data_directory`` does not exist or is not a directory.
        """
        # validate inputs
        data_directory = Path(data_directory).expanduser()
        if not data_directory.exists():
            raise ValueError(f"Data directory {data_directory} does not exist.")
        if not data_directory.is_dir():
            raise ValueError(f"Data directory {data_directory} is not a directory.")

        # save attributes
        self.data_directory = data_directory
        self.target_type = target_type
        self.loader_cls = loader_cls
        self.experiment_cls = experiment_cls
        self.data_cls = data_cls
        self.recursive = recursive
        self.pattern = pattern

    #endregion

    ############################
    #region --- JOB BUILDING ---
    ############################
    def _matches_input_kind(self, path: Path) -> bool:
        """Check whether a path matches the configured input kind."""
        match self.target_type:
            case 'file':
                return path.is_file()
            case 'folder':
                return path.is_dir()
            case _:
                raise ValueError(f'Target type {self.target_type} not recognized.')

    def _build_jobs(
            self,
            inputs: list[Path],
            loader_kwargs: LoaderKwargsInput,
            preprocess_kwargs: dict[str, Any],
            trial_extraction_kwargs: dict[str, Any],
            ) -> list[dict[str, Any]]:
        """Build the per-input job dictionaries."""
        jobs = [
            {
                'input': fpath,
                'loader_kwargs': loader_args,
                'preprocess_kwargs': preprocess_kwargs,
                'trial_extraction_kwargs': trial_extraction_kwargs,
            }
            for fpath in inputs
            for loader_args in self._resolve_loader_kwargs(fpath, loader_kwargs)
        ]
        return jobs

    def _normalize_loader_kwargs(
            self,
            loader_kwargs: LoaderKwargs,
            input_path: Path | None = None,
            ) -> list[dict[str, Any]]:
        """Normalize loader kwargs to a list of per-job dictionaries."""
        if isinstance(loader_kwargs, dict):
            return [dict(loader_kwargs)]

        try:
            normalized = list(loader_kwargs)
        except TypeError as e:
            context = '' if input_path is None else f' for input {input_path}'
            raise TypeError(
                f'loader_kwargs{context} must be a dict or a list of dicts.'
            ) from e

        for i, kwargs in enumerate(normalized):
            if not isinstance(kwargs, dict):
                context = '' if input_path is None else f' for input {input_path}'
                raise TypeError(
                    f'loader_kwargs{context} must be a dict or a list of dicts; '
                    f'item {i} has type {type(kwargs).__name__}.'
                )

        return [dict(kwargs) for kwargs in normalized]

    def _resolve_loader_kwargs(
            self,
            input_path: Path,
            loader_kwargs: LoaderKwargsInput,
            ) -> list[dict[str, Any]]:
        """Resolve static or path-aware loader kwargs for one input path."""
        resolved_kwargs = loader_kwargs(input_path) if callable(loader_kwargs) else loader_kwargs
        return self._normalize_loader_kwargs(resolved_kwargs, input_path=input_path)

    #endregion

    ##################################
    #region --- VALIDATION HELPERS ---
    ##################################
    def _validate_kwargs(
            self,
            func: Callable,
            kwargs: dict[str, Any],
            supplied_positionally: Iterable[str] = (),
            ) -> None:
        """Validate keyword arguments against a callable signature."""
        sig = inspect.signature(func)
        params = sig.parameters
        skip = {'self', 'cls', *supplied_positionally}

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

    def _validate_loader_kwargs(
            self,
            loader_kwargs: dict[str, Any],
            ) -> None:
        """Validate one resolved loader kwargs dictionary."""
        loader_sig = inspect.signature(self.loader_cls.__init__)
        loader_positional = [
            name for name, param in loader_sig.parameters.items()
            if param.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
            and name != 'self'
        ]
        loader_path_arg = loader_positional[0] if len(loader_positional) > 0 else None

        self._validate_kwargs(
            self.loader_cls.__init__,
            loader_kwargs,
            supplied_positionally=() if loader_path_arg is None else (loader_path_arg,),
        )

    def _validate_static_kwargs(
            self,
            preprocess_kwargs: dict[str, Any],
            trial_extraction_kwargs: dict[str, Any],
            dashboard_kwargs: dict[str, Any],
            ) -> None:
        """Validate pipeline kwargs that are independent of discovered inputs."""
        self._validate_kwargs(self.experiment_cls.preprocess_signal, preprocess_kwargs)
        self._validate_kwargs(self.experiment_cls.extract_trial_data, trial_extraction_kwargs)
        self._validate_kwargs(self.experiment_cls.plot_dashboard, dashboard_kwargs)

    def _validate_all_kwargs(
            self,
            loader_kwargs: LoaderKwargsInput,
            preprocess_kwargs: dict[str, Any],
            trial_extraction_kwargs: dict[str, Any],
            dashboard_kwargs: dict[str, Any],
            ) -> None:
        """Validate all user-supplied pipeline keyword dictionaries."""
        if not callable(loader_kwargs):
            for loader_kwargs_i in self._normalize_loader_kwargs(loader_kwargs):
                self._validate_loader_kwargs(loader_kwargs_i)

        self._validate_static_kwargs(preprocess_kwargs, trial_extraction_kwargs, dashboard_kwargs)

    def _kwargs_with_defaults(
            self,
            func: Callable,
            kwargs: dict[str, Any],
            ) -> dict[str, Any]:
        """Overlay supplied keyword arguments on a callable's defaults."""
        defaults = {
            name: param.default
            for name, param in inspect.signature(func).parameters.items()
            if name not in {'self', 'cls'}
            and param.default is not inspect.Parameter.empty
        }
        return defaults | kwargs

    def _validate_output_dir(self, output_dir: Path | None) -> None:
        """Validate or create the output directory."""
        if output_dir is None:
            return
        path = Path(output_dir)
        if not path.exists():
            if not path.parent.exists():
                raise ValueError(f"Parent directory does not exist: {path.parent}")
            path.mkdir(exist_ok=True)

    def _validate_save_dashboard(self, save_dashboards: bool, output_dir: Path | None) -> Path | None:
        """Validate dashboard saving and create its output directory."""
        if (output_dir is None) and save_dashboards:
            raise ValueError('output_dir must be specified when save_dashboards is True')
        elif (output_dir is not None) and save_dashboards:
            dashboard_dir = output_dir / 'dashboards'
            dashboard_dir.mkdir(exist_ok=True)
            return dashboard_dir

    def _validate_low_memory_mode(
            self,
            low_memory_mode: bool,
            trial_output_path: str | Path | None,
            ) -> None:
        """Validate output requirements for low-memory accumulation."""
        if low_memory_mode:
            if trial_output_path is None:
                raise ValueError(f'Using low memory mode without an output file.')
            elif os.path.exists(trial_output_path):
                raise ValueError(f'Using low memory mode with preexsisting output file: {trial_output_path}')

    #endregion

    ################################
    #region --- PIPELINE HELPERS ---
    ################################
    def _load_experiment(self, path: Path, loader_kwargs: dict[str, Any]) -> type[PhotometryExperiment]:
        """Load one experiment from a discovered input path."""
        loader = self.loader_cls(path, **loader_kwargs)
        exp: type[PhotometryExperiment] = loader.load(exp_cls=self.experiment_cls)
        return exp

    def _run_preprocessing(self, exp: type[PhotometryExperiment], preprocess_kwargs: dict[str, Any]) -> None:
        """Run preprocessing on one experiment."""
        exp.preprocess_signal(**preprocess_kwargs)

    def _run_trial_extraction(self, exp: type[PhotometryExperiment], trial_extraction_kwargs: dict[str, Any]) -> None:
        """Run trial extraction on one experiment."""
        exp.extract_trial_data(**trial_extraction_kwargs)

    def _accumulate_result(
            self,
            exp: type[PhotometryExperiment],
            trial_output_path: Path | None = None,
            low_memory_mode: bool = False,
            ) -> None:
        """Accumulate one experiment's trial data in memory or on disk."""
        # four cases, low memory mode * first path
        if low_memory_mode:
            if os.path.exists(trial_output_path):
                exp.trial_data.append_on_disk_h5ad(trial_output_path)
            else:
                exp.trial_data.write_h5ad(trial_output_path)

        else:
            if self.trial_data is None:
                self.trial_data = exp.trial_data.copy()
            else:
                self.trial_data.combine_obj(exp.trial_data, inplace=True, check_time=True, time_tol=1e-6)

    def _finalize_result(
            self,
            trial_output_path: Path | None = None,
            low_memory_mode: bool = False,
            ) -> type[PhotometryData]:
        """Finalize accumulated trial data after all jobs finish."""
        if low_memory_mode:
            trial_data: type[PhotometryData] = self.data_cls.read_h5ad(trial_output_path)

        else:
            if self.trial_data is None:
                raise ValueError('Trial data is missing. Perhaps all jobs errored.')

            trial_data = self.trial_data
            trial_data.obs.reset_index(drop=True, inplace=True)
            trial_data.obs.index = trial_data.obs.index.astype(str)

        if trial_output_path is not None:
            trial_data.write_h5ad(trial_output_path)

        return trial_data

    # --- public jobs API ---
    def discover_inputs(self) -> list[Path]:
        """Discover candidate inputs under ``self.data_directory``.

        Returns
        -------
        list[Path]
            Paths matching ``target_type`` and the optional glob pattern.
        """
        if self.pattern is None:
            candidates = (
                self.data_directory.rglob('*')
                if self.recursive
                else self.data_directory.iterdir()
            )
        else:
            candidates = (
                self.data_directory.rglob(self.pattern)
                if self.recursive
                else self.data_directory.glob(self.pattern)
            )

        inputs = [path for path in candidates if self._matches_input_kind(path)]
        return inputs

    #endregion

    ############################
    #region --- PIPELINE API ---
    ############################
    def run(
            self,

            # args
            loader_kwargs: LoaderKwargsInput,
            preprocess_kwargs: dict[str, Any],
            trial_extraction_kwargs: dict[str, Any],

            # I/O
            output_dir: str | Path | None = None,
            log_file: str | None = None,
            trial_output_file: str | None = 'trials.h5ad',
            params_file: str | None = 'pipeline_params.json',

            # pipeline params
            low_memory_mode: bool = False,

            # inheritance
            passdown_metadata: list[str] | None = ['source'],

            # dashboards
            save_dashboards: bool = False,
            dashboard_kwargs: dict[str, Any] = {},

            # custom operations
            id_builder: Callable[[type[PhotometryExperiment]], str] | None = None,
            post_load_operation: Callable[[type[PhotometryExperiment]], None] | None = None,
            post_preprocess_operation: Callable[[type[PhotometryExperiment]], None] | None = None,
            post_trial_extraction_operation: Callable[[type[PhotometryExperiment]], None] | None = None,
            logger: logging.Logger | logging.LoggerAdapter | None = None,
            ) -> type[PhotometryData]:
        """Run the batch processing pipeline over all discovered inputs.

        For each discovered input, the pipeline constructs a loader, loads an
        experiment, preprocesses the continuous signal, extracts trial-wise
        data, optionally applies custom hook operations, passes selected
        metadata down into ``trial_data.obs``, and accumulates the resulting
        trial-data objects in memory or on disk.

        Parameters
        ----------
        loader_kwargs : dict[str, Any], list[dict[str, Any]], or Callable[[Path], ...]
            Keyword arguments passed to ``loader_cls`` for each job, excluding
            the discovered input path supplied positionally. A list creates
            multiple jobs per discovered input. A callable receives each
            discovered input path and must return a dict or list of dicts.
        preprocess_kwargs : dict[str, Any]
            Keyword arguments passed to
            ``PhotometryExperiment.preprocess_signal``.
        trial_extraction_kwargs : dict[str, Any]
            Keyword arguments passed to
            ``PhotometryExperiment.extract_trial_data``.
        output_dir : str, Path, or None, default=None
            Directory where outputs should be written. If ``None``, no output
            files are written.
        log_file : str or None, default=None
            Optional log-file path.
        trial_output_file : str or None, default='trials.h5ad'
            Name of the accumulated trial-data file written under
            ``output_dir``.
        params_file : str or None, default='pipeline_params.json'
            JSON file used to save the effective preprocessing and trial
            extraction parameters. Relative paths are written under
            ``output_dir`` when one is provided. If ``None``, parameters are
            not saved.
        low_memory_mode : bool, default=False
            If ``True``, accumulate trial data directly on disk.
        passdown_metadata : list[str] or None, default=['source']
            Metadata keys from ``exp.metadata`` copied into
            ``exp.trial_data.obs`` for each processed experiment.
        save_dashboards : bool, default=False
            If ``True``, save one dashboard per experiment in
            ``output_dir / 'dashboards'``.
        dashboard_kwargs : dict[str, Any], default={}
            Keyword arguments passed to ``PhotometryExperiment.plot_dashboard``.
        id_builder : Callable[[type[PhotometryExperiment]], str] or None, default=None
            Optional callable used to assign an experiment ID after loading.
        post_load_operation : Callable[[type[PhotometryExperiment]], None] or None, default=None
            Optional callable run immediately after loading.
        post_preprocess_operation : Callable[[type[PhotometryExperiment]], None] or None, default=None
            Optional callable run immediately after preprocessing.
        post_trial_extraction_operation : Callable[[type[PhotometryExperiment]], None] or None, default=None
            Optional callable run immediately after trial extraction.
        logger : logging.Logger, logging.LoggerAdapter, or None, default=None
            Logger used for this run. When omitted and ``log_file`` is supplied,
            an isolated file logger is created and the file is overwritten.

        Returns
        -------
        PhotometryData
            Trial-data object containing all extracted trials.

        Raises
        ------
        TypeError
            If provided keyword arguments do not match the target method
            signatures.
        ValueError
            If output configuration is invalid or no inputs are discovered.
        """

        # --- set up logger ---
        owned_logger_handler = None
        if logger is None and log_file is not None:
            logger, owned_logger_handler = _build_file_logger(
                f'{__name__}.pipeline.{id(self)}.{id(log_file)}',
                log_file,
            )
        elif logger is None:
            logger = logging.getLogger(__name__)
        logger.info('Beginning pipeline')

        # --- coerce inputs ---
        if output_dir is not None:
            output_dir = Path(output_dir)
            trial_output_path = (
                None
                if trial_output_file is None
                else os.path.join(output_dir, trial_output_file)
            )
        else:
            trial_output_path = None

        # --- validate static inputs ---
        logger.info('Validating static inputs...')
        self._validate_output_dir(output_dir)
        self._validate_low_memory_mode(low_memory_mode, trial_output_path)
        self._validate_static_kwargs(preprocess_kwargs, trial_extraction_kwargs, dashboard_kwargs)
        dashboard_dir = self._validate_save_dashboard(save_dashboards, output_dir)

        # --- save effective processing parameters ---
        if params_file is not None:
            params_path = Path(params_file)
            if output_dir is not None and not params_path.is_absolute():
                params_path = output_dir / params_path
            params = {
                'preprocess_signal': self._kwargs_with_defaults(
                    self.experiment_cls.preprocess_signal,
                    preprocess_kwargs,
                ),
                'extract_trial_data': self._kwargs_with_defaults(
                    self.experiment_cls.extract_trial_data,
                    trial_extraction_kwargs,
                ),
            }
            with params_path.open('w') as f:
                json.dump(params, f, indent=4)

        # --- construct jobs ---
        logger.info('Discovering inputs...')
        inputs = self.discover_inputs()
        if len(inputs) == 0:
            raise ValueError(f'No inputs discovered!')

        logger.info('Building jobs...')
        jobs = self._build_jobs(
            inputs=inputs,
            loader_kwargs=loader_kwargs,
            preprocess_kwargs=preprocess_kwargs,
            trial_extraction_kwargs=trial_extraction_kwargs
        )

        # --- validate resolved kwargs ---
        logger.info('Validating loader_kwargs per-job...')
        for job in jobs:
            self._validate_loader_kwargs(job['loader_kwargs'])

        # --- set up job iteration ---
        n_jobs = len(jobs)
        self.trial_data: type[PhotometryData] = None
        n_errors = 0
        n_processed = 0

        # --- iterate over jobs ---
        logger.info(f'Iterating over {n_jobs} jobs...\n')
        for i, job in enumerate(jobs):
            try:
                logger.info(f'Processing job {job["input"]} ({i+1}/{n_jobs})')

                # 1. load experiment
                logger.info(f'Loading expriment...')
                exp = self._load_experiment(job['input'], job['loader_kwargs'])

                if post_load_operation is not None:
                    logger.info(f'Running custom post-loading operation...')
                    post_load_operation(exp)

                # 2. run preprocess
                logger.info(f'Preprocessing signal...')
                self._run_preprocessing(exp, job['preprocess_kwargs'])

                if post_preprocess_operation is not None:
                    logger.info(f'Running custom post-preprocessing operation...')
                    post_preprocess_operation(exp)

                # 3. run extract trial data
                logger.info(f'Extracting trial data...')
                self._run_trial_extraction(exp, job['trial_extraction_kwargs'])

                if post_trial_extraction_operation is not None:
                    logger.info(f'Running custom post-trial extraction operation...')
                    post_trial_extraction_operation(exp)

                # warn if any trial windows were invalid
                if exp.metadata['invalid_windows'] is not None:
                    logger.warning(
                        f'Invalid trial windows at indexes {exp.metadata["invalid_windows"]} '
                        f'have been dropped.'
                    )

                # assign uid if function provided
                if id_builder is not None:
                    logger.info(f'Assiging experiment ID...')
                    exp.id = id_builder(exp)
                    exp.trial_data.obs['experiment_id'] = exp.id
                    logger.info(f'Expriment ID = {str(exp.id)}')

                # pass down specified metadata
                if passdown_metadata is not None:
                    logger.info(f'Passing down experiment metadata as columns...')
                    exp.trial_data.add_obs_columns(add_from=exp.metadata, keys=passdown_metadata)

                # save dashboards (optional)
                if save_dashboards and (dashboard_dir is not None):
                    dashboard_file = f'experiment_{str(i+1)}.svg' if exp.id is None else f'{str(exp.id)}.svg'
                    dashboard_fpath = str(dashboard_dir / dashboard_file)
                    logger.info(f'Saving dashboard to {dashboard_file}...')
                    exp.plot_dashboard(save=dashboard_fpath, **dashboard_kwargs)

                # accumulate object
                logger.info(f'Accumulating result...')
                self._accumulate_result(exp, trial_output_path, low_memory_mode)

                # log expriment info
                logger.info(
                    f'Finished processing experiment with '
                    f'{exp.trial_data.n_trials} trials x {exp.trial_data.n_times} timepoints.'
                )

                # clean up before next iteration
                logger.info(f'Job {i+1} complete.\n')
                del exp
                n_processed += 1

            # handle errors
            except Exception as e:
                logger.error(f'Error processing {job["input"]}: \n\t {e}\n', exc_info=True)
                n_errors += 1

        # --- finalize ---
        logger.info('Finalizng results...')
        trial_data = self._finalize_result(trial_output_path, low_memory_mode)

        logger.info(
            f'Processing pipeline complete with {n_errors} errors '
            f'and {n_processed} successes out of {n_jobs} jobs.'
        )

        if owned_logger_handler is not None:
            _close_file_logger(logger, owned_logger_handler)

        return trial_data

    #endregion

##############################
#region --- MULTI-PIPELINE ---
##############################
class MultiPipeline:
    """Run a ``PhotometryPipeline`` across multiple preprocessing settings.

    Parameters
    ----------
    pipeline : PhotometryPipeline
        Pipeline instance used for every preprocessing run.
    """

    def __init__(
            self,
            pipeline: PhotometryPipeline,
            ) -> None:
        """Initialize the multi-run preprocessing wrapper."""
        self.pipeline = pipeline

    def build_preprocess_kwarg_permutations(
            self,
            default_kwargs: dict[str, Any],
            to_permute_kwargs: dict[str, list[Any]],
            name_from: list[str],
            ) -> dict[str, dict[str, Any]]:
        """Build named preprocessing kwargs from crossed parameter values.

        Parameters
        ----------
        default_kwargs : dict[str, Any]
            Baseline keyword arguments for
            ``PhotometryExperiment.preprocess_signal()``.
        to_permute_kwargs : dict[str, list[Any]]
            Keyword arguments whose values are fully crossed to form
            preprocessing configurations.
        name_from : list[str]
            Keys used to build each output name from the generated
            configuration.

        Returns
        -------
        dict[str, dict[str, Any]]
            Mapping from generated run names to preprocessing keyword
            dictionaries.
        """
        self.pipeline._validate_kwargs(PhotometryExperiment.preprocess_signal, default_kwargs)
        self.pipeline._validate_kwargs(PhotometryExperiment.preprocess_signal, to_permute_kwargs)

        def _permutation_generator(to_permute: dict):
            """Yield dictionaries for each crossed preprocessing permutation."""
            for inp in itertools.product(*to_permute.values()):
                yield dict(zip(to_permute.keys(), inp))

        permutations = [p for p in _permutation_generator(to_permute_kwargs)]

        names = [
            '_'.join([ f"{key}-{str(p[key])}" for key in name_from ]).replace('/', '-').replace('\\', '-') + '_perm' + str(i)
            for i, p in enumerate(permutations)
        ]

        return dict(zip(names, permutations))

    def build_preprocess_kwarg_across(
            self,
            default_kwargs: dict[str, Any],
            across_kwargs: dict[str, list[Any]],
            name_from: list[str],
            ) -> dict[str, dict[str, Any]]:
        """Build named preprocessing kwargs by varying one key at a time.

        Parameters
        ----------
        default_kwargs : dict[str, Any]
            Baseline keyword arguments for
            ``PhotometryExperiment.preprocess_signal()``.
        across_kwargs : dict[str, list[Any]]
            Keyword arguments and candidate values to vary independently.
            Unlike ``build_preprocess_kwarg_permutations()``, values are not
            crossed across keys.
        name_from : list[str]
            Keys used to build each output name from the generated
            configuration.

        Returns
        -------
        dict[str, dict[str, Any]]
            Mapping from generated run names to preprocessing keyword
            dictionaries.
        """
        self.pipeline._validate_kwargs(PhotometryExperiment.preprocess_signal, default_kwargs)
        self.pipeline._validate_kwargs(PhotometryExperiment.preprocess_signal, across_kwargs)

        combos = {}
        i = 0
        for across_key, across_vals in across_kwargs.items():
            for val in across_vals:
                args = default_kwargs | {across_key : val}
                name = '_'.join([ f"{key}-{str(args[key])}" for key in name_from ]).replace('/', '-').replace('\\', '-') + '_combo' + str(i)
                combos[name] = args
                i = i + 1

        return combos

    def run(
            self,
            # args
            all_preprocess_kwargs: MultiPreprocessKwargsInput,
            loader_kwargs: LoaderKwargsInput,
            trial_extraction_kwargs: dict[str, Any],

            # I/O
            output_dir: str | Path,
            master_log_file: str = 'multi_pipeline.log',
            params_file: str | None = 'pipeline_params.json',
            sub_log_file: str = 'pipeline.log',
            trial_output_file: str = 'trials.h5ad',

            # pipeline params
            skip_existing: bool = False,
            low_memory_mode: bool = False,

            # inheritance
            passdown_metadata: list[str] | None = ['source'],

            # dashboards
            save_dashboards: bool = False,
            dashboard_kwargs: dict[str, Any] = {},

            # custom operations
            id_builder: Callable[[type[PhotometryExperiment]], str] | None = None,
            post_load_operation: Callable[[type[PhotometryExperiment]], None] | None = None,
            post_preprocess_operation: Callable[[type[PhotometryExperiment]], None] | None = None,
            post_trial_extraction_operation: Callable[[type[PhotometryExperiment]], None] | None = None,
            ) -> None:
        """Run the wrapped pipeline once for each preprocessing configuration.

        Parameters
        ----------
        all_preprocess_kwargs : dict[str, dict[str, Any]]
            Mapping from sub-run names to preprocessing keyword dictionaries.
            Each sub-run name becomes a child directory of ``output_dir``.
        loader_kwargs : LoaderKwargsInput
            Loader keyword arguments, or a resolver returning them, passed to
            ``PhotometryPipeline.run()``.
        trial_extraction_kwargs : dict[str, Any]
            Keyword arguments passed to trial extraction.
        output_dir : str | Path
            Parent directory for all preprocessing sub-runs.
        master_log_file : str, optional
            File name for the meta-log showing the success of each individual
            pipeline.
        params_file : str or None, optional
            File name used to save each sub-run's effective preprocessing and
            trial-extraction parameters, by default ``"pipeline_params.json"``.
            If ``None``, parameters are not saved.
        sub_log_file : str, optional
            File name used for each sub-run pipeline log, by default
            ``"pipeline.log"``.
        trial_output_file : str, optional
            File name used for each sub-run trial output, by default
            ``"trials.h5ad"``.
        skip_existing : bool, optional
            Whether to skip running the pipeline for parameter sets that already exsist
        low_memory_mode : bool, optional
            Whether to run the wrapped pipeline in low-memory mode, by default
            False.
        passdown_metadata : list[str] | None, optional
            Metadata fields passed down during trial extraction, by default
            ``["source"]``.
        save_dashboards : bool, optional
            Whether to save pipeline dashboards for each sub-run, by default
            False.
        dashboard_kwargs : dict[str, Any], optional
            Keyword arguments passed to dashboard generation, by default ``{}``.
        id_builder : Callable[[type[PhotometryExperiment]], str] | None, optional
            Optional function passed to the wrapped pipeline for building
            experiment identifiers, by default None.
        post_load_operation : Callable[[type[PhotometryExperiment]], None] | None, optional
            Optional callback run after loading in each pipeline run, by default
            None.
        post_preprocess_operation : Callable[[type[PhotometryExperiment]], None] | None, optional
            Optional callback run after preprocessing in each pipeline run, by
            default None.
        post_trial_extraction_operation : Callable[[type[PhotometryExperiment]], None] | None, optional
            Optional callback run after trial extraction in each pipeline run,
            by default None.
        """
        # create parent output dir
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir(exist_ok=True)

        # --- set up logger ---
        master_log_path = output_dir / master_log_file
        master_logger, master_handler = _build_file_logger(
            f'{__name__}.multi.{id(self)}',
            master_log_path,
        )
        master_logger.info(
            f'Beginning multi-pipeline accross {len(all_preprocess_kwargs)} preprocessing parameter sets...\n'
        )

        # loop through preprocess params
        n_errors = 0
        for i, (sub_run_name, preprocess_kwargs) in enumerate(all_preprocess_kwargs.items()):
            master_logger.info(
                f'Running pipeline {sub_run_name} ({i+1}/{len(all_preprocess_kwargs)})...'
            )

            # catch errors for individual pipelines
            try:
                # set up subfolder names
                master_logger.info(f'Setting up subfolders...')
                sub_output_dir = output_dir / sub_run_name
                if not sub_output_dir.exists():
                    sub_output_dir.mkdir(exist_ok=True)

                # skip if processed file already exsists
                trial_output_fpath = sub_output_dir / trial_output_file
                if skip_existing and trial_output_fpath.exists():
                    master_logger.info(
                        f'Processed file {trial_output_fpath} already exsits! Skipping parameter set.'
                        )
                    continue
                
                # set up child log
                child_log_path = sub_output_dir / sub_log_file
                child_logger, child_handler = _build_file_logger(
                    f'{__name__}.multi.{id(self)}.{i}',
                    child_log_path,
                )

                # run pipeline
                master_logger.info(f'Executing pipeline...')
                try:
                    trials: PhotometryData = self.pipeline.run(
                        loader_kwargs=loader_kwargs,
                        preprocess_kwargs=preprocess_kwargs,
                        trial_extraction_kwargs=trial_extraction_kwargs,
                        output_dir=sub_output_dir,
                        log_file=None,
                        trial_output_file=trial_output_file,
                        params_file=params_file,
                        passdown_metadata=passdown_metadata,
                        low_memory_mode=low_memory_mode,
                        id_builder=id_builder,
                        post_load_operation=post_load_operation,
                        post_preprocess_operation=post_preprocess_operation,
                        post_trial_extraction_operation=post_trial_extraction_operation,
                        save_dashboards=save_dashboards,
                        dashboard_kwargs=dashboard_kwargs,
                        logger=child_logger,
                    )
                finally:
                    _close_file_logger(child_logger, child_handler)
                master_logger.info(
                    f'Finished running pipeline, resulting in PhotometryData of: \n'
                    f'\t{trials.n_trials} trials x {trials.n_times} timepoints.\n'
                )

            # handle errors
            except Exception as e:
                master_logger.exception(f'Error running pipeline {sub_run_name}: \n\t {e}\n')
                n_errors += 1

        # end multi pipeline
        master_logger.info(
            f'Multi-pipeline run complete with {n_errors} errors out of '
            f'{len(all_preprocess_kwargs)} pipelines.'
        )
        _close_file_logger(master_logger, master_handler)
#endregion
