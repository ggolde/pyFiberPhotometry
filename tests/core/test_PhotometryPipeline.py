import inspect
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from PhoPro.core.PhotometryExperiment import PhotometryExperiment
from PhoPro.core.PhotometryData import PhotometryData
from PhoPro.core.PhotometryPipeline import MultiPipeline

# --- component tests ---
def test_pipeline_identifies_correct_input_data(dummy_pipeline):
    expected = ['dummy_file1.csv', 'dummy_file2.csv', 'dummy_file3.csv']
    discovered_paths = dummy_pipeline.discover_inputs()
    discovered_files = [str(path.name) for path in discovered_paths]

    assert all(f in expected for f in discovered_files)
    assert len(expected) == len(discovered_files)
    
def test_pipeline_constructs_jobs_correctly(dummy_pipeline):
    inputs = dummy_pipeline.discover_inputs()
    one_loader_jobs = dummy_pipeline._build_jobs(inputs, {}, {}, {})
    two_loader_jobs = dummy_pipeline._build_jobs(inputs, [{}, {}], {}, {})

    assert len(one_loader_jobs) == len(inputs)
    assert len(two_loader_jobs) == 2*len(inputs)

def test_pipeline_constructs_jobs_from_path_aware_loader_kwargs(dummy_pipeline):
    inputs = dummy_pipeline.discover_inputs()
    received_inputs = []

    def loader_kwargs(path):
        received_inputs.append(path)
        return {'annotation_file': path.with_suffix('.json')}

    jobs = dummy_pipeline._build_jobs(inputs, loader_kwargs, {}, {})

    assert received_inputs == inputs
    assert len(jobs) == len(inputs)
    assert all(
        job['loader_kwargs']['annotation_file'] == job['input'].with_suffix('.json')
        for job in jobs
    )

def test_pipeline_path_aware_loader_kwargs_can_create_multiple_jobs(dummy_pipeline):
    inputs = dummy_pipeline.discover_inputs()

    def loader_kwargs(path):
        return [
            {'annotation_file': path.with_suffix('.json')},
            {'annotation_file': path.with_name(f'{path.stem}_alternate.json')},
        ]

    jobs = dummy_pipeline._build_jobs(inputs, loader_kwargs, {}, {})

    assert len(jobs) == 2*len(inputs)
    assert all(isinstance(job['loader_kwargs'], dict) for job in jobs)

def test_pipeline_path_aware_loader_kwargs_validates_result_shape(dummy_pipeline):
    inputs = dummy_pipeline.discover_inputs()

    def loader_kwargs(path):
        return ['bad-loader-kwargs']

    with pytest.raises(TypeError, match='loader_kwargs.*must be a dict or a list of dicts'):
        dummy_pipeline._build_jobs(inputs, loader_kwargs, {}, {})

def test_pipeline_validates_resolved_loader_kwargs(dummy_pipeline):
    inputs = dummy_pipeline.discover_inputs()

    def loader_kwargs(path):
        return {'unexpected_kwarg': True}

    jobs = dummy_pipeline._build_jobs(inputs[:1], loader_kwargs, {}, {})

    with pytest.raises(TypeError, match='Unexpected kwargs'):
        dummy_pipeline._validate_loader_kwargs(jobs[0]['loader_kwargs'])

def test_pipeline_parameter_kwargs_include_function_defaults(dummy_pipeline):
    supplied = {'order': 8}

    effective = dummy_pipeline._kwargs_with_defaults(
        dummy_pipeline.experiment_cls.preprocess_signal,
        supplied,
    )

    signature = inspect.signature(dummy_pipeline.experiment_cls.preprocess_signal)
    assert effective['order'] == 8
    assert effective['cutoff_frequency'] == signature.parameters['cutoff_frequency'].default
    assert effective['maxiter'] == signature.parameters['maxiter'].default


def test_pipeline_concatenates_in_memory_results_once(
        dummy_pipeline,
        photometry_data,
        monkeypatch,
        ):
    dummy_pipeline._initialize_accumulation(None, low_memory_mode=False)

    combine_calls = []
    original_combine = PhotometryData.combine_obj

    def recording_combine(self, to_append, *args, **kwargs):
        combine_calls.append(to_append)
        return original_combine(self, to_append, *args, **kwargs)

    monkeypatch.setattr(PhotometryData, 'combine_obj', recording_combine)

    for job_number in range(3):
        result = photometry_data.copy()
        result.obs['job_number'] = job_number
        dummy_pipeline._accumulate_result(
            SimpleNamespace(trial_data=result),
            low_memory_mode=False,
        )

    combined = dummy_pipeline._finalize_result(
        trial_output_path=None,
        low_memory_mode=False,
    )

    assert len(combine_calls) == 1
    assert len(combine_calls[0]) == 2
    assert combined.obs['job_number'].tolist() == [
        0, 0, 0,
        1, 1, 1,
        2, 2, 2,
    ]
    assert dummy_pipeline.trial_data is combined


@pytest.mark.parametrize('n_results', [1, 3])
def test_pipeline_concatenates_low_memory_shards_once(
        dummy_pipeline,
        photometry_data,
        tmp_path,
        monkeypatch,
        n_results,
        ):
    output_path = tmp_path / 'trials.h5ad'
    dummy_pipeline._initialize_accumulation(output_path, low_memory_mode=True)

    concat_calls = []
    pipeline_module = importlib.import_module('PhoPro.core.PhotometryPipeline')
    original_concat = pipeline_module.concat_on_disk

    def recording_concat(*args, **kwargs):
        concat_calls.append(list(kwargs['in_files']))
        return original_concat(*args, **kwargs)

    monkeypatch.setattr(pipeline_module, 'concat_on_disk', recording_concat)

    for job_number in range(n_results):
        result = photometry_data.copy()
        result.obs['job_number'] = job_number
        dummy_pipeline._accumulate_result(
            SimpleNamespace(trial_data=result),
            low_memory_mode=True,
        )

    shard_dir = Path(dummy_pipeline._trial_shard_dir.name)
    combined = dummy_pipeline._finalize_result(
        trial_output_path=output_path,
        low_memory_mode=True,
    )

    assert len(concat_calls) == 1
    assert len(concat_calls[0]) == n_results
    assert combined.obs['job_number'].tolist() == [
        job_number
        for job_number in range(n_results)
        for _ in range(photometry_data.n_trials)
    ]
    assert output_path.exists()
    assert not shard_dir.exists()


def test_pipeline_rejects_misaligned_result_before_accumulation(
        dummy_pipeline,
        photometry_data,
        ):
    dummy_pipeline._initialize_accumulation(None, low_memory_mode=False)
    dummy_pipeline._accumulate_result(
        SimpleNamespace(trial_data=photometry_data.copy()),
        low_memory_mode=False,
    )

    misaligned = photometry_data.copy()
    misaligned.var['t'] = misaligned.ts + 0.1

    with pytest.raises(ValueError, match='Time-series misalignment'):
        dummy_pipeline._accumulate_result(
            SimpleNamespace(trial_data=misaligned),
            low_memory_mode=False,
        )

    assert len(dummy_pipeline._trial_results) == 1

# --- workflow tests ---
def test_pipeline_full_run(dummy_pipeline):

    # pipeline functions
    def id_builder(exp: PhotometryExperiment) -> str:
        source = exp.metadata.get('source', 'UNKNOWN/UNKNOWN.ext')
        uid = str(source).split('/')[-1].split('.')[0]
        return uid
    
    def post_load_operation(exp: PhotometryExperiment) -> None:
        exp.metadata['post_load_operation'] = 'success'

    def post_preprocess_operation(exp: PhotometryExperiment) -> None:
        exp.metadata['post_preprocess_operation'] = 'success'

    def post_extraction_operation(exp: PhotometryExperiment) -> None:
        exp.metadata['post_extraction_operation'] = 'success'
        exp.trial_data.obs['act_directly_on_trials'] = 'success'

    # vars
    inputs = dummy_pipeline.discover_inputs()

    passdown_metadata = [
        'source', 'key1', 'post_load_operation',
        'post_preprocess_operation', 'post_extraction_operation',
        ]

    data = dummy_pipeline.run(
        loader_kwargs={},
        preprocess_kwargs=dict(
            cutoff_frequency=3.0,
            order=4,
            correction_method="dF/F",
            signal_normalization="none",
            fit_using="IRLS",
            maxiter=200,
            c=3,
        ),
        trial_extraction_kwargs=dict(
            align_to="event",
            center_on=["choice_left", "choice_right"],
            trial_bounds=(-2.0, 4.0),
            baseline_bounds=(-2.0, 0.0),
            trial_normalization="none",
            check_overlap=True,
            event_conflict_logic="first",
        ),
        passdown_metadata=passdown_metadata,
        id_builder=id_builder,
        post_load_operation=post_load_operation,
        post_preprocess_operation=post_preprocess_operation,
        post_trial_extraction_operation=post_extraction_operation,
        params_file=None,
    )

    # validate
    missing_cols = [col for col in passdown_metadata if col not in data.obs]

    assert missing_cols == []
    assert (data.obs['key1'] == 'value1').all()
    assert (data.obs['post_load_operation'] == 'success').all()
    assert (data.obs['post_preprocess_operation'] == 'success').all()
    assert (data.obs['post_extraction_operation'] == 'success').all()
    assert (data.obs['act_directly_on_trials'] == 'success').all()
    assert (int(data.obs['experiment_id'].nunique()) == len(inputs))

def test_pipeline_saves_effective_parameters(dummy_pipeline, tmp_path):
    params_file = 'custom_params.json'
    dummy_pipeline.run(
        loader_kwargs={},
        preprocess_kwargs={'order': 8},
        trial_extraction_kwargs={
            'align_to': 'event',
            'center_on': ['choice_left'],
            'trial_bounds': (-2.0, 4.0),
        },
        output_dir=tmp_path,
        trial_output_file=None,
        params_file=params_file,
    )

    with (tmp_path / params_file).open() as f:
        params = json.load(f)

    assert params['preprocess_signal']['order'] == 8
    assert params['preprocess_signal']['cutoff_frequency'] == 3.0
    assert params['extract_trial_data']['align_to'] == 'event'
    assert 'trial_bounds' in params['extract_trial_data']


def test_pipeline_log_file_is_overwritten_on_rerun(dummy_pipeline, tmp_path):
    log_path = tmp_path / 'pipeline.log'
    kwargs = {
        'loader_kwargs': {},
        'preprocess_kwargs': {},
        'trial_extraction_kwargs': {
            'align_to': 'event',
            'center_on': ['choice_left'],
            'trial_bounds': (-2.0, 4.0),
        },
        'log_file': str(log_path),
        'params_file': None,
    }

    dummy_pipeline.run(**kwargs)
    with log_path.open('a') as f:
        f.write('SENTINEL FROM FIRST RUN\n')

    dummy_pipeline.run(**kwargs)

    assert 'SENTINEL FROM FIRST RUN' not in log_path.read_text()
    assert 'Beginning pipeline' in log_path.read_text()


def test_multi_pipeline_separates_master_and_sub_run_logs(tmp_path):
    class Result:
        n_trials = 2
        n_times = 3

    class RecordingPipeline:
        def run(self, *, logger, output_dir, **kwargs):
            logger.info('child-only detail for %s', Path(output_dir).name)
            return Result()

    pipeline = MultiPipeline(RecordingPipeline())
    pipeline.run(
        all_preprocess_kwargs={'first': {}, 'second': {}},
        loader_kwargs={},
        trial_extraction_kwargs={},
        output_dir=tmp_path,
        params_file=None,
    )

    master_text = (tmp_path / 'multi_pipeline.log').read_text()
    first_text = (tmp_path / 'first' / 'pipeline.log').read_text()
    second_text = (tmp_path / 'second' / 'pipeline.log').read_text()

    assert 'Finished running pipeline' in master_text
    assert 'child-only detail' not in master_text
    assert 'child-only detail for first' in first_text
    assert 'child-only detail for second' in second_text
