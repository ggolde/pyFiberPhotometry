import numpy as np
import pytest
import pathlib
import h5py

from PhoPro.core.PhotometryLoader import CSVLoader, EventDecoder, H5Loader


TEST_DATA = pathlib.Path(__file__).parent.parent / 'test_data'
DATA_DIR = TEST_DATA / 'data'
EVENT_DIR = TEST_DATA / 'events'
EXPECTED_EVENTS = {
    'event1': np.asarray([4.0]),
    'event2': np.asarray([1.0]),
    'event3': np.asarray([3.0, 5.0]),
}

# --- correct loading ---

def test_csv_loader_proper_loading():
    target_file = DATA_DIR / 'dummy_experiment.csv'
    event_cols = ['event1', 'event2', 'event3']

    loader = CSVLoader(
        target_file,
        time_col='time',
        signal_col='experimental',
        isosbestic_col='isosbestic',
        event_cols=event_cols,
        downsample=None,
    )
    exp = loader.load()

    assert np.array_equal(exp.raw_signal, [10.4, 3.2, 2.4, 5.7, 6.5, 5.3])
    assert np.array_equal(exp.raw_isosbestic, [3.1, 2.8, 2.6, 2.5, 2.4, 2.3])
    assert set(exp.events) == set(EXPECTED_EVENTS)
    for label, timestamps in EXPECTED_EVENTS.items():
        assert np.array_equal(exp.events[label], timestamps)
    assert exp.frequency == 1.0

def test_annotation_handler_json():
    target_file = DATA_DIR / 'dummy_experiment.csv'
    target_annotation = DATA_DIR / 'dummy_annotation.json'
    event_cols = ['event1', 'event2', 'event3']

    loader = CSVLoader(
        target_file,
        time_col='time',
        signal_col='experimental',
        isosbestic_col='isosbestic',
        event_cols=event_cols,
        downsample=None,
        annotation_file=target_annotation,
        annotation_handler='json'
    )
    exp = loader.load()

    assert 'key1' in exp.metadata
    assert 'key2' in exp.metadata
    assert exp.metadata['key1'] == 'value1'
    assert exp.metadata['key2'] == 50

def test_annotation_handler_yaml():
    target_file = DATA_DIR / 'dummy_experiment.csv'
    target_annotation = DATA_DIR / 'dummy_annotation.yml'
    event_cols = ['event1', 'event2', 'event3']

    loader = CSVLoader(
        target_file,
        time_col='time',
        signal_col='experimental',
        isosbestic_col='isosbestic',
        event_cols=event_cols,
        downsample=None,
        annotation_file=target_annotation,
        annotation_handler='yaml'
    )
    exp = loader.load()

    assert 'key1' in exp.metadata
    assert 'key2' in exp.metadata
    assert exp.metadata['key1'] == 'value1'
    assert exp.metadata['key2'] == 50


def test_event_decoder_supports_all_builtin_encodings():
    time = np.asarray([0.0, 1.0, 2.0, 3.0])

    binary = EventDecoder.decode({'cue': [0, 1, np.nan, 1]}, 'binary', time=time)
    timestamps = EventDecoder.decode({'cue': [1.0, np.nan, 3.0]}, 'timestamp')
    long = EventDecoder.decode(
        {'event_label': ['cue', 'reward', 'cue', None]},
        'long',
        time=time,
    )

    assert np.array_equal(binary['cue'], [1.0, 3.0])
    assert np.array_equal(timestamps['cue'], [1.0, 3.0])
    assert np.array_equal(long['cue'], [0.0, 2.0])
    assert np.array_equal(long['reward'], [1.0])


def test_event_decoder_uses_storage_independent_custom_callable():
    def decode_second_sample(events, time):
        return {label: time[np.asarray(values, dtype=bool)] + 0.5 for label, values in events.items()}

    events = EventDecoder.decode(
        {'cue': [False, True]},
        decode_second_sample,
        time=[0.0, 1.0],
    )

    assert np.array_equal(events['cue'], [1.5])


@pytest.mark.parametrize(
    ('event_file', 'encoding', 'event_cols'),
    [
        ('event_binary.csv', 'binary', ['event1', 'event2', 'event3']),
        ('event_timestamp.csv', 'timestamp', ['event1', 'event2', 'event3']),
        ('event_long.csv', 'long', 'event'),
    ],
)
def test_csv_loader_decodes_separate_event_files(event_file, encoding, event_cols):
    loader = CSVLoader(
        csv=DATA_DIR / 'dummy_experiment.csv',
        signal_col='experimental',
        event_file=EVENT_DIR / event_file,
        event_cols=event_cols,
        event_encoding=encoding,
    )
    data = loader.extract_data()

    assert set(data['events']) == set(EXPECTED_EVENTS)
    for label, timestamps in EXPECTED_EVENTS.items():
        assert np.array_equal(data['events'][label], timestamps)


def test_csv_loader_defaults_separate_event_file_to_timestamp_encoding():
    loader = CSVLoader(
        csv=DATA_DIR / 'dummy_experiment.csv',
        signal_col='experimental',
        event_file=EVENT_DIR / 'event_timestamp.csv',
        event_cols=['event1', 'event2', 'event3'],
    )

    events = loader.extract_data()['events']

    for label, timestamps in EXPECTED_EVENTS.items():
        assert np.array_equal(events[label], timestamps)


def test_csv_loader_decodes_separate_event_file_with_custom_callable():
    def decode_nonzero(events, time):
        assert time is not None
        return {
            label: time[np.asarray(values, dtype=float) > 0]
            for label, values in events.items()
        }

    loader = CSVLoader(
        csv=DATA_DIR / 'dummy_experiment.csv',
        signal_col='experimental',
        event_file=EVENT_DIR / 'event_binary.csv',
        event_cols=['event1', 'event2', 'event3'],
        event_encoding=decode_nonzero,
    )

    events = loader.extract_data()['events']

    for label, timestamps in EXPECTED_EVENTS.items():
        assert np.array_equal(events[label], timestamps)


@pytest.mark.parametrize(
    ('encoding', 'event_values', 'expected'),
    [
        ('binary', [0, 1, 0, 1], {'cue': np.asarray([1.0, 3.0])}),
        ('timestamp', [0.5, 2.5, np.nan], {'cue': np.asarray([0.5, 2.5])}),
        (
            'long',
            np.asarray(['cue', 'reward', 'cue', 'reward'], dtype='S'),
            {'cue': np.asarray([0.0, 2.0]), 'reward': np.asarray([1.0, 3.0])},
        ),
    ],
)
def test_h5_loader_uses_shared_event_decoder(tmp_path, encoding, event_values, expected):
    file = tmp_path / 'experiment.h5'
    with h5py.File(file, 'w') as h5_file:
        h5_file.create_dataset('time', data=[0.0, 1.0, 2.0, 3.0])
        h5_file.create_dataset('signal', data=[10.0, 11.0, 12.0, 13.0])
        h5_file.create_dataset('events', data=event_values)

    loader = H5Loader(
        file=file,
        time_path='time',
        signal_path='signal',
        event_paths={'event_label' if encoding == 'long' else 'cue': 'events'},
        event_encoding=encoding,
    )
    data = loader.extract_data()

    assert np.array_equal(data['raw_signal'], [10.0, 11.0, 12.0, 13.0])
    assert data['raw_isosbestic'] is None
    assert set(data['events']) == set(expected)
    for label, timestamps in expected.items():
        assert np.array_equal(data['events'][label], timestamps)


def test_h5_loader_reports_missing_event_path(tmp_path):
    file = tmp_path / 'experiment.h5'
    with h5py.File(file, 'w') as h5_file:
        h5_file.create_dataset('time', data=[0.0, 1.0])
        h5_file.create_dataset('signal', data=[10.0, 11.0])

    loader = H5Loader(
        file=file,
        time_path='time',
        signal_path='signal',
        event_paths={'cue': 'missing'},
        event_encoding='timestamp',
    )

    with pytest.raises(KeyError, match='missing'):
        loader.extract_data()


def test_h5_loader_reads_doric_fixture():
    loader = H5Loader(
        file=DATA_DIR / 'test_doric.doric',
        time_path='Traces/Console/Time(s)/Console_time(s)',
        signal_path='Traces/Console/AIn-1 - Raw/AIn-1 - Raw',
        isosbestic_path='Traces/Console/AIn-2 - Raw/AIn-2 - Raw',
        event_paths={'digital_input': 'Traces/Console/DI--O-1/DI--O-1'},
        event_encoding='binary',
        downsample=100,
        downsample_kwargs={'method': 'mean'},
    )

    data = loader.extract_data()

    assert data['raw_signal'].shape == data['raw_isosbestic'].shape
    assert data['raw_signal'].shape == data['time'].shape
    assert data['raw_signal'].size == 13_869
    assert data['events']['digital_input'].size == 107_474
    assert np.isclose(data['events']['digital_input'][0], 39.186126)
    assert np.isclose(data['events']['digital_input'][-1], 115.129798)
