import numpy as np
import pandas as pd
import pathlib
import pytest

from PhoPro.core.PhotometryExperiment import PhotometryExperiment
from PhoPro.core.PhotometryData import PhotometryData
from PhoPro.core.PhotometryLoader import PhotometryLoader
from PhoPro.core.PhotometryPipeline import PhotometryPipeline
from PhoPro.sim.SimulatedPhotometry import SimulatedPhotometry

# simulated data
def simulate_data() -> SimulatedPhotometry:
    sim = SimulatedPhotometry.from_parameters(
        length_sec=120,
        frequency=20,
        event_label='event',
        event_amplitude=0.05,
        n_events=12,
        iso_bleach_scale=1.0,
        gaussian_noise_scale_exp=0.1,
        mult_noise_magnitude_exp=0.1,
        mult_noise_exponent_exp=1,
        seed=7,
    )
    sim.add_event_relative_to(
        relative_to='event',
        time_range=(0.2, 0.8),
        overall_prob=1.0,
        labels=['choice_left', 'choice_right'],
        amplitudes=[0.1, 0.1],
        choice_probs=[0.5, 0.5],
    )
    return sim

# set up dummy experiment tester subclass
class DummyExperiment(PhotometryExperiment):
    def run_pipeline(self) -> None:
        self.preprocess_signal(
            detrend=False,
            cutoff_frequency=3.0,
            order=4,
            correction_method="dF/F",
            signal_normalization="none",
            fit_using="IRLS",
            maxiter=200,
            c=3,
        )
        self.extract_trial_data(
            align_to="event",
            center_on=["choice_left", "choice_right"],
            trial_bounds=(-2.0, 4.0),
            baseline_bounds=(-2.0, 0.0),
            trial_normalization="none",
            check_overlap=True,
            event_conflict_logic="first",
        )

# dummy loader class for pipeline tests
class DummyLoader(PhotometryLoader):
    def __init__(
            self,
            file: str,
            ) -> None:
        self.file = file
        self.metadata = {'key1': 'value1'}
        return
    
    def extract_data(self) -> dict:
        exp = simulate_data()
        self.metadata['source'] = self.file
        data = dict(
            raw_signal = exp.F_exp,
            raw_isosbestic = exp.F_iso,
            time = exp.time,
            frequency = exp.freq,
            events = exp.event_layer.timestamps_to_dict(),
            metadata = self.metadata,
        )
        return data

@pytest.fixture
def sim():
    return simulate_data()


@pytest.fixture
def experiment(sim):
    return DummyExperiment(
        raw_signal=sim.F_exp,
        raw_isosbestic=sim.F_iso,
        time=sim.time,
        frequency=sim.freq,
        events=sim.event_layer.timestamps_to_dict(),
        metadata={'source':'simulated_test'}
    )


@pytest.fixture
def single_channel_experiment(sim):
    return PhotometryExperiment(
        raw_signal=sim.F_exp,
        raw_isosbestic=None,
        time=sim.time,
        frequency=sim.freq,
        events=sim.event_layer.timestamps_to_dict().copy(),
        metadata={'source':'simulated_test'}
    )


@pytest.fixture
def photometry_data():
    trace_size = (3, 20)
    rng = np.random.default_rng(42)

    obs = pd.DataFrame({
        "animal": ["a", "a", "b"],
        "condition": ["x", "x", "y"],
        "score": [1.0, 3.0, 5.0],
    })

    data = rng.random(size=trace_size, dtype=float)
    time = np.arange(0, 20)
    layers = {'layer' : rng.random(size=trace_size, dtype=float)}
    metadata = {'frequency' : 1.0}

    return PhotometryData.from_arrays(
        obs=obs,
        data=data,
        time_points=time,
        layers=layers,
        metadata=metadata,
    )

@pytest.fixture
def dummy_pipeline():
    test_dir = pathlib.Path(__file__).parent.parent.resolve()
    target_dir = test_dir / 'test_data/dummy_pipeline'
    pipeline = PhotometryPipeline(
        data_directory=target_dir,
        target_type='file',
        recursive=False,
        pattern='dummy_file*.csv',
        loader_cls=DummyLoader,
    )

    return pipeline
