import json

import numpy as np

from PhoPro.sim.SimulatedLibrary import SimulatedLibrary, SimulatedParamLoader
from PhoPro.sim.kernels import gamma_kernel
from PhoPro.sim.layers import SamplerNormal, SamplerUniform


def test_simulated_library_round_trips_samplers_and_callables(tmp_path):
    amplitude_sampler = SamplerNormal(mean=0.02, std=0.001)
    tau_sampler = SamplerUniform(low=0.2, high=0.4)
    library = SimulatedLibrary.from_permutations(
        contanst_kwargs={
            'length_sec': 8.0,
            'frequency': 20.0,
            'event_kernel': gamma_kernel,
            'event_amplitude': amplitude_sampler,
            'event_kernel_params': {
                'shape_k': 3.0,
                'tau_sec': tau_sampler,
            },
            'event_buffer_sec': 1.0,
            'bleaching_params_iso': None,
            'iso_bleach_scale': 1.0,
            'mult_noise_magnitude_exp': None,
            'gaussian_noise_scale_exp': None,
            'movement_attenuation': None,
            'max_event_duration_sec': 5.0,
        },
        to_permute_kwargs={'n_events': [3]},
        seed=17,
    )
    params_file = tmp_path / 'params.json'

    library.to_json(params_file)

    encoded = json.loads(params_file.read_text())['0']
    assert encoded['event_kernel']['__phopro_type__'] == 'callable'
    assert encoded['event_amplitude']['__phopro_type__'] == 'parameter_sampler'
    assert encoded['event_kernel_params']['tau_sec']['name'] == 'uniform'

    loaded = SimulatedLibrary.from_json(params_file)
    params = loaded.params['0']
    assert params['event_kernel'] is gamma_kernel
    assert params['event_amplitude'] == amplitude_sampler
    assert params['event_kernel_params']['tau_sec'] == tau_sampler
    assert params['CONDITION_ID'] is None

    data = SimulatedParamLoader(json=str(params_file), key='0').extract_data()
    assert data['raw_signal'].shape == data['time'].shape
    assert data['raw_isosbestic'].shape == data['time'].shape
    assert data['events']['trial_cue'].size == 3
    assert np.isfinite(data['raw_signal']).all()


def test_simulated_library_supports_no_across_or_permutation_dimensions():
    library = SimulatedLibrary.from_permutations(
        contanst_kwargs={'frequency': 20.0},
        to_permute_kwargs=None,
        across_kwargs=None,
        replicates=2,
        seed=5,
    )

    assert library.n_samples == 2
    assert {params['frequency'] for params in library.params.values()} == {20.0}
    assert {params['ACROSS_ID'] for params in library.params.values()} == {0}
    assert {params['PERM_ID'] for params in library.params.values()} == {0}
    assert {params['CONDITION_ID'] for params in library.params.values()} == {None}
    assert {params['REP_ID'] for params in library.params.values()} == {0, 1}
