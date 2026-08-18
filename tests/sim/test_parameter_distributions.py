import numpy as np
import pytest

from PhoPro.sim.SimulatedPhotometry import SimulatedPhotometry
from PhoPro.sim.kernels import gamma_kernel
from PhoPro.sim.layers import (
    EventLayer,
    EventSpec,
    SamplerNormal,
    SamplerUniform,
)


def test_sampler_validation():
    with pytest.raises(ValueError, match='low'):
        SamplerUniform(2.0, 1.0)

    with pytest.raises(ValueError, match='std'):
        SamplerNormal(1.0, -0.1)


def test_event_spec_samples_once_per_onset():
    spec = EventSpec(
        onsets=np.array([1.0, 2.0, 3.0]),
        amplitude=SamplerUniform(0.1, 0.2),
        kernel_func=gamma_kernel,
        kernel_params={
            'shape_k': 3.0,
            'tau_sec': SamplerUniform(0.2, 0.4),
        },
    )

    amplitudes, params = spec.sample_parameters(
        size=spec.onsets.size,
        rng=np.random.default_rng(5),
    )

    assert amplitudes.shape == spec.onsets.shape
    assert params['shape_k'].shape == spec.onsets.shape
    assert params['tau_sec'].shape == spec.onsets.shape
    assert np.all((0.1 <= amplitudes) & (amplitudes <= 0.2))
    assert np.all((0.2 <= params['tau_sec']) & (params['tau_sec'] <= 0.4))


def test_event_layer_distribution_render_is_reproducible():
    time = np.arange(0.0, 12.0, 0.02)
    layer = EventLayer(
        specs={
            'cue': EventSpec(
                onsets=np.array([1.0, 4.0, 7.0]),
                amplitude=SamplerUniform(0.1, 0.2),
                kernel_func=gamma_kernel,
                kernel_params={
                    'shape_k': 3.0,
                    'tau_sec': SamplerUniform(0.2, 0.4),
                },
            )
        },
        max_duration_sec=10.0,
    )

    first = layer.render(time, 50.0, np.random.default_rng(9))
    second = layer.render(time, 50.0, np.random.default_rng(9))
    different = layer.render(time, 50.0, np.random.default_rng(10))

    assert np.array_equal(first, second)
    assert not np.array_equal(first, different)
    assert np.isfinite(first).all()
    assert np.max(first) > 0


def test_fixed_event_layer_render_remains_backward_compatible():
    time = np.arange(0.0, 5.0, 0.02)
    layer = EventLayer(
        specs={
            'cue': EventSpec(
                onsets=np.array([1.0]),
                amplitude=0.2,
                kernel_func=gamma_kernel,
                kernel_params={'shape_k': 3.0, 'tau_sec': 0.3},
            )
        },
        max_duration_sec=5.0,
    )

    rendered = layer.render(time, 50.0)

    assert rendered.shape == time.shape
    assert np.isclose(rendered.max(), 0.2, rtol=0.01)


def test_simulated_photometry_supports_distributed_event_parameters():
    sim = SimulatedPhotometry.from_parameters(
        length_sec=20,
        frequency=20,
        n_events=3,
        event_buffer_sec=2.0,
        event_amplitude=SamplerUniform(0.01, 0.03),
        event_kernel=gamma_kernel,
        event_kernel_params={
            'shape_k': 3.0,
            'tau_sec': SamplerUniform(0.2, 0.5),
        },
        bleaching_params_iso=None,
        iso_bleach_scale=1.0,
        mult_noise_magnitude_exp=None,
        gaussian_noise_scale_exp=None,
        movement_attenuation=None,
        seed=17,
        max_event_duration_sec=10.0,
    )
    first_event_trace = sim.E.copy()
    first_signal = sim.F_exp.copy()

    sim.build_traces(seed=17)

    assert np.array_equal(first_event_trace, sim.E)
    assert np.array_equal(first_signal, sim.F_exp)
    assert np.isfinite(sim.F_exp).all()
