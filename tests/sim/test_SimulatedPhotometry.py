import numpy as np

from PhoPro.core.PhotometryData import PhotometryData
from PhoPro.core.PhotometryExperiment import PhotometryExperiment
from PhoPro.sim.SimulatedPhotometry import SimulatedPhotometry
from PhoPro.sim.kernels import alpha_kernel, gaussian_kernel
from PhoPro.sim.layers import EventSpec


def make_small_sim(seed=11):
    return SimulatedPhotometry.from_parameters(
        length_sec=30,
        frequency=20,
        event_label="cue",
        n_events=4,
        event_buffer_sec=2.0,
        event_kernel=alpha_kernel,
        event_kernel_params={"tau_sec": 0.25},
        event_amplitude=0.05,
        bleaching_params_iso=None,
        iso_bleach_scale=1.0,
        gaussian_noise_scale_exp=0.05,
        gaussian_noise_scale_iso=None,
        movement_attenuation=0.2,
        n_spike_artifacts=0,
        n_jump_artifacts=0,
        seed=seed,
    )


def test_from_parameters_builds_all_public_trace_arrays():
    sim = make_small_sim()

    assert sim.time.shape == sim.F_exp.shape == sim.F_iso.shape
    assert sim.clean_exp.shape == sim.time.shape
    assert sim.N_exp.shape == sim.time.shape
    assert sim.E.shape == sim.time.shape
    assert sim.freq == 20
    assert sim.event_layer.event_labels == ["cue"]
    assert np.isfinite(sim.F_exp).all()
    assert np.isfinite(sim.F_iso).all()


def test_build_traces_is_repeatable_with_same_seed():
    sim = make_small_sim(seed=5)
    first = sim.F_exp.copy()

    sim.build_traces(seed=5)
    second = sim.F_exp.copy()

    assert np.allclose(first, second)


def test_get_add_and_relative_event_methods_update_event_specs():
    sim = make_small_sim()
    onsets = np.asarray([5.0, 10.0])

    sim.add_event(
        label="reward",
        onsets=onsets,
        amplitude=0.1,
        kernel_func=gaussian_kernel,
        kernel_params={"center_sec": 0.2, "sigma_sec": 0.05},
    )
    sim.add_event_relative_to(
        relative_to="cue",
        time_range=(0.1, 0.2),
        labels=["choice_left", "choice_right"],
        overall_prob=1.0,
        choice_probs=[1.0, 0.0],
        amplitudes=[0.2, 0.3],
        kernel_funcs=[SimulatedPhotometry.kernel_alpha, SimulatedPhotometry.kernel_alpha],
        kernel_params=[{"tau_sec": 0.2}, {"tau_sec": 0.3}],
        seed=3,
    )

    reward = sim.get_event_spec("reward")
    assert isinstance(reward, EventSpec)
    assert np.array_equal(reward.onsets, onsets)
    assert "choice_left" in sim.event_layer.event_labels
    assert sim.get_event_spec("choice_right").onsets.size == 0


def test_kernel_static_methods_return_expected_shape():
    time = np.linspace(0.0, 1.0, 25)

    kernels = [
        SimulatedPhotometry.kernel_gamma(time, amplitude=1.0, shape_k=3, tau_sec=0.2),
        SimulatedPhotometry.kernel_exp_deacy(time, amplitude=1.0, tau_sec=0.2),
        SimulatedPhotometry.kernel_alpha(time, amplitude=1.0, tau_sec=0.2),
        SimulatedPhotometry.kernel_diff_of_exp(time, amplitude=1.0, tau_rise_sec=0.1, tau_decay_sec=0.4),
        SimulatedPhotometry.kernel_sum_of_exp(time, amplitude=1.0, tau_fast_sec=0.1, tau_slow_sec=0.5, fast_weight=0.25),
        SimulatedPhotometry.kernel_gaussian(time, amplitude=1.0, center_sec=0.5, sigma_sec=0.1),
    ]

    for kernel in kernels:
        assert kernel.shape == time.shape
        assert np.isfinite(kernel).all()


def test_to_photometry_experiment_supports_single_and_dual_channel_exports():
    sim = make_small_sim()

    dual = sim.to_PhotometryExperiment(downsample=None)
    single = sim.to_PhotometryExperiment(as_single_channel=True, downsample=2)

    assert isinstance(dual, PhotometryExperiment)
    assert dual.raw_isosbestic is not None
    assert dual.frequency == sim.freq
    assert "cue" in dual.events

    assert single.raw_isosbestic is None
    assert single.frequency == sim.freq / 2
    assert single.raw_signal.size == sim.F_exp.size // 2


def test_to_photometry_data_uses_event_extraction_path():
    sim = make_small_sim()

    trials = sim.to_PhotometryData(
        align_to="cue",
        trial_bounds=(-0.5, 1.0),
        center_on=None,
        window_alignment="interp",
        invalid_window_policy="drop",
    )

    assert isinstance(trials, PhotometryData)
    assert trials.n_trials == sim.get_event_spec("cue").onsets.size
    assert "cue" in trials.obs.columns
    assert np.allclose(trials.obs["cue"].to_numpy(dtype=float), 0.0)


def test_to_long_dataframe_exports_condensed_and_full_layers():
    sim = make_small_sim()

    condensed = sim.to_long_dataframe(condensed=True, downsample=5)
    full = sim.to_long_dataframe(condensed=False, only_layers=["full_signal"], downsample=10)

    assert {"time", "layer", "trace", "value"} <= set(condensed.columns)
    assert {"neural_trace", "photobleaching", "artifacts", "noise", "full_signal"} <= set(condensed["layer"])
    assert set(full["layer"]) == {"full_signal"}
    assert set(full["trace"]) == {"experimental", "isosbestic"}


def test_plot_methods_return_plotnine_objects():
    sim = make_small_sim()

    layers_plot = sim.plot_layers(downsample=10)
    traces_plot = sim.plot_traces(downsample=10)

    assert hasattr(layers_plot, "draw")
    assert hasattr(traces_plot, "draw")
