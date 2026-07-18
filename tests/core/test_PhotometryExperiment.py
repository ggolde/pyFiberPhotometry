import numpy as np
import pandas as pd
import pytest

from PhoPro.core.PhotometryExperiment import PhotometryExperiment
from PhoPro.core.PhotometryData import PhotometryData
from PhoPro.utils.equations import neg_bi_exponential_5


# --- preprocessing ---

def test_preprocess_signal_outputs(experiment):
    experiment.preprocess_signal(detrend=False)

    assert np.issubdtype(experiment.signal.dtype, np.floating)
    assert experiment.signal.shape == experiment.time.shape
    assert np.isfinite(experiment.signal).all()
    assert np.isfinite(experiment.fitted_ref).all()
    assert experiment.filt_sig.shape == experiment.raw_signal.shape
    assert experiment.fitted_ref.shape == experiment.raw_signal.shape
    assert "reference_fit" in experiment.metadata
    assert experiment.metadata["reference_fit"]["type"] == "isosbestic"
    assert "r2_val" in experiment.metadata["reference_fit"]
    assert np.isfinite(experiment.metadata["reference_fit"]["r2_val"])


def test_preprocess_recovers_signal_reasonably(experiment, sim):
    experiment.preprocess_signal(
        detrend=False,
        correction_method="dF/F",
        signal_normalization="nullZ",
        fit_using="IRLS",
    )

    recovered = experiment.signal
    truth = sim.E[:recovered.size]

    corr = np.corrcoef(recovered, truth)[0, 1]
    assert np.isfinite(corr)
    assert corr > 0.35


def test_preprocess_detrends_both_dual_channel_signals(experiment, monkeypatch):
    calls = []

    def detrend_signal(signal, window_dur_sec):
        calls.append((signal.copy(), window_dur_sec))
        return signal - 1.0, 1.0, []

    monkeypatch.setattr(experiment, "detrend_signal", detrend_signal)

    experiment.preprocess_signal(
        cutoff_frequency=None,
        correction_method="none",
        detrend=True,
        window_dur_sec=2.5,
    )

    assert len(calls) == 2
    assert np.array_equal(calls[0][0], experiment.raw_signal)
    assert np.array_equal(calls[1][0], experiment.raw_isosbestic)
    assert calls[0][1] == calls[1][1] == 2.5
    assert np.array_equal(experiment.filt_sig, experiment.raw_signal - 1.0)
    assert np.array_equal(experiment.filt_iso, experiment.raw_isosbestic - 1.0)


def test_preprocess_rejects_channel_incompatible_correction_methods(
    experiment,
    single_channel_experiment,
):
    with pytest.raises(ValueError, match="single channel"):
        experiment.preprocess_signal(correction_method="dB/B", detrend=False)

    with pytest.raises(ValueError, match="dual channel"):
        single_channel_experiment.preprocess_signal(correction_method="dF/F", detrend=False)


def test_fit_photobleaching_curve_recovers_smooth_bleaching_trend(experiment):
    params = (2.0, 12.5, 1.0, 100.0, 5.0)
    expected = neg_bi_exponential_5(experiment.time, *params)
    signal = expected + 1e-3 * np.sin(0.7 * experiment.time)

    fitted_curve, r2_val, fitted_params = experiment.fit_photobleaching_curve(
        signal=signal,
        window_dur_sec=2.0,
    )

    assert fitted_curve.shape == signal.shape
    assert len(fitted_params) == 5
    assert np.isfinite(fitted_curve).all()
    assert np.isfinite(fitted_params).all()
    assert r2_val > 0.99
    assert np.nanmedian(np.abs(fitted_curve - expected)) < 0.05


def test_preprocess_single_channel_processing(single_channel_experiment):
    exp = single_channel_experiment

    exp.preprocess_signal(
        detrend=False,
        correction_method="dB/B",
        signal_normalization="none",
        cutoff_frequency=3.0,
    )

    assert exp.channel_mode == "single"
    assert exp.metadata["reference_fit"]["type"] == "photobleaching"
    assert exp.metadata["correction_method"] == "dB/B"
    assert exp.filt_sig.shape == exp.raw_signal.shape
    assert exp.fitted_ref.shape == exp.raw_signal.shape
    assert exp.signal.shape == exp.raw_signal.shape
    assert np.isfinite(exp.filt_sig).all()
    assert np.isfinite(exp.fitted_ref).all()
    assert np.isfinite(exp.signal).all()


def test_preprocess_requires_detector_when_corrector_is_given(experiment):
    class Corrector:
        def correct(self, signal, time, artifacts):
            return signal

    with pytest.raises(ValueError, match="artifact_detector"):
        experiment.preprocess_signal(artifact_corrector=Corrector(), detrend=False)


def test_preprocess_accepts_callable_correction_and_normalization(experiment):
    def correction(signal, fitted_ref):
        return signal - fitted_ref

    def normalize(signal):
        return signal / np.nanstd(signal)

    experiment.preprocess_signal(
        detrend=False,
        correction_method=correction,
        signal_normalization=normalize,
    )

    assert experiment.metadata["correction_method"] == "correction"
    assert np.isfinite(experiment.signal).all()


def test_constructor_properties_reflect_channel_and_pipeline_state(experiment, single_channel_experiment):
    assert experiment.has_isosbestic
    assert experiment.channel_mode == "dual"
    assert not experiment.has_ran_preprocess
    assert not experiment.has_ran_extraction
    assert experiment.n_times == experiment.time.size

    assert not single_channel_experiment.has_isosbestic
    assert single_channel_experiment.channel_mode == "single"


def test_preprocess_delegates_artifact_detector_and_corrector(experiment):
    class Detector:
        def detect(self, signal, reference, time):
            self.signal_seen = signal
            self.reference_seen = reference
            return "artifact_result"

    class Corrector:
        def correct(self, signal, time, artifacts):
            assert artifacts == "artifact_result"
            return signal + 1.0

    detector = Detector()
    experiment.preprocess_signal(
        detrend=False,
        artifact_detector=detector,
        artifact_corrector=Corrector(),
    )

    assert experiment.artifacts == "artifact_result"
    assert detector.reference_seen is experiment.fitted_ref
    assert np.isfinite(experiment.signal).all()


def test_lowpass_filter_preserves_shape_and_reduces_high_frequency_power(experiment):
    time = experiment.time
    signal = np.sin(2 * np.pi * 0.2 * time) + 0.5 * np.sin(2 * np.pi * 6.0 * time)

    filtered = experiment.low_frequency_pass_butter(
        signal,
        sample_frequency=experiment.frequency,
        cutoff_frequency=1.0,
        order=4,
    )

    assert filtered.shape == signal.shape
    assert np.std(filtered - np.sin(2 * np.pi * 0.2 * time)) < np.std(signal - np.sin(2 * np.pi * 0.2 * time))


def test_fit_isosbestic_supports_callable_and_rejects_unknown_method(experiment):
    signal = np.asarray([2.0, 4.0, 6.0, 8.0])
    iso = np.asarray([1.0, 2.0, 3.0, 4.0])

    fitted, r2_val, params = experiment.fit_isosbestic_to_signal(
        signal,
        iso,
        fit_using=lambda sig, ref: (ref * 2.0, {"scale": 2.0}),
    )

    assert np.allclose(fitted, signal)
    assert np.isclose(r2_val, 1.0)
    assert params == {"scale": 2.0}

    with pytest.raises(ValueError, match="not recognized"):
        experiment.fit_isosbestic_to_signal(signal, iso, fit_using="bogus")


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("dF", np.asarray([1.0, 2.0])),
        ("dB", np.asarray([1.0, 2.0])),
        ("dF/F", np.asarray([1.0, 1.0])),
        ("dB/B", np.asarray([1.0, 1.0])),
        ("none", np.asarray([2.0, 4.0])),
    ],
)
def test_apply_correction_method_variants(experiment, method, expected):
    out = experiment._apply_correction_method(
        correction_method=method,
        signal=np.asarray([2.0, 4.0]),
        fitted_ref=np.asarray([1.0, 2.0]),
    )

    assert np.allclose(out, expected)


@pytest.mark.parametrize("method", ["zscore", "nullZ", "none"])
def test_apply_signal_normalization_variants(experiment, method):
    signal = np.asarray([1.0, 2.0, 3.0])

    out = experiment._apply_signal_normalization(method, signal)

    assert out.shape == signal.shape
    assert np.isfinite(out).all()


def test_apply_normalization_helpers_reject_unknown_methods(experiment):
    with pytest.raises(ValueError, match="correction"):
        experiment._apply_correction_method("bad", np.ones(3), np.ones(3))

    with pytest.raises(ValueError, match="normalization"):
        experiment._apply_signal_normalization("bad", np.ones(3))


# --- trial extraction: basic outputs ---

def test_extract_trial_data_creates_trial_and_baseline_objects(experiment, sim):
    experiment.run_pipeline()

    assert isinstance(experiment.trial_data, PhotometryData)
    assert isinstance(experiment.baseline_data, PhotometryData)
    assert experiment.trial_data.n_trials == sim.event_layer.specs["event"].onsets.size
    assert experiment.baseline_data.n_trials == sim.event_layer.specs["event"].onsets.size
    assert experiment.trial_data.n_times > 0
    assert experiment.baseline_data.n_times > 0
    assert "event" in experiment.trial_data.obs.columns
    assert np.isfinite(experiment.trial_data.X).all()
    assert np.isfinite(experiment.baseline_data.X).all()


def test_extract_trial_data_center_on_none_uses_alignment_event(experiment):
    experiment.preprocess_signal(detrend=False)

    experiment.extract_trial_data(
        align_to="event",
        trial_bounds=(-1.0, 2.0),
        baseline_bounds=None,
        trial_normalization="none",
    )

    assert experiment.trial_data.n_trials == experiment.events["event"].size
    assert "event" in experiment.trial_data.obs.columns
    assert np.allclose(experiment.trial_data.obs["event"].to_numpy(dtype=float), 0.0)


def test_extract_trial_data_single_center_on_does_not_overlap(experiment):
    experiment.preprocess_signal(detrend=False)

    experiment.extract_trial_data(
        align_to="event",
        center_on="choice_left",
        trial_bounds=(-1.0, 2.0),
        baseline_bounds=None,
        trial_normalization="none",
        check_overlap=True,
    )

    assert "choice_left" in experiment.trial_data.obs.columns
    assert experiment.trial_data.obs["choice_left"].notna().any()


# --- trial extraction: alignment inputs ---

def test_extract_trial_data_accepts_eventless_alignment_timestamps(experiment):
    experiment.preprocess_signal(detrend=False)

    centers = [10.0, 20.0, 30.0]
    experiment.extract_trial_data(
        align_to=centers,
        trial_bounds=(-1.0, 2.0),
        baseline_bounds=None,
        trial_normalization="none",
    )

    assert isinstance(experiment.trial_data, PhotometryData)
    assert experiment.trial_data.n_trials == len(centers)
    assert "ALIGNMENTS" in experiment.trial_data.obs.columns
    assert "align_event" not in experiment.trial_data.obs.columns
    assert np.allclose(experiment.trial_data.obs["ALIGNMENTS"].to_numpy(dtype=float), 0.0)
    assert np.isfinite(experiment.trial_data.X).all()


def test_extract_trial_data_accepts_scalar_integer_alignment_timestamp(experiment):
    experiment.preprocess_signal(detrend=False)

    experiment.extract_trial_data(
        align_to=20,
        trial_bounds=(-1.0, 2.0),
        baseline_bounds=None,
        trial_normalization="none",
    )

    assert experiment.trial_data.n_trials == 1
    assert "ALIGNMENTS" in experiment.trial_data.obs.columns
    assert np.allclose(experiment.trial_data.obs["ALIGNMENTS"].to_numpy(dtype=float), 0.0)


def test_extract_trial_data_accepts_multiple_alignment_event_labels(experiment):
    experiment.preprocess_signal(detrend=False)

    experiment.events["align_a"] = np.asarray([10.0, 20.0])
    experiment.events["align_b"] = np.asarray([10.0, 15.0])

    experiment.extract_trial_data(
        align_to=["align_b", "align_a"],
        trial_bounds=(-1.0, 2.0),
        baseline_bounds=None,
        trial_normalization="none",
    )

    assert isinstance(experiment.trial_data, PhotometryData)
    assert experiment.trial_data.n_trials == 4
    assert "ALIGNMENTS" in experiment.trial_data.obs.columns
    assert "align_event" in experiment.trial_data.obs.columns
    assert experiment.trial_data.obs["align_event"].tolist() == [
        "align_b",
        "align_a",
        "align_b",
        "align_a",
    ]
    assert np.allclose(experiment.trial_data.obs["ALIGNMENTS"].to_numpy(dtype=float), 0.0)


# --- trial extraction: validation and errors ---

def test_extract_trial_data_overlap_check_raises(experiment):
    experiment.preprocess_signal(detrend=False)

    experiment.events["choice_left"] = experiment.events["event"] + 0.3
    experiment.events["choice_right"] = experiment.events["event"] + 0.4

    with pytest.raises(ValueError, match="over lap|overlap"):
        experiment.extract_trial_data(
            align_to="event",
            center_on=["choice_left", "choice_right"],
            trial_bounds=(-1.0, 2.0),
            baseline_bounds=(-1.0, 0.0),
            check_overlap=True,
        )


def test_extract_trial_data_reports_missing_alignment_labels(experiment):
    experiment.preprocess_signal(detrend=False)

    with pytest.raises(KeyError, match="missing"):
        experiment.extract_trial_data(
            align_to=["event", "missing"],
            trial_bounds=(-1.0, 2.0),
        )


def test_extract_trial_data_reports_missing_center_on_labels(experiment):
    experiment.preprocess_signal(detrend=False)

    with pytest.raises(KeyError, match="missing_center"):
        experiment.extract_trial_data(
            align_to="event",
            center_on=["choice_left", "missing_center"],
            trial_bounds=(-1.0, 2.0),
        )


@pytest.mark.parametrize("align_to", [[], "empty_event"])
def test_extract_trial_data_rejects_empty_alignments(experiment, align_to):
    experiment.preprocess_signal(detrend=False)
    experiment.events["empty_event"] = np.asarray([], dtype=float)

    with pytest.raises(ValueError, match="align_to|timestamp|event label"):
        experiment.extract_trial_data(
            align_to=align_to,
            trial_bounds=(-1.0, 2.0),
        )


@pytest.mark.parametrize("trial_normalization", ["zscore", "zero", "mad", "amp"])
def test_extract_trial_data_requires_baseline_for_baseline_normalizations(
    experiment,
    trial_normalization,
):
    experiment.preprocess_signal(detrend=False)

    with pytest.raises(ValueError, match="Baseline bounds"):
        experiment.extract_trial_data(
            align_to="event",
            trial_bounds=(-1.0, 2.0),
            baseline_bounds=None,
            trial_normalization=trial_normalization,
        )


def test_extract_trial_data_none_tolerances_default_to_trial_bounds(experiment):
    experiment.preprocess_signal(detrend=False)

    experiment.extract_trial_data(
        align_to="event",
        center_on="choice_left",
        trial_bounds=(-1.0, 2.0),
        event_tolerences={"choice_left": None},
        baseline_bounds=None,
        trial_normalization="none",
    )

    assert "choice_left" in experiment.trial_data.obs.columns
    assert experiment.trial_data.obs["choice_left"].notna().any()


# --- trial extraction: invalid windows ---

def test_extract_trial_data_drop_invalid_windows_keeps_align_event_aligned(experiment):
    experiment.preprocess_signal(detrend=False)

    experiment.events["align_a"] = np.asarray([0.5, 30.0])
    experiment.events["align_b"] = np.asarray([40.0, experiment.time[-1] - 0.25])

    experiment.extract_trial_data(
        align_to=["align_a", "align_b"],
        trial_bounds=(-1.0, 2.0),
        baseline_bounds=None,
        trial_normalization="none",
        invalid_window_policy="drop",
    )

    assert experiment.metadata["invalid_windows"] == [0, 3]
    assert experiment.trial_data.n_trials == 2
    assert experiment.trial_data.obs["align_event"].tolist() == ["align_a", "align_b"]


def test_extract_trial_data_invalid_window_policy_error_raises(experiment):
    experiment.preprocess_signal(detrend=False)

    with pytest.raises(ValueError, match="Invalid trial windows"):
        experiment.extract_trial_data(
            align_to=[0.5],
            trial_bounds=(-1.0, 2.0),
            invalid_window_policy="error",
        )


def test_extract_trial_data_all_invalid_windows_raises(experiment):
    experiment.preprocess_signal(detrend=False)

    with pytest.raises(ValueError, match="No trials remain"):
        experiment.extract_trial_data(
            align_to=[0.5],
            trial_bounds=(-1.0, 2.0),
            invalid_window_policy="drop",
        )


# --- trial extraction: normalization ---

def test_extract_trial_data_custom_normalization_receives_baseline(experiment):
    experiment.preprocess_signal(detrend=False)
    calls = {}

    def normalize(trial_signals, baseline_signals):
        calls["trial_shape"] = trial_signals.shape
        calls["baseline_shape"] = baseline_signals.shape
        return trial_signals - np.nanmean(baseline_signals, axis=1, keepdims=True)

    experiment.extract_trial_data(
        align_to="event",
        trial_bounds=(-1.0, 2.0),
        baseline_bounds=(-1.0, 0.0),
        trial_normalization=normalize,
    )

    assert calls["trial_shape"] == experiment.trial_data.X.shape
    assert calls["baseline_shape"] == experiment.baseline_data.X.shape
    assert np.isfinite(experiment.trial_data.X).all()


def test_extract_trial_data_interp_alignment_has_shared_timebase(experiment):
    experiment.preprocess_signal(detrend=False)

    experiment.extract_trial_data(
        align_to=[10.25, 20.5, 30.75],
        trial_bounds=(-1.0, 2.0),
        window_alignment="interp",
        trial_normalization="none",
    )

    assert np.isclose(experiment.trial_data.ts[0], -1.0)
    assert np.isclose(experiment.trial_data.ts[-1], 1.95)
    assert np.allclose(experiment.trial_data.obs["ALIGNMENTS"].to_numpy(dtype=float), 0.0)


def test_find_timestamp_in_intervals_supports_first_last_and_mean(experiment):
    timestamps = np.asarray([1.0, 1.5, 2.0, 5.0])
    intervals = np.asarray([[0.5, 2.0], [4.0, 6.0], [7.0, 8.0]])

    assert np.allclose(
        experiment._find_timestamp_in_intervals("event", timestamps, intervals, logic="first")["event"],
        [1.0, 5.0, np.nan],
        equal_nan=True,
    )
    assert np.allclose(
        experiment._find_timestamp_in_intervals("event", timestamps, intervals, logic="last")["event"],
        [2.0, 5.0, np.nan],
        equal_nan=True,
    )
    assert np.allclose(
        experiment._find_timestamp_in_intervals("event", timestamps, intervals, logic="mean")["event"],
        [1.5, 5.0, np.nan],
        equal_nan=True,
    )

    with pytest.raises(ValueError, match="not recognized"):
        experiment._find_timestamp_in_intervals("event", timestamps, intervals, logic="bogus")


def test_find_timestamp_in_intervals_all_returns_numbered_occurrences(experiment):
    timestamps = np.asarray([1.0, 1.5, 2.0, 5.0])
    intervals = np.asarray([[0.5, 2.0], [4.0, 6.0], [7.0, 8.0]])

    result = experiment._find_timestamp_in_intervals(
        "event",
        timestamps,
        intervals,
        logic="all",
    )

    assert list(result) == ["event", "event_occurrence_2", "event_occurrence_3"]
    assert np.allclose(result["event"], [1.0, 5.0, np.nan], equal_nan=True)
    assert np.allclose(result["event_occurrence_2"], [1.5, np.nan, np.nan], equal_nan=True)
    assert np.allclose(result["event_occurrence_3"], [2.0, np.nan, np.nan], equal_nan=True)


def test_find_timestamp_in_intervals_all_handles_no_matches(experiment):
    result = experiment._find_timestamp_in_intervals(
        "event",
        timestamps=np.asarray([1.0, 2.0, 3.0]),
        time_intervals=np.asarray([[4.0, 5.0], [6.0, 7.0]]),
        logic="all",
    )

    assert list(result) == ["event"]
    assert np.allclose(result["event"], [np.nan, np.nan], equal_nan=True)


def test_annotate_intervals_all_adds_repeated_event_columns(experiment):
    result = experiment._annotate_intervals(
        align_label="cue",
        series=np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        centers=np.asarray([1.0, 3.0]),
        events={
            "lick": np.asarray([1.1, 1.2, 1.3, 3.2]),
            "empty": np.asarray([], dtype=float),
        },
        tolorences={
            "lick": (0.0, 0.4),
            "empty": (0.0, 0.4),
            "missing": (0.0, 0.4),
        },
        logic="all",
    )

    assert list(result) == ["cue", "lick", "lick_occurrence_2", "lick_occurrence_3", "empty", "missing"]
    assert np.allclose(result["cue"], [1.0, 3.0])
    assert np.allclose(result["lick"], [1.1, 3.2], equal_nan=True)
    assert np.allclose(result["lick_occurrence_2"], [1.2, np.nan], equal_nan=True)
    assert np.allclose(result["lick_occurrence_3"], [1.3, np.nan], equal_nan=True)
    assert np.allclose(result["empty"], [np.nan, np.nan], equal_nan=True)
    assert np.allclose(result["missing"], [np.nan, np.nan], equal_nan=True)


def test_extract_trial_data_all_event_conflict_adds_repeated_event_columns(experiment):
    experiment.preprocess_signal(detrend=False)
    experiment.events["repeat_event"] = np.sort(np.concatenate([
        experiment.events["event"] + 0.2,
        experiment.events["event"] + 0.4,
    ]))

    experiment.extract_trial_data(
        align_to="event",
        center_on="repeat_event",
        trial_bounds=(-1.0, 2.0),
        event_tolerences={"repeat_event": (0.0, 0.6)},
        all_events=False,
        baseline_bounds=None,
        trial_normalization="none",
        event_conflict_logic="all",
    )

    assert "repeat_event" in experiment.trial_data.obs.columns
    assert "repeat_event_occurrence_2" in experiment.trial_data.obs.columns
    assert np.allclose(experiment.trial_data.obs["repeat_event"].to_numpy(dtype=float), 0.0)
    assert np.allclose(experiment.trial_data.obs["repeat_event_occurrence_2"].to_numpy(dtype=float), 0.2)


def test_find_interval_bounds_and_nearest_timestamp_mask(experiment):
    bounds = experiment._find_interval_bounds(
        series=np.asarray([0.0, 1.0, 2.0, 3.0, 4.0]),
        centers=np.asarray([1.0, 3.0]),
        bounds=(-0.5, 0.5),
    )
    mask = experiment._nearest_timestamp_mask(
        times=np.asarray([0.0, 1.0, 2.0, 3.0]),
        timestamps=np.asarray([0.2, 2.8]),
    )

    assert np.array_equal(bounds, np.asarray([[1, 2], [3, 4]]))
    assert mask.tolist() == [True, False, False, True]


def test_create_windows_rejects_unknown_strategy(experiment):
    with pytest.raises(ValueError, match="not recognized"):
        experiment._create_windows(
            signal=experiment.raw_signal,
            time=experiment.time,
            events={},
            centers=np.asarray([10.0]),
            bounds=(-1.0, 1.0),
            strategy="bogus",
        )


# --- export, trimming, and plotting ---

def test_to_wide_dataframe_includes_named_traces_and_event_columns(experiment):
    experiment.preprocess_signal(detrend=False)

    df = experiment.to_wide_dataframe(export_events=True)

    expected = {
        "time",
        "raw_signal",
        "raw_isosbestic",
        "processed_signal",
        "fitted_reference",
        "filtered_signal",
        "filtered_isosbestic",
        "event",
    }
    assert expected <= set(df.columns)
    assert len(df) == experiment.n_times
    assert df["event"].sum() == experiment.events["event"].size


def test_to_long_dataframe_includes_filtered_sources(experiment):
    experiment.preprocess_signal(detrend=False)

    df = experiment.to_long_dataframe()

    assert {"time", "source", "value"} <= set(df.columns)
    assert {
        "raw_signal",
        "raw_isosbestic",
        "fitted_reference",
        "processed_signal",
        "filtered_signal",
        "filtered_isosbestic",
    } <= set(df["source"])


def test_write_csv_writes_wide_and_long_tables(experiment, tmp_path):
    experiment.preprocess_signal(detrend=False)
    wide_path = tmp_path / "wide.csv"
    long_path = tmp_path / "long.csv"

    experiment.write_csv(str(wide_path), format="wide")
    experiment.write_csv(str(long_path), format="long")

    assert "raw_signal" in pd.read_csv(wide_path).columns
    assert "source" in pd.read_csv(long_path).columns

    with pytest.raises(ValueError, match="not recognized"):
        experiment.write_csv(str(tmp_path / "bad.csv"), format="bad")


def test_load_csv_classmethod_constructs_experiment_from_csv(tmp_path):
    path = tmp_path / "experiment.csv"
    df = pd.DataFrame({
        "time": [0.0, 1.0, 2.0, 3.0],
        "signal": [1.0, 2.0, 3.0, 4.0],
        "isosbestic": [1.0, 1.0, 1.0, 1.0],
        "cue": [0, 1, 0, 0],
    })
    df.to_csv(path, index=False)

    exp = PhotometryExperiment.load_CSV(
        str(path),
        event_cols="cue",
        downsample=None,
    )

    assert isinstance(exp, PhotometryExperiment)
    assert np.allclose(exp.events["cue"], [1.0])


def test_trim_times_by_index_filters_all_timeseries_and_events(experiment):
    experiment.preprocess_signal(detrend=False)
    experiment.events["edge"] = np.asarray([experiment.time[0], experiment.time[10], experiment.time[-1]])

    experiment.trim_times_by_index(start_idx=5, stop_idx=15)

    assert experiment.n_times == 10
    assert experiment.raw_signal.shape == experiment.time.shape
    assert experiment.signal.shape == experiment.time.shape
    assert np.all(experiment.events["edge"] >= experiment.time[0])
    assert np.all(experiment.events["edge"] <= experiment.time[-1])


def test_trim_times_by_values_validates_bounds(experiment):
    with pytest.raises(ValueError, match="Lower bound"):
        experiment.trim_times_by_values(lower=5.0, upper=4.0)

    with pytest.raises(ValueError, match="less than"):
        experiment.trim_times_by_index(start_idx=4, stop_idx=4)

    with pytest.raises(ValueError, match="outside"):
        experiment.trim_times_by_index(start_idx=-1, stop_idx=4)


def test_trim_times_by_values_selects_inclusive_time_range(experiment):
    lower = experiment.time[10]
    upper = experiment.time[20]

    experiment.trim_times_by_values(lower=lower, upper=upper)

    assert np.isclose(experiment.time[0], lower)
    assert np.isclose(experiment.time[-1], upper)


def test_plot_dashboard_returns_plot_object_and_requires_preprocessed_signal(experiment):
    raw_plot = experiment.plot_dashboard(raw=True, downsample=None)
    assert hasattr(raw_plot, "draw")

    with pytest.raises(ValueError, match="preprocess_signal"):
        experiment.plot_dashboard(raw=False)

    experiment.preprocess_signal(detrend=False)
    processed_plot = experiment.plot_dashboard(raw=False, downsample=None)
    assert processed_plot is not None
