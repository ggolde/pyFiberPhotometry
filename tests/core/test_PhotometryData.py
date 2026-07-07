import warnings
import json

import numpy as np
import pandas as pd
import pytest

from PhoPro.core.PhotometryData import PhotometryData
from PhoPro.analysis.peaks import PeakResult
from PhoPro.utils.io import clean_axis_dtype


# --- constructors and basic shape contracts ---

def test_from_arrays_fixture_shapes_and_metadata(photometry_data):
    assert isinstance(photometry_data, PhotometryData)
    assert photometry_data.n_trials == 3
    assert photometry_data.n_times == 20
    assert photometry_data.X.shape == (3, 20)
    assert photometry_data.obs.index.tolist() == ["0", "1", "2"]
    assert photometry_data.get_layer("layer").shape == (3, 20)
    assert photometry_data.uns["frequency"] == 1.0
    assert np.isclose(photometry_data.freq, 1.0)


def test_from_arrays_rejects_mismatched_layer_shape(photometry_data):
    with pytest.raises(AssertionError):
        PhotometryData.from_arrays(
            obs=photometry_data.obs,
            data=photometry_data.X,
            time_points=photometry_data.ts,
            layers={"bad": np.ones((photometry_data.n_trials, photometry_data.n_times - 1))},
        )


def test_str_repr_and_property_setters_preserve_accessors(photometry_data):
    text = str(photometry_data)
    assert "3 trials" in text
    assert repr(photometry_data) == text

    new_x = np.ones_like(photometry_data.X)
    new_obs = photometry_data.obs.assign(flag=True)
    new_var = photometry_data.var.assign(time_idx=np.arange(photometry_data.n_times))
    new_uns = {"frequency": 2.0, "source": "setter_test"}

    photometry_data.X = new_x
    photometry_data.obs = new_obs
    photometry_data.var = new_var
    photometry_data.uns = new_uns

    assert np.array_equal(photometry_data.X, new_x)
    assert photometry_data.obs["flag"].all()
    assert "time_idx" in photometry_data.var
    assert photometry_data.uns["source"] == "setter_test"
    assert np.isclose(photometry_data.dt, 1 / photometry_data.freq)


def test_h5ad_roundtrip_preserves_data_layers_and_metadata(photometry_data, tmp_path):
    path = tmp_path / "photometry.h5ad"

    photometry_data.write_h5ad(str(path))
    loaded = PhotometryData.read_h5ad(str(path))

    assert isinstance(loaded, PhotometryData)
    assert np.allclose(loaded.X, photometry_data.X)
    assert np.allclose(loaded.get_layer("layer"), photometry_data.get_layer("layer"))
    assert loaded.obs["animal"].tolist() == photometry_data.obs["animal"].tolist()
    assert loaded.uns["frequency"] == photometry_data.uns["frequency"]


def test_clean_axis_dtype_matches_anndata_string_categorical_logic():
    df = pd.DataFrame(
        {
            "low_card": ["b", "a", "b"],
            "unique": ["a", "b", "c"],
            "numeric": pd.Series([1, None, 3], dtype="object"),
            "payload": pd.Series(
                [[1, np.int64(2)], {"a": np.float64(1.5)}, None],
                dtype="object",
            ),
        },
        index=[0, 1, 2],
    )

    cleaned = clean_axis_dtype(df)

    assert cleaned.index.tolist() == ["0", "1", "2"]
    assert isinstance(cleaned["low_card"].dtype, pd.CategoricalDtype)
    assert cleaned["low_card"].cat.categories.tolist() == ["a", "b"]
    assert not isinstance(cleaned["unique"].dtype, pd.CategoricalDtype)
    assert cleaned["numeric"].dtype == float
    assert str(cleaned["payload"].dtype) == "string"
    assert json.loads(cleaned["payload"].iloc[0]) == [1, 2]
    assert json.loads(cleaned["payload"].iloc[1]) == {"a": 1.5}
    assert pd.isna(cleaned["payload"].iloc[2])


def test_h5ad_write_cleans_axis_dtypes_without_mutating_source(tmp_path):
    obs = pd.DataFrame(
        {
            "animal": ["b", "a", "b"],
            "score": pd.Series([1, None, 3], dtype="object"),
            "payload": pd.Series([[1, 2], {"a": np.int64(3)}, None], dtype="object"),
        }
    )
    data = np.ones((3, 4))
    time = np.arange(4, dtype=float)
    trials = PhotometryData.from_arrays(obs=obs, data=data, time_points=time)
    path = tmp_path / "cleaned.h5ad"

    trials.write_h5ad(str(path))
    loaded = PhotometryData.read_h5ad(str(path))

    assert isinstance(loaded.obs["animal"].dtype, pd.CategoricalDtype)
    assert loaded.obs["animal"].tolist() == ["b", "a", "b"]
    assert loaded.obs["score"].dtype == float
    assert loaded.obs["score"].isna().tolist() == [False, True, False]
    assert json.loads(loaded.obs["payload"].iloc[0]) == [1, 2]
    assert json.loads(loaded.obs["payload"].iloc[1]) == {"a": 3}
    assert pd.isna(loaded.obs["payload"].iloc[2])
    assert not isinstance(trials.obs["animal"].dtype, pd.CategoricalDtype)
    assert trials.obs["score"].dtype == object
    assert trials.obs["payload"].dtype == object


def test_zarr_roundtrip_preserves_shape_if_zarr_is_available(photometry_data, tmp_path):
    pytest.importorskip("zarr")
    path = tmp_path / "photometry.zarr"

    photometry_data.write_zarr(str(path))
    loaded = PhotometryData.read_zarr(str(path))

    assert loaded.X.shape == photometry_data.X.shape
    assert loaded.get_layer("layer").shape == photometry_data.get_layer("layer").shape


def test_append_on_disk_h5ad_creates_then_appends(photometry_data, tmp_path):
    path = tmp_path / "combined.h5ad"
    other = photometry_data.filter_rows(np.array([True, False, True]), inplace=False)

    photometry_data.append_on_disk_h5ad(str(path))
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Observation names are not unique")
        other.append_on_disk_h5ad(str(path))
    loaded = PhotometryData.read_h5ad(str(path))

    assert loaded.n_trials == photometry_data.n_trials + other.n_trials
    assert loaded.n_times == photometry_data.n_times
    assert loaded.obs.index.is_unique


# --- row, metadata, and object operations ---

def test_copy_returns_independent_object(photometry_data):
    copied = photometry_data.copy()
    copied.X[0, 0] = -1.0

    assert isinstance(copied, PhotometryData)
    assert copied is not photometry_data
    assert photometry_data.X[0, 0] != -1.0


def test_pipe_passes_object_and_arguments(photometry_data):
    def add_scaled_score(data, scale):
        return data.mutate_obs(scaled_score=data.obs["score"] * scale)

    out = photometry_data.pipe(add_scaled_score, 2.0)

    assert out.obs["scaled_score"].tolist() == [2.0, 6.0, 10.0]


def test_downsample_reduces_time_dimension_layers_and_frequency(photometry_data):
    out = photometry_data.downsample(2)

    assert out.n_trials == photometry_data.n_trials
    assert out.n_times == 10
    assert out.X.shape == (3, 10)
    assert out.get_layer("layer").shape == (3, 10)
    assert out.obs["animal"].tolist() == photometry_data.obs["animal"].tolist()
    assert np.isclose(out.uns["frequency"], 0.5)
    assert np.isclose(out.freq, 0.5, atol=0.1)


def test_downsample_none_returns_self(photometry_data):
    assert photometry_data.downsample(None) is photometry_data


def test_combine_obj_returns_combined_object(photometry_data):
    out = photometry_data.combine_obj(photometry_data, inplace=False)

    assert isinstance(out, PhotometryData)
    assert out.n_trials == 6
    assert out.n_times == photometry_data.n_times
    assert out.obs["animal"].tolist() == ["a", "a", "b", "a", "a", "b"]


def test_combine_obj_inplace_updates_object(photometry_data):
    other = photometry_data.copy()

    result = photometry_data.combine_obj(other, inplace=True)

    assert result is None
    assert photometry_data.n_trials == 6
    assert photometry_data.n_times == other.n_times


def test_filter_rows_returns_subset_with_layers_and_metadata(photometry_data):
    out = photometry_data.filter_rows(np.array([True, False, True]), inplace=False)

    assert isinstance(out, PhotometryData)
    assert out.n_trials == 2
    assert out.n_times == photometry_data.n_times
    assert out.get_layer("layer").shape == (2, photometry_data.n_times)
    assert out.uns["frequency"] == photometry_data.uns["frequency"]


def test_filter_rows_inplace_updates_object(photometry_data):
    result = photometry_data.filter_rows(np.array([True, False, True]), inplace=True)

    assert result is None
    assert photometry_data.n_trials == 2
    assert photometry_data.obs["animal"].tolist() == ["a", "b"]


def test_filter_rows_none_is_noop_for_copy_and_inplace(photometry_data):
    out = photometry_data.filter_rows(None, inplace=False)
    result = photometry_data.filter_rows(None, inplace=True)

    assert out is photometry_data
    assert result is None
    assert photometry_data.n_trials == 3


def test_mutate_obs_accepts_values_callables_and_none(photometry_data):
    out = photometry_data.mutate_obs(
        constant="ok",
        doubled=lambda data: data.obs["score"] * 2,
        skipped=lambda data: None,
    )

    assert out.obs["constant"].tolist() == ["ok", "ok", "ok"]
    assert out.obs["doubled"].tolist() == [2.0, 6.0, 10.0]
    assert "skipped" not in out.obs
    assert "constant" not in photometry_data.obs


def test_add_obs_columns_and_drop_obs_columns(photometry_data):
    photometry_data.add_obs_columns({"session": ["s1", "s1", "s2"]})

    assert "session" in photometry_data.obs.columns
    assert photometry_data.obs["session"].tolist() == ["s1", "s1", "s2"]

    photometry_data.drop_obs_columns(["session"])
    assert "session" not in photometry_data.obs.columns


def test_add_metadata_updates_uns(photometry_data):
    photometry_data.add_metadata({"lab": "test_lab", "batch": 2}, keys=["lab"])

    assert photometry_data.uns["lab"] == "test_lab"
    assert "batch" not in photometry_data.uns


def test_get_text_value_counts_summarizes_column(photometry_data):
    text = photometry_data.get_text_value_counts("animal")

    assert "a: 2" in text
    assert "b: 1" in text


# --- aggregation ---

def test_collapse_groups_trials_and_adds_metric_layers(photometry_data):
    out = photometry_data.collapse(
        group_on=["animal"],
        data_cols=["score"],
        collapse_cols=["condition"],
    )

    assert isinstance(out, PhotometryData)
    assert out.n_trials == 2
    assert "n" in out.obs.columns
    assert "score" in out.obs.columns
    assert "score_std" in out.obs.columns
    assert "condition" in out.obs.columns
    assert "std" in out.adata.layers
    assert sorted(out.obs["animal"].tolist()) == ["a", "b"]


def test_collapse_without_groups_collapses_all_trials(photometry_data):
    out = photometry_data.collapse(
        group_on=None,
        data_cols=["score"],
        count_col="n",
    )

    assert out.n_trials == 1
    assert out.obs.loc["0", "n"] == photometry_data.n_trials
    assert np.isclose(out.obs.loc["0", "score"], np.mean([1.0, 3.0, 5.0]))


# --- windowing ---

def test_window_returns_centered_timebase(photometry_data):
    out = photometry_data.window(
        centers=10.0,
        bounds=(-3.0, 4.0),
    )

    assert isinstance(out, PhotometryData)
    assert out.n_trials == photometry_data.n_trials
    assert np.isclose(out.ts[0], -3.0)
    assert np.isclose(out.ts[-1], 3.0)
    assert out.X.shape[1] == 7


def test_window_accepts_centers_from_obs_column_and_recenters_events(photometry_data):
    photometry_data.add_obs_columns({
        "center": [5.0, 10.0, 15.0],
        "event_time": [6.0, 11.0, 16.0],
    })

    out = photometry_data.window(
        centers="center",
        bounds=(-2.0, 3.0),
        event_cols=["event_time"],
    )

    assert out.n_trials == photometry_data.n_trials
    assert out.n_times == 5
    assert np.allclose(out.obs["event_time"].to_numpy(dtype=float), 1.0)
    assert out.get_layer("layer").shape == out.X.shape


def test_window_drop_invalid_masks_obs_data_and_layers(photometry_data):
    photometry_data.add_obs_columns({
        "center": [1.0, 10.0, 18.0],
        "event_time": [1.0, 10.0, 18.0],
    })

    out = photometry_data.window(
        centers="center",
        bounds=(-2.0, 2.0),
        event_cols=["event_time"],
        invalid_window_policy="drop",
    )

    assert out.n_trials == 2
    assert out.obs["animal"].tolist() == ["a", "b"]
    assert np.allclose(out.obs["event_time"].to_numpy(dtype=float), 0.0)
    assert out.X.shape == (2, 4)
    assert out.get_layer("layer").shape == (2, 4)


def test_window_invalid_policy_error_raises(photometry_data):
    with pytest.raises(ValueError, match="Invalid trial windows"):
        photometry_data.window(
            centers=np.asarray([1.0, 10.0, 18.0]),
            bounds=(-2.0, 2.0),
            invalid_window_policy="error",
        )


def test_window_interp_strategy_uses_exact_shared_timebase(photometry_data):
    out = photometry_data.window(
        centers=np.asarray([5.25, 10.5, 14.75]),
        bounds=(-1.0, 2.0),
        strategy="interp",
    )

    assert out.X.shape == (photometry_data.n_trials, 3)
    assert np.allclose(out.ts, np.asarray([-1.0, 0.0, 1.0]))


def test_window_rejects_unknown_strategy(photometry_data):
    with pytest.raises(ValueError, match="not recognized"):
        photometry_data.window(
            centers=10.0,
            bounds=(-1.0, 1.0),
            strategy="bogus",
        )


# --- numerical summaries and dataframe exports ---

def test_difference_calculates_repeated_discrete_derivative(photometry_data):
    data = np.asarray([[0.0, 1.0, 3.0, 6.0]])
    obj = PhotometryData.from_arrays(
        obs=pd.DataFrame({"trial": [0]}),
        data=data,
        time_points=np.arange(data.shape[1]),
    )

    assert np.allclose(obj.difference(n=1), [[0.0, 1.0, 2.0, 3.0]])
    assert np.allclose(obj.difference(n=2), [[0.0, 1.0, 1.0, 1.0]])


def test_area_under_curve_matches_numpy_trapezoid(photometry_data):
    auc = photometry_data.area_under_curve()
    expected = np.trapezoid(y=photometry_data.X, x=photometry_data.ts, axis=1)

    assert np.allclose(auc, expected)


def test_area_under_curve_applies_transformation(photometry_data):
    auc = photometry_data.area_under_curve(transformation=lambda x: x * 2)
    expected = np.trapezoid(y=photometry_data.X * 2, x=photometry_data.ts, axis=1)

    assert np.allclose(auc, expected)


def test_area_under_curve_can_window_before_integrating(photometry_data):
    auc = photometry_data.area_under_curve(centers=10.0, bounds=(-2.0, 2.0))
    windows = photometry_data.window(centers=10.0, bounds=(-2.0, 2.0))
    expected = np.trapezoid(y=windows.X, x=windows.ts, axis=1)

    assert np.allclose(auc, expected)


def test_trials_to_long_df_has_expected_columns_and_size(photometry_data):
    df = photometry_data.trials_to_long_df(
        err_layer="layer",
        obs_cols=["animal", "condition"],
        downsample=None,
    )

    assert {"trial_idx", "time_idx", "time", "signal", "animal", "condition", "layer"} <= set(df.columns)
    assert len(df) == photometry_data.n_trials * photometry_data.n_times
    assert df["time_idx"].min() == 0
    assert df["time_idx"].max() == photometry_data.n_times - 1


def test_trials_to_long_df_can_export_named_layer_and_downsample(photometry_data):
    df = photometry_data.trials_to_long_df(
        layer="layer",
        obs_cols="animal",
        downsample=10,
    )

    assert len(df) == photometry_data.n_trials * 2
    assert df["time_idx"].max() == 1
    assert np.allclose(
        df.loc[df["trial_idx"] == "0", "signal"].to_numpy(),
        photometry_data.downsample(10).get_layer("layer")[0],
    )


def test_trials_to_wide_df_has_obs_and_signal_columns(photometry_data):
    df = photometry_data.trials_to_wide_df(
        obs_cols=["animal"],
        signal_prefix="sig",
        downsample=2,
    )

    signal_cols = [col for col in df.columns if col.startswith("sig.")]
    assert df["animal"].tolist() == photometry_data.obs["animal"].tolist()
    assert len(signal_cols) == 10
    assert len(df) == photometry_data.n_trials


def test_trials_to_wide_df_defaults_to_all_obs_and_can_export_layer(photometry_data):
    df = photometry_data.trials_to_wide_df(layer="layer", signal_prefix="layer_sig")

    assert {"animal", "condition", "score"} <= set(df.columns)
    assert len([col for col in df.columns if col.startswith("layer_sig.")]) == photometry_data.n_times


# --- peak wrappers and metadata statistics ---

def _peak_test_data() -> PhotometryData:
    signals = np.zeros((2, 10), dtype=float)
    signals[0, 2:4] = 2.0
    signals[1, 6:8] = -2.0
    obs = pd.DataFrame({"trial": [0, 1], "group": ["a", "b"]})
    return PhotometryData.from_arrays(obs=obs, data=signals, time_points=np.arange(10))


def test_detect_peaks_static_threshold_returns_peak_result():
    obj = _peak_test_data()
    baselines = np.zeros((1, obj.n_times), dtype=float)

    result = obj.detect_peaks_static_threshold(
        center_method="zeros",
        scale_method="ones",
        test_magnitude=0.5,
        baselines=baselines,
        direction="both",
    )

    assert isinstance(result, PeakResult)
    assert result.n_peaks == 2
    assert set(result.df["direction"]) == {"positive", "negative"}


def test_detect_peaks_static_threshold_rejects_baseline_row_mismatch():
    obj = _peak_test_data()

    with pytest.raises(ValueError, match="Baselines"):
        obj.detect_peaks_static_threshold(
            center_method="zeros",
            scale_method="ones",
            baselines=np.zeros((3, obj.n_times)),
        )


def test_detect_peaks_rolling_threshold_returns_peak_result():
    obj = _peak_test_data()

    result = obj.detect_peaks_rolling_threshold(
        window_width_sec=3.0,
        center_method="zeros",
        scale_method="ones",
        test_magnitude=0.5,
        direction="positive",
    )

    assert isinstance(result, PeakResult)
    assert result.n_peaks == 1
    assert result.df.iloc[0]["trial_idx"] == 0


# --- plotting ---

def test_plot_trials_returns_plotnine_object(photometry_data):
    plot = photometry_data.plot_trials(
        sel=np.array([True, False, True]),
        label_with=["animal", "condition"],
        group_on="animal",
        err_layer="layer",
        downsample=None,
        line_kwargs={"color": "black"},
    )

    assert hasattr(plot, "draw")
