# Signal Processing

This tutorial gives an in depth overview of the ``PhotometryExperiment`` class. ``PhotometryExperiment`` represents one continuous photometry recording and provides tools for preprocessing traces, inspecting reference fits, trimming recordings, exporting continuous data, and extracting trial-wise ``PhotometryData``.

The most important idea is that ``PhotometryExperiment`` works with continuous data, while ``PhotometryData`` works with trial-wise data.

A typical workflow is:

1. Load a continuous experiment.

2. Inspect raw traces and event labels.

3. Preprocess the continuous signal.

4. Extract trial windows around behavioral events.

5. Analyze or save the resulting ``PhotometryData`` object.

# Setup

First we import the packages used throughout the tutorial.


```python
import numpy as np
import pandas as pd
from plotnine import * # type: ignore

from PhoPro import PhotometryExperiment, PhotometryData
```

The examples below use the bundled CSV experiment. To keep each section independent, we will define helper functions that reload a fresh experiment whenever needed.


```python
EVENT_COLS = ['trial_cue', 'lever1', 'lever2', 'shock']


def load_example(isosbestic: bool = True) -> PhotometryExperiment:
    return PhotometryExperiment.load_CSV(
        csv='data/experiments/example_experiment.csv',
        time_col='time',
        signal_col='raw_signal',
        isosbestic_col='raw_isosbestic' if isosbestic else None,
        event_cols=EVENT_COLS,
        annotation_file='data/experiments/example_annotation.json',
        annotation_handler='json',
    )


def load_processed(**preprocess_kwargs) -> PhotometryExperiment:
    exp = load_example()
    exp.id = 'Processed example'
    kwargs = dict(
        cutoff_frequency=3,
        order=4,
        correction_method='dF/F',
        fit_using='IRLS',
        maxiter=2000,
        c=2,
    )
    kwargs.update(preprocess_kwargs)
    exp.preprocess_signal(**kwargs)
    return exp
```

# 1. Loading a Continuous Experiment

``PhotometryExperiment`` can be created directly from arrays, but most workflows use a loader. The convenience class methods ``load_CSV`` and ``load_TDT`` create the appropriate loader and return a ``PhotometryExperiment``.


```python
exp = load_example()
exp.id = 'Dual-channel example'

exp
```




    Dual channel photometry experiment with 100000 timepoints.



A dual-channel experiment contains both the experimental signal and an isosbestic reference. A single-channel experiment has only the experimental signal.


```python
print(exp.channel_mode)
print(exp.has_isosbestic)
print(exp.n_times)
print(exp.frequency)
```

    dual
    True
    100000
    100.0


The continuous arrays are stored on the object. ``time`` aligns with ``raw_signal`` and, when present, ``raw_isosbestic``.


```python
print(exp.time[:5])
print(exp.raw_signal[:5])
print(exp.raw_isosbestic[:5]) #type: ignore
```

    [0.   0.01 0.02 0.03 0.04]
    [59.54784198 60.0574298  59.03015301 59.7573142  59.54165367]
    [47.59561321 47.39307264 47.60456976 48.05470631 48.2274789 ]


Events are stored in a dictionary mapping event labels to timestamp arrays.


```python
print(exp.event_labels)
{k: len(v) for k, v in exp.events.items()}
```

    ['trial_cue', 'lever1', 'lever2', 'shock']





    {'trial_cue': 20, 'lever1': 9, 'lever2': 6, 'shock': 3}



Metadata from annotation files is stored in ``metadata``. Built-in loaders also include the source file path.


```python
exp.metadata
```




    {'source': 'data/experiments/example_experiment.csv',
     'subject': 'animal_1',
     'sex': 'male',
     'age': 'young'}



# 2. Inspecting Raw Data

``plot_dashboard`` gives a quick view of the continuous recording. Before preprocessing, ``raw='auto'`` shows the raw signal and raw isosbestic traces.


```python
exp.plot_dashboard(downsample=20)
```




    
![png](Signal%20Processing_files/Signal%20Processing_17_0.png)
    



The plotting methods return plotnine objects, so they can be modified before display.


```python
p: ggplot = exp.plot_dashboard(downsample=30)

p + labs(title='Raw dual-channel recording')
```




    
![png](Signal%20Processing_files/Signal%20Processing_19_0.png)
    



Continuous data can also be inspected as a dataframe. ``to_wide_dataframe`` keeps one row per timepoint.


```python
raw_wide = exp.to_wide_dataframe(
    downsample=100,
    export_events=True,
)

raw_wide.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>raw_signal</th>
      <th>raw_isosbestic</th>
      <th>trial_cue</th>
      <th>lever1</th>
      <th>lever2</th>
      <th>shock</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.495</td>
      <td>59.842767</td>
      <td>47.842826</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.495</td>
      <td>60.204349</td>
      <td>48.203209</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.495</td>
      <td>60.822751</td>
      <td>48.539868</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.495</td>
      <td>60.502840</td>
      <td>48.514150</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.495</td>
      <td>59.996514</td>
      <td>48.391587</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



``to_long_dataframe`` keeps one row per timepoint and trace source. This is the format used by the dashboard plotting helper.


```python
raw_long = exp.to_long_dataframe(downsample=100)
raw_long.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>source</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.495</td>
      <td>raw_signal</td>
      <td>59.842767</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.495</td>
      <td>raw_signal</td>
      <td>60.204349</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.495</td>
      <td>raw_signal</td>
      <td>60.822751</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.495</td>
      <td>raw_signal</td>
      <td>60.502840</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.495</td>
      <td>raw_signal</td>
      <td>59.996514</td>
    </tr>
  </tbody>
</table>
</div>



# 3. Low-pass Filtering and Reference Fitting

``preprocess_signal`` combines the main preprocessing steps, but the lower-level helpers are also available. The first step is usually a low-pass Butterworth filter.


```python
filtered_signal = exp.low_frequency_pass_butter(
    signal=exp.raw_signal,
    sample_frequency=exp.frequency,
    cutoff_frequency=3,
    order=4,
)

filtered_isosbestic = exp.low_frequency_pass_butter(
    signal=exp.raw_isosbestic,
    sample_frequency=exp.frequency,
    cutoff_frequency=3,
    order=4,
)

filtered_signal[:5]
```




    array([59.53001198, 59.56028827, 59.59053474, 59.6204367 , 59.64970329])



In dual-channel experiments, the isosbestic trace is fitted to the experimental signal before correction. The package supports ordinary least squares and iteratively reweighted least squares, with or without an intercept.


```python
fitted_ref, r2_val, coeffs = exp.fit_isosbestic_to_signal(
    signal=filtered_signal,
    isosbestic=filtered_isosbestic,
    fit_using='IRLS',
    maxiter=2000,
    c=2,
)

print(r2_val)
print(coeffs)
```

    0.9993991834603729
    [0.02467583 1.24905654]


The fitted reference is the trace that will be subtracted or divided away during preprocessing.


```python
pd.DataFrame({
    'time': exp.time[:5],
    'filtered_signal': filtered_signal[:5],
    'filtered_isosbestic': filtered_isosbestic[:5],
    'fitted_reference': fitted_ref[:5],
})
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>filtered_signal</th>
      <th>filtered_isosbestic</th>
      <th>fitted_reference</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00</td>
      <td>59.530012</td>
      <td>47.625996</td>
      <td>59.512238</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.01</td>
      <td>59.560288</td>
      <td>47.656528</td>
      <td>59.550373</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02</td>
      <td>59.590535</td>
      <td>47.684835</td>
      <td>59.585732</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03</td>
      <td>59.620437</td>
      <td>47.710558</td>
      <td>59.617859</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.04</td>
      <td>59.649703</td>
      <td>47.733396</td>
      <td>59.646385</td>
    </tr>
  </tbody>
</table>
</div>



# 4. Dual-channel Preprocessing

``preprocess_signal`` runs filtering, reference fitting, correction, optional whole-signal normalization, and optional artifact correction. For many dual-channel experiments, a 3 Hz low-pass filter, IRLS reference fitting, and ``dF/F`` correction are a reasonable starting point.


```python
exp = load_processed()

print(exp.has_ran_preprocess)
print(exp.signal.shape)
print(exp.fitted_ref.shape)
```

    True
    (100000,)
    (100000,)


After preprocessing, several attributes are populated:

* ``filt_sig``: filtered experimental signal.

* ``filt_iso``: filtered isosbestic signal for dual-channel data.

* ``fitted_ref``: fitted reference trace.

* ``signal``: final processed signal.

* ``metadata['reference_fit']``: fit type, fit quality, and coefficients.

* ``metadata['correction_method']``: the correction method used.


```python
exp.metadata['reference_fit']
```




    {'type': 'isosbestic',
     'r2_val': 0.9993991834603729,
     'coeffs': array([0.02467583, 1.24905654])}




```python
exp.metadata['correction_method']
```




    'dF/F'



After preprocessing, ``plot_dashboard(raw=False)`` shows the filtered signal, fitted reference, and processed trace.


```python
exp.plot_dashboard(raw=False, downsample=20)
```




    
![png](Signal%20Processing_files/Signal%20Processing_36_0.png)
    



The correction method can be changed. ``dF/F`` returns ``(signal - fitted_reference) / fitted_reference``. ``dF`` returns ``signal - fitted_reference``.


```python
df_exp = load_processed(correction_method='dF')

print(df_exp.metadata['correction_method'])
print(df_exp.signal[:5])
```

    dF
    [0.01777443 0.0099152  0.00480323 0.00257781 0.0033181 ]


Whole-signal normalization can be applied after correction. Built-in options are ``'none'``, ``'zscore'``, and ``'nullZ'``.


```python
z_exp = load_processed(signal_normalization='zscore')

print(np.mean(z_exp.signal))
print(np.std(z_exp.signal))
```

    -9.094947017729283e-18
    1.0


Custom correction and normalization functions can be passed directly. A custom correction function receives ``signal`` and ``fitted_ref`` and must return a one-dimensional processed signal.


```python
def percent_dF_F(signal: np.ndarray, fitted_ref: np.ndarray) -> np.ndarray:
    return 100 * (signal - fitted_ref) / np.maximum(fitted_ref, np.finfo(np.float32).eps)

custom_exp = load_processed(correction_method=percent_dF_F)

print(custom_exp.metadata['correction_method'])
print(custom_exp.signal[:5])
```

    percent_dF_F
    [0.02986685 0.0166501  0.00806105 0.00432389 0.00556295]


# 5. Single-channel Preprocessing

Single-channel experiments do not have an isosbestic reference. In that case, ``preprocess_signal`` fits a photobleaching curve and uses single-channel correction methods such as ``dB/B`` or ``dB``.


```python
single = load_example(isosbestic=False)

print(single.channel_mode)
print(single.has_isosbestic)
```

    single
    False



```python
single.preprocess_signal(
    cutoff_frequency=3,
    order=4,
    correction_method='dB/B',
    signal_normalization='none',
    channel_mode='single',
)

print(single.metadata['reference_fit']['type'])
print(single.metadata['reference_fit']['r2_val'])
```

    photobleaching
    0.8024286718698792


The photobleaching curve can also be fit directly with ``fit_photobleaching_curve``.


```python
bleach_curve, bleach_r2, bleach_params = single.fit_photobleaching_curve(
    signal=single.filt_sig,
)

print(bleach_r2)
print(bleach_params)
```

    0.8024286718698792
    [4.00215366e+01 3.20521630e+02 2.36696064e-13 2.12983463e+03
     2.05897601e+01]


# 6. Optional Artifact Detection and Correction

``preprocess_signal`` accepts an ``artifact_detector`` and ``artifact_corrector``. Detectors identify artifact intervals and correctors replace or adjust the processed signal in those intervals.

Artifact settings are experiment-specific, so the safest pattern is to inspect the raw and processed traces, tune the detector on representative recordings, then apply the chosen detector and corrector consistently.


```python
from PhoPro.analysis.artifact import ODS_Detector, Spline_Corrector

# Example configuration. Tune thresholds for your own data before using it in production.
detector = ODS_Detector(
    score_threshold=8,
    jump_score_threshold=8,
    expand_sec=(0.5, 2),
    buffer_sec=1.5,
    n_chunks=50,
)

corrector = Spline_Corrector(
    anchor_sec=(0.2, 0.2),
    correct_spikes=True,
    correct_jumps=True,
)

print(detector)
print(corrector)
```

    <PhoPro.analysis.artifact.ODS_Detector object at 0x348cc46d0>
    <PhoPro.analysis.artifact.Spline_Corrector object at 0x341ea1090>


To run artifact correction, pass the detector and corrector to ``preprocess_signal``. This cell is commented out so the tutorial does not imply that one detector configuration is universally appropriate.


```python
# artifact_exp = load_example()
# artifact_exp.preprocess_signal(
#     cutoff_frequency=3,
#     order=4,
#     correction_method='dF/F',
#     fit_using='IRLS',
#     artifact_detector=detector,
#     artifact_corrector=corrector,
# )
# artifact_exp.artifacts.df.head()
```

# 7. Trimming Continuous Recordings

``trim_times_by_index`` and ``trim_times_by_values`` remove timepoints from the continuous recording. They also trim every present trace and remove event timestamps outside the new time range.

It is best to trim before trial extraction or artifact correction so all derived results stay aligned.


```python
trimmed = load_example()
trimmed.trim_times_by_values(lower=100, upper=200)

print(trimmed.time[0], trimmed.time[-1])
print(trimmed.n_times)
print({k: len(v) for k, v in trimmed.events.items()})
```

    100.0 200.0
    10001
    {'trial_cue': 2, 'lever1': 1, 'lever2': 0, 'shock': 0}


Index trimming uses an inclusive ``start_idx`` and an exclusive ``stop_idx``.


```python
trimmed_by_index = load_example()
trimmed_by_index.trim_times_by_index(start_idx=1000, stop_idx=2000)

print(trimmed_by_index.time[0], trimmed_by_index.time[-1])
print(trimmed_by_index.raw_signal.shape)
```

    10.0 19.99
    (1000,)


# 8. Exporting Continuous Data

Continuous experiment traces can be exported as wide or long dataframes. The export methods include whatever traces are available at the time they are called.


```python
processed_wide = exp.to_wide_dataframe(
    downsample=100,
    export_events=True,
)

processed_wide.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>raw_signal</th>
      <th>raw_isosbestic</th>
      <th>processed_signal</th>
      <th>fitted_reference</th>
      <th>filtered_signal</th>
      <th>filtered_isosbestic</th>
      <th>trial_cue</th>
      <th>lever1</th>
      <th>lever2</th>
      <th>shock</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.495</td>
      <td>59.842767</td>
      <td>47.842826</td>
      <td>0.001123</td>
      <td>59.772316</td>
      <td>59.839362</td>
      <td>47.834213</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.495</td>
      <td>60.204349</td>
      <td>48.203209</td>
      <td>-0.000197</td>
      <td>60.226421</td>
      <td>60.214495</td>
      <td>48.197776</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.495</td>
      <td>60.822751</td>
      <td>48.539868</td>
      <td>0.002501</td>
      <td>60.659348</td>
      <td>60.810882</td>
      <td>48.544379</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.495</td>
      <td>60.502840</td>
      <td>48.514150</td>
      <td>-0.001993</td>
      <td>60.616940</td>
      <td>60.495888</td>
      <td>48.510423</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.495</td>
      <td>59.996514</td>
      <td>48.391587</td>
      <td>-0.007808</td>
      <td>60.481491</td>
      <td>60.009231</td>
      <td>48.401984</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
processed_long = exp.to_long_dataframe(downsample=100)
processed_long.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>source</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.495</td>
      <td>raw_signal</td>
      <td>59.842767</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.495</td>
      <td>raw_signal</td>
      <td>60.204349</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.495</td>
      <td>raw_signal</td>
      <td>60.822751</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.495</td>
      <td>raw_signal</td>
      <td>60.502840</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.495</td>
      <td>raw_signal</td>
      <td>59.996514</td>
    </tr>
  </tbody>
</table>
</div>



Use ``write_csv`` when you want to save the continuous traces to disk.


```python
exp.write_csv(
    file='output/processed_experiment_wide.csv',
    downsample=100,
    export_events=True,
    format='wide',
)

pd.read_csv('output/processed_experiment_wide.csv').head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>time</th>
      <th>raw_signal</th>
      <th>raw_isosbestic</th>
      <th>processed_signal</th>
      <th>fitted_reference</th>
      <th>filtered_signal</th>
      <th>filtered_isosbestic</th>
      <th>trial_cue</th>
      <th>lever1</th>
      <th>lever2</th>
      <th>shock</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.495</td>
      <td>59.842767</td>
      <td>47.842826</td>
      <td>0.001123</td>
      <td>59.772316</td>
      <td>59.839362</td>
      <td>47.834213</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1.495</td>
      <td>60.204349</td>
      <td>48.203209</td>
      <td>-0.000197</td>
      <td>60.226420</td>
      <td>60.214495</td>
      <td>48.197776</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2.495</td>
      <td>60.822751</td>
      <td>48.539868</td>
      <td>0.002501</td>
      <td>60.659348</td>
      <td>60.810882</td>
      <td>48.544379</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>3.495</td>
      <td>60.502840</td>
      <td>48.514150</td>
      <td>-0.001993</td>
      <td>60.616940</td>
      <td>60.495888</td>
      <td>48.510423</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>4.495</td>
      <td>59.996514</td>
      <td>48.391587</td>
      <td>-0.007808</td>
      <td>60.481490</td>
      <td>60.009231</td>
      <td>48.401984</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



# 9. Extracting Trial Data

After preprocessing, ``extract_trial_data`` slices the continuous signal into trial windows and stores the result in ``trial_data`` as a ``PhotometryData`` object.

The key arguments are:

* ``align_to``: event label, event labels, timestamp, or timestamps that define candidate trials.

* ``center_on``: optional event label or labels used as time zero inside each trial.

* ``trial_bounds``: window bounds relative to the final trial center.

* ``event_tolerences``: event annotation windows relative to the original ``align_to`` timestamp.

* ``baseline_bounds``: optional baseline window bounds relative to ``align_to``.

* ``trial_normalization``: optional per-trial normalization using the baseline window.

* ``window_alignment``: ``'nearest'`` sampled windows or ``'interp'`` exact interpolated grids.

* ``invalid_window_policy``: ``'drop'`` or ``'error'`` for windows outside the recording.


```python
trial_exp = load_processed()

trial_exp.extract_trial_data(
    align_to='trial_cue',
    center_on=['lever1', 'lever2'],
    trial_bounds=(-8, 8),
    event_tolerences={
        'lever1': (2, 4),
        'lever2': (2, 4),
        'shock': None,
    },
    baseline_bounds=(-5, -1),
    trial_normalization='zscore',
    check_overlap=True,
    all_events=True,
    window_alignment='nearest',
)

trials = trial_exp.trial_data
trials
```




    Photometry dataset with 20 trials, 1600 timepoints, and 7 observations.



The extracted trials are now avaliable a an ``PhotometryData`` object as the ``trial_data`` attribute.


```python
trials.obs.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>trial_num</th>
      <th>start_time</th>
      <th>stop_time</th>
      <th>trial_cue</th>
      <th>lever1</th>
      <th>lever2</th>
      <th>shock</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>15.52</td>
      <td>31.51</td>
      <td>-3.52</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>65.23</td>
      <td>81.22</td>
      <td>-2.71</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.22</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>113.05</td>
      <td>129.04</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>167.51</td>
      <td>183.50</td>
      <td>-3.94</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>217.89</td>
      <td>233.88</td>
      <td>-3.79</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
p = (
    trials
    .mutate_obs(
        has_shock = lambda d: ~d.obs['shock'].isna(),
    )
    .plot_trials(
        group_on='has_shock',
        downsample=4,
    )
)

p.show()
```


    
![png](Signal%20Processing_files/Signal%20Processing_65_0.png)
    


If ``baseline_bounds`` is supplied, the baseline windows are stored separately as ``baseline_data``.


```python
trial_exp.baseline_data
```




    Photometry dataset with 20 trials, 400 timepoints, and 7 observations.



# 10. Alignment and Centering Semantics

``align_to`` defines which candidate timestamps become trials. ``center_on`` decides where time zero should be in each extracted trial.

If no selected ``center_on`` event is present for a trial, that trial remains centered on its ``align_to`` timestamp.


```python
cue_centered = load_processed()
cue_centered.extract_trial_data(
    align_to='trial_cue',
    center_on=None,
    trial_bounds=(-4, 6),
    trial_normalization='none',
)

cue_centered.trial_data.obs.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>trial_num</th>
      <th>start_time</th>
      <th>stop_time</th>
      <th>trial_cue</th>
      <th>lever1</th>
      <th>lever2</th>
      <th>shock</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>16.00</td>
      <td>25.99</td>
      <td>0.0</td>
      <td>3.52</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>66.52</td>
      <td>76.51</td>
      <td>0.0</td>
      <td>2.71</td>
      <td>NaN</td>
      <td>3.93</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>117.05</td>
      <td>127.04</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>167.57</td>
      <td>177.56</td>
      <td>0.0</td>
      <td>3.94</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>218.10</td>
      <td>228.09</td>
      <td>0.0</td>
      <td>3.79</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



``align_to`` can also receive multiple labels. In that case, all timestamps are pooled and an ``align_event`` column records which event created each trial.


```python
choice_aligned = load_processed()
choice_aligned.extract_trial_data(
    align_to=['lever1', 'lever2'],
    center_on=None,
    trial_bounds=(-4, 4),
    event_tolerences={'shock': (-1, 2)},
    trial_normalization='none',
    invalid_window_policy='drop',
)

choice_aligned.trial_data.obs.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>trial_num</th>
      <th>start_time</th>
      <th>stop_time</th>
      <th>ALIGNMENTS</th>
      <th>shock</th>
      <th>trial_cue</th>
      <th>lever1</th>
      <th>lever2</th>
      <th>align_event</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>19.52</td>
      <td>27.51</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>-3.52</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>lever1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>69.23</td>
      <td>77.22</td>
      <td>0.0</td>
      <td>1.22</td>
      <td>-2.71</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>lever1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>171.51</td>
      <td>179.50</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>-3.94</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>lever1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>221.89</td>
      <td>229.88</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>-3.79</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>lever1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>322.71</td>
      <td>330.70</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>-3.56</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>lever2</td>
    </tr>
  </tbody>
</table>
</div>



Numeric timestamps are accepted too, which is useful for manually defined windows or external event tables.


```python
manual_aligned = load_processed()
manual_aligned.extract_trial_data(
    align_to=[100.0, 200.0, 300.0],
    center_on=None,
    trial_bounds=(-2, 2),
    trial_normalization='none',
)

manual_aligned.trial_data.obs.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>trial_num</th>
      <th>start_time</th>
      <th>stop_time</th>
      <th>ALIGNMENTS</th>
      <th>trial_cue</th>
      <th>lever1</th>
      <th>lever2</th>
      <th>shock</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>98.0</td>
      <td>101.99</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>198.0</td>
      <td>201.99</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>298.0</td>
      <td>301.99</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



# 11. Window Alignment: Nearest vs Interp

``window_alignment='nearest'`` rounds each requested window to sampled timepoints. ``window_alignment='interp'`` interpolates the signal onto an exact centered grid. Both return fixed-size trial matrices, but the time grids can differ slightly.


```python
nearest_exp = load_processed()
nearest_exp.extract_trial_data(
    align_to='trial_cue',
    center_on='lever1',
    trial_bounds=(-2, 4),
    trial_normalization='none',
    window_alignment='nearest',
)

interp_exp = load_processed()
interp_exp.extract_trial_data(
    align_to='trial_cue',
    center_on='lever1',
    trial_bounds=(-2, 4),
    trial_normalization='none',
    window_alignment='interp',
)

print(nearest_exp.trial_data.ts[:5])
print(interp_exp.trial_data.ts[:5])
```

    [-2.   -1.99 -1.98 -1.97 -1.96]
    [-2.   -1.99 -1.98 -1.97 -1.96]


# 12. Invalid Windows

A trial window is invalid if the requested trial or baseline interval extends outside the continuous recording. ``invalid_window_policy='error'`` makes this explicit. ``'drop'`` removes those trials and records their indexes in ``metadata['invalid_windows']``.


```python
edge_case = load_processed()

try:
    edge_case.extract_trial_data(
        align_to=[1.0, 100.0],
        center_on=None,
        trial_bounds=(-8, 8),
        trial_normalization='none',
        invalid_window_policy='error',
    )
except ValueError as err:
    print(err)
```

    Invalid trial windows that extend outside signal range at trial indicies [0]



```python
drop_case = load_processed()
drop_case.extract_trial_data(
    align_to=[1.0, 100.0],
    center_on=None,
    trial_bounds=(-8, 8),
    trial_normalization='none',
    invalid_window_policy='drop',
)

print(drop_case.metadata['invalid_windows'])
print(drop_case.trial_data)
```

    [0]
    Photometry dataset with 1 trials, 1600 timepoints, and 8 observations.


# 13. Trial-wise Normalization

Baseline-dependent normalizations require ``baseline_bounds``. Built-in trial normalizations are:

* ``'none'``: leave trial windows unchanged.

* ``'zero'``: subtract the baseline center.

* ``'zscore'``: center and scale by baseline standard deviation.

* ``'mad'``: robustly scale by baseline median absolute deviation.

* ``'amp'``: scale by baseline amplitude.

Custom normalization functions receive ``trial_signals`` and ``baseline_signals`` and must return an array with the same shape as ``trial_signals``.


```python
zero_exp = load_processed()
zero_exp.extract_trial_data(
    align_to='trial_cue',
    center_on=['lever1', 'lever2'],
    trial_bounds=(-4, 6),
    baseline_bounds=(-5, -1),
    trial_normalization='zero',
)

zero_exp.trial_data.X[:2, :5]
```




    array([[ 0.00092141,  0.00078276,  0.00064532,  0.00051357,  0.00039203],
           [-0.00111444, -0.00087772, -0.00067669, -0.0005156 , -0.00039684]])




```python
def baseline_percent_change(trial_signals: np.ndarray, baseline_signals: np.ndarray) -> np.ndarray:
    baseline_mean = np.nanmean(baseline_signals, axis=1, keepdims=True)
    return 100 * (trial_signals - baseline_mean) / np.maximum(np.abs(baseline_mean), np.finfo(np.float32).eps)

custom_trial_norm = load_processed()
custom_trial_norm.extract_trial_data(
    align_to='trial_cue',
    center_on=['lever1', 'lever2'],
    trial_bounds=(-4, 6),
    baseline_bounds=(-5, -1),
    trial_normalization=baseline_percent_change,
)

custom_trial_norm.trial_data.X[:2, :5]
```




    array([[165.19599303, 140.33778971, 115.69673012,  92.07650281,
             70.28621454],
           [-78.25548717, -61.63363494, -47.5170997 , -36.20528335,
            -27.86618643]])



# 14. Saving Extracted Trial Data

Once trial data have been extracted, ``trial_data`` is a ``PhotometryData`` object and can be saved as ``.h5ad`` or zarr.


```python
trial_exp.trial_data.write_h5ad('output/signal_processing_trials.h5ad')

loaded_trials = PhotometryData.read_h5ad('output/signal_processing_trials.h5ad')
loaded_trials
```




    Photometry dataset with 20 trials, 1600 timepoints, and 7 observations.



# Summary

``PhotometryExperiment`` is the bridge between raw continuous recordings and trial-wise analysis. It keeps raw traces, event timestamps, metadata, filtered traces, fitted references, processed signals, artifact results, and extracted trial data on one object.

The most common pattern is:

1. Load an experiment with ``load_CSV``, ``load_TDT``, or a loader class.

2. Inspect raw traces and event labels with ``plot_dashboard`` and the dataframe export methods.

3. Run ``preprocess_signal`` with settings appropriate for dual-channel or single-channel data.

4. Use ``extract_trial_data`` to build a ``PhotometryData`` object aligned to behaviorally meaningful events.

5. Save the processed continuous data or extracted trial data for later analysis.

# AI Use Disclaimer

Generative AI was used to assist in the creation of this tutorial. I plan to replace it in the future with a more polished version.
