# Introduction

This tutorial requires ipykernel and Jupyter Notebooks. It covers the basics of the ``PhoPro`` package for photometry data processing, handling, and analysis.

While using this package does not require much coding or technical knowledge, it is the good to get familiar with:

* [Basic Python](https://www.w3schools.com/python/default.asp).

* The [numpy package](https://www.w3schools.com/python/numpy/default.asp) and numpy arrays.

* The [pandas package](https://www.w3schools.com/python/pandas/) and pandas DataFrames.

* The [components of photometry measurements]().

* (Optional) the [plotnine package](https://plotnine.org) as that is what this package uses as a plotting backend.

``PhoPro`` is organized into 4 main modules:

* **PhotometryLoader** - classes to load various photometry data formats.

* **PhotometryExperiment** - a class for processing and windowing photometry experiments with many avaliable preprocessing methods.

* **PhotometryData** - a class holding trial-wise signals and metadata with advanced filtering, averaging, windowing, and analysis functionality.

* **PhotometryPipeline** - a class for easy, highly-customizable bulk processing of photometry data.

# Setup

First we need to import the necessary packages. If you have trouble importing any of the ``PhoPro`` modules check to make sure your enviroment is active in the Jupyter Notebook and all dependencies are install properly (see the [Installation]() guide on the docs).


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PhoPro import PhotometryExperiment, PhotometryData, PhotometryPipeline
```

# 1. Loading Data

We will begin by learning how to load data from a CSV file into the ``PhoPro`` package. The CSV file contains the columns: "time", containing timepoints; "raw_signal" and "raw_isosbestic", containing the signal from experimental and isosbestic wavelengths; and "trial_cue", "lever1", "lever2", and "shock", containing event timestamps.


```python
# import the loader class for your specifc data format
from PhoPro import CSVLoader

# initialize the loader
loader = CSVLoader(
    csv= 'data/experiments/example_experiment.csv',
    time_col= 'time',
    signal_col= 'raw_signal',
    isosbestic_col= 'raw_isosbestic',
    event_cols= ['trial_cue', 'lever1', 'lever2', 'shock']
)

# extract the data and load it into the PhotometryExperiment class
exp = loader.load()

# see if data was successfully loaded
exp
```




    Dual channel photometry experiment with 100000 timepoints.



We can also load in important metadata from annotation files. Currently supported formats are JSON and YML. Built-in loaders also automatically pass down the source file to experimental metadata.


```python
# initialize the loader
loader = CSVLoader(
    csv= 'data/experiments/example_experiment.csv',
    time_col= 'time',
    signal_col= 'raw_signal',
    isosbestic_col= 'raw_isosbestic',
    event_cols= ['trial_cue', 'lever1', 'lever2', 'shock'],
    annotation_file='data/experiments/example_annotation.json',
    annotation_handler='json',
)

# extract the data and load it into the PhotometryExperiment class
exp = loader.load()

# see what matedata was picked up
exp.metadata
```




    {'source': 'data/experiments/example_experiment.csv',
     'subject': 'animal_1',
     'sex': 'male',
     'age': 'young'}



Other built in loaders, like the ``TDTLoader``, behave very similarly to the ``CSVLoader``.

# 2. Signal processing
Once you have your data loaded, the next step in most workflows is to preprocess the signals. The exact steps depend on whether or not the experiment conatins an isosbestic trace:

* **Dual-channel** experiments have an isosbestic reference signal. They are generally more common and produce a cleaner signal by using the concentration agnostic isosbestic signal to correct signal artifacts and photobleaching. 

* **Single-channel** experiments do NOT have an isosbestic signal. They are usually less common and require extra steps to correct for photobleaching and artifacts.

## 2.1. Dual-channel processing

The general scheme for processing a **dual-channel** experiment is:

1. Apply a low-pass filter to both the experimental and isosbestic signals to filter out high frequency noise.

2. Fit the isosbestic to the experimental signal using a linear equation: $\text{Exp} = \beta_1 \cdot \text{Iso} + \beta_0$.

3. Apply an isosbestic correction either ``dF/F`` or ``dF``.
    * ``dF/F`` $= (\text{Exp.} - \text{Iso}_\text{ fitted}) / \text{Iso}_\text{ fitted}$
    * ``dF`` $= (\text{Exp.} - \text{Iso}_\text{ fitted})$

4. Optionally normalize the entire signal.

For most experiments the following parameters are recommended based on the findings in [Keevers et al., 2025](https://doi.org/10.1117/1.NPh.12.2.025003):

* A low-pass filter with a cutoff of 3 Hz and order of 4.
* Fitting the isosbestic curve with iteratively-reweighted least squares (IRLS).
* Using the ``dF/F`` correction.
* No whole-trial normalization.

Let's begin by loading in our same example experiment from the previous section.


```python
loader = CSVLoader(
    csv= 'data/experiments/example_experiment.csv',
    time_col= 'time',
    signal_col= 'raw_signal',
    isosbestic_col= 'raw_isosbestic',
    event_cols= ['trial_cue', 'lever1', 'lever2', 'shock'],
    annotation_file='data/experiments/example_annotation.json',
    annotation_handler='json',
)
exp = loader.load()

# this time we will assign an ID to our experiment
exp.id = 'Dual-channel example'
```

Now we can examine the raw traces with ``.plot_dashboard()``.


```python
exp.plot_dashboard()
```




    
![png](Introduction_files/Introduction_15_0.png)
    



We can see that the raw experimental and isosbestic signals have significant photobleaching attenuation and movement artifacts. Notice, however, that the stark movement artifacts are shared between experimental and isosbestic signals. This is why using the isosbestic as a reference channel is an effective way to correct for artifacts.

Now is a good time to mention that all the ploting methods in this package return a ``ggplot`` object from the ``plotnine`` package, that can be directly modified to fit you publication or presentation needs. For example:


```python
from plotnine import ggplot, theme, labs

p: ggplot = exp.plot_dashboard()
p = (
    p
    + labs(title='New title')
    + theme(figure_size=(3.5, 3))
)
p.show()
```


    
![png](Introduction_files/Introduction_18_0.png)
    


Now let's process the signal using the ``.preprocess_signal()`` methods. This method will:

* Apply a low-pass [butterworth filter](https://en.wikipedia.org/wiki/Butterworth_filter) with a specified cutoff frequency and order

* Fit the isosbestic to the experimental signal using a customizable method

* Apply the isosbestic correction


```python
exp.preprocess_signal(
    # lowpass butterworth params
    cutoff_frequency=3,
    order=4,

    # correction method and isosbestic fit params
    correction_method='dF/F',
    fit_using='IRLS',
    maxiter=2000,
    c=2,

    # artifact correction params (mostly useful for single channel experiments)
    artifact_detector=None,
    artifact_corrector=None,

    channel_mode='auto'
)

# examine the fitted isosbestic and processed signal
exp.plot_dashboard(raw=False)
```




    
![png](Introduction_files/Introduction_20_0.png)
    



Notice how:

* The filtered signals have much less high-frequency noise due to low-pass filtering.

* The fitted isosbestic overlaps very nicely with the experimental signal except for some peak transients.

* The processed signal is free of photobleaching attenuation and movement artifacts.

* The transient spikes in the processed signal have a relatively stable amplitudes over time, while the magnitude of noise increases. This is because photobleaching attenuation decreases the signal-to-noise ratio.

Once ``.preprocess_signal()`` is run, the ``PhotometryExperiment`` object automatically populates its ``.metadata`` with reference fitting and correction type information.


```python
# view metadata
exp.metadata
```




    {'source': 'data/experiments/example_experiment.csv',
     'subject': 'animal_1',
     'sex': 'male',
     'age': 'young',
     'reference_fit': {'type': 'isosbestic',
      'r2_val': 0.9993991834603729,
      'coeffs': array([0.02467583, 1.24905654])},
     'correction_method': 'dF/F'}



Let's now compare the ``dF/F`` correction to the ``dF`` correction.


```python
exp.preprocess_signal(
    cutoff_frequency=3,
    order=4,
    # same parameters as before but with dF
    correction_method='dF',
    fit_using='IRLS',
    maxiter=2000,
    c=2,
    artifact_detector=None,
    artifact_corrector=None,
)

exp.plot_dashboard(raw=False)
```




    
![png](Introduction_files/Introduction_25_0.png)
    



Now notice how:

* The noise is constant while the transient peaks are decreasing in amplitude.

* This is because the ``dF`` correction does not correct for the photobleaching attenuation of the signal.

## 2.2. Single-channel processing

The general scheme for processing a **single-channel** experiment is:

1. Apply a low-pass filter to both the experimental signal to filter out high frequency noise.

2. Fit a photobleaching curve to the experimental signal.
    * This package uses a negative biexponential equation to model photobleaching: $\quad \alpha_1 \exp(-t / \tau_1) + \alpha_2 \exp(-t / \tau_2) + c$

3. Apply an reference correction either ``dB/B`` or ``dB``.
    * ``dF/F`` $= (\text{Exp.} - \text{B}_\text{ fitted}) / \text{B}_\text{ fitted}$
    * ``dF`` $= (\text{Exp.} - \text{B}_\text{ fitted})$

4. Detect and correct artifacts.
    * Since a reference channel isosbestic curve that shares artifacts is not present manual artifact detection and correction is required.

We will begin by loading our example experiment like in 2.1, but we will not load the isosbestic trace. Notice how our dashboard plot only contains the raw experimental signal now.


```python
# initialize the loader
loader = CSVLoader(
    csv = 'data/experiments/example_experiment.csv',
    time_col = 'time',
    signal_col = 'raw_signal',
    isosbestic_col = None,
    event_cols = ['trial_cue', 'lever1', 'lever2', 'shock']
)

# extract the data and load it into the PhotometryExperiment class
exp = loader.load()
exp.id = 'Single-channel Example'
exp.plot_dashboard()
```




    
![png](Introduction_files/Introduction_29_0.png)
    




```python
exp.preprocess_signal(
    # lowpass butterworth params
    cutoff_frequency=3,
    order=4,

    # correction method and isosbestic fit params
    correction_method='dB/B',
    fit_using='IRLS',
    maxiter=1000,
    c=2,
)
exp.plot_dashboard()
```




    
![png](Introduction_files/Introduction_30_0.png)
    



Our photobleaching curve fit very well. However, our processed signal still contains major artifacts - most prominently one large "jump" artifact.

To correct those artifacts, we need to setup our artifact handlers from the ``artifact`` module. The two most common types of artifacts are:

* Spike artifacts: sharp pings that readily return to baseline

* Jump artifacts: sharp changes that change the baseline value of signal for an extended period

Below we will use a detector the uses outlier derivative score-based detection and a corrector that fills artifacts with a spline interpolator.


```python
# import artifact handlers
from PhoPro.analysis.artifact import ODS_Detector, Spline_Corrector

# instantiate artifact detector and corrector
detector = ODS_Detector(
    score_threshold=5,
    jump_score_threshold=5,
    expand_sec=(0.5, 2),
    buffer_sec=1.5,
    n_chunks=20,
)

corrector = Spline_Corrector(
    anchor_sec=(0.2, 0.2),
    correct_spikes=True,
    correct_jumps=True,
)

exp.preprocess_signal(
    # lowpass butterworth params
    cutoff_frequency=3,
    order=4,

    # correction method and isosbestic fit params
    correction_method='dB/B',
    fit_using='IRLS',
    maxiter=1000,
    c=2,

    # pass in our artifact handlers
    artifact_detector=detector,
    artifact_corrector=corrector,
)
exp.plot_dashboard()
```




    
![png](Introduction_files/Introduction_32_0.png)
    



We can see the recovered signal with artifact correction has much less artifacting. Manual artifact correction is not perfect so using an isosbestic reference signal is preferred.

# 3. Trial Windowing

Once we have our processed, hopefully artifact-free, signal, we usually want to window the whole it into many smalled trials based on event or manual timestamps.

First, let's reload and process our example signal and inspect which events are present.


```python
# load
loader = CSVLoader(
    csv= 'data/experiments/example_experiment.csv',
    time_col= 'time',
    signal_col= 'raw_signal',
    isosbestic_col= 'raw_isosbestic',
    event_cols= ['trial_cue', 'lever1', 'lever2', 'shock'],
    annotation_file='data/experiments/example_annotation.json',
    annotation_handler='json',
)
exp = loader.load()
exp.id = 'Trial Extraction Example'

# process
exp.preprocess_signal(
    correction_method='dF/F',
    fit_using='IRLS',
)

# examine events present
exp.event_labels
```




    ['trial_cue', 'lever1', 'lever2', 'shock']



This simulated dataset models a risky-decision making behavioral experiment wherein:
*  A light turns on signifying the beginning of trial, represented by "trial_cue".

* Then the rat has between 2 and 4 seconds after the "trial_cue" to choose 1 of 2 levers that when pressed register as "lever1" and "lever2".

* "lever1" gives the rat a large food reward but there is a chance of the rat being shocked immediately after, represented by the "shock" event.

* "lever2" gives the rat a small food reward.

To slice the processed signal in ``PhotometryExperiment`` into a many events we use the method ``.extract_trial_data()``, which holds its result as a ``PhotometryData`` object in the ``.trial_data`` attribute.


```python
exp.extract_trial_data(
    # what event should we consider the "start" of a trial
    align_to='trial_cue',

    # what event(s) within a trial do we want to center on, i.e. be 0s
    # they should be mutually exclusive and if none are present in a trial it is centered to "align_to"
    center_on=['lever1', 'lever2'],

    # how long in seconds should our trials be relative to "center_on"
    trial_bounds=(-8, 8),

    # expected range of our events relative to "align_to"
    # if the value is None, the range defaults to "trial_bounds"
    event_tolerences={
        'lever1':(2, 4),
        'lever2':(2, 4),
        'shock':None,
    },

    # optional bounds for a "baseline" relative to the "align_to" event in seconds
    baseline_bounds=None,

    # which trial-wise normalization should be preformed
    # most normalization require a baseline bound to be present
    trial_normalization='none',

    # if multiple of the same event are within specified tolerences which should be picked
    event_conflict_logic='first',

    # should an error be thrown if multiple "center_on" event are present in the same trial and within tolerence
    check_overlap=True,

    # toggle whether events that are not in "event_tolerences" are passed down
    all_events=True,

    # the method for how windows are constructed
    window_alignment='interp'
)

trials = exp.trial_data
trials
```




    Photometry dataset with 20 trials, 1600 timepoints, and 7 observations.



We can see from the displayed ``PhotometryData``, that we extracted 20 trials, each with 999 timepoints.

# 4. Handling Trial-wise Data
This package stores trial-wise information in a ``PhotometryData`` object that uses [anndata](https://github.com/scverse/anndata) as the underlying data structure. This object can be written to and read from the .h5ad format and has many useful functions for data handling and analysis. It can also hold the trial-wise data from multiple experiments.

First let's save the data we extracted in 3. and load back in a clean file.


```python
trials.write_h5ad('output/example_trials.h5ad')
trials = PhotometryData.read_h5ad('data/trials/example_trials.h5ad')
trials
```




    Photometry dataset with 20 trials, 1598 timepoints, and 7 observations.



The main attributes of a ``PhotometryData`` object are:

* ``.obs`` a pandas DataFrame holding trial-wise metadata (i.e. subject, event timestamps)

* ``.X`` a 2D numpy array holding the processed signals of each trial (of shape n_trials, n_times)

* ``.ts`` a 1D numpy array of the time series

Let's examine these attributes below. Note that the event timestamp present in the experiment automatically populate ``.obs``, with ``NaN`` values corresponding to the event being abscent in that trial. 

Three other columns are also automatically populated by ``extract_trial_data()``:

* ``trial_num``: the order of trials found in the experiment.

* ``start_time``: the time within the experiment that the trial starts.

* ``stop_time``: the time within the experiment that the trial ends.


```python
trials.obs.head(5)
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
      <td>31.50581</td>
      <td>-3.52</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>65.23</td>
      <td>81.21581</td>
      <td>-2.71</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.22</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>113.05</td>
      <td>129.03581</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>167.51</td>
      <td>183.49581</td>
      <td>-3.94</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>217.89</td>
      <td>233.87581</td>
      <td>-3.79</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
display(
    trials.ts[:10],     # time series
    trials.X[:2, :10],  # trial signals
    trials.X.shape,     # of shape (n_trials, n_times)
)
```


    array([-8.       , -7.9899901, -7.9799802, -7.9699703, -7.9599604,
           -7.9499505, -7.9399406, -7.9299307, -7.9199208, -7.9099109])



    array([[0.00568005, 0.00563251, 0.00561191, 0.00562358, 0.00567088,
            0.00575452, 0.00587314, 0.00602279, 0.00619719, 0.00638825],
           [0.00108119, 0.00089587, 0.00072785, 0.00058132, 0.00045983,
            0.0003662 , 0.00030247, 0.00026987, 0.00026861, 0.0002981 ]])



    (20, 1598)


We will go ahead and drop the start and end time columns, as it is mainly diagnostic information and they won't be useful for us.


```python
trials.drop_obs_columns(['start_time', 'stop_time'])
trials.obs.head(5)
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
      <td>-3.52</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>-2.71</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.22</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>-3.94</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>-3.79</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Now let's see what we are working with by graphing the trials with ``.plot_trials()``.


```python
trials.plot_trials()
```




    
![png](Introduction_files/Introduction_49_0.png)
    



We can see that there are multiple distinct shapes of trials. Let's catagorize each trial by the events present. We can store this label in ``.obs``. Manipulating and analyzing data often requires the manipulation of ``.obs``.

* ``NoResponse`` corresponds to the subject failing to choose a lever.

* ``type1`` corresponds to the subject choosing the small reward lever ("lever2").

* ``type2`` corresponds to the subject choosing the large reward lever ("lever1") but NOT recieving a shock.

* ``type3`` corresponds to the subject choosing the large reward lever ("lever1") AND recieving a shock.


```python
# we can use a function to manipulate the .obs dataframe
def label_trials(df: pd.DataFrame) -> pd.DataFrame:
    # label trial outcome by event present
    # remeber a NaN value represents a missing event
    has_lever1 = ~df['lever1'].isna()
    has_lever2 = ~df['lever2'].isna()
    has_shock = ~df['shock'].isna()

    df['trial_label'] = pd.NA
    df.loc[has_lever2, 'trial_label'] = 'type1'
    df.loc[has_lever1 & ~has_shock, 'trial_label'] = 'type2'
    df.loc[has_lever1 & has_shock, 'trial_label'] = 'type3'
    df.loc[~has_lever1 & ~has_lever2, 'trial_label'] = 'NoResponse'
    return df

# note the new "trial_label" column
trials.obs = label_trials(trials.obs)
trials.obs.head(5)
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
      <th>trial_cue</th>
      <th>lever1</th>
      <th>lever2</th>
      <th>shock</th>
      <th>trial_label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>-3.52</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>type2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>-2.71</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.22</td>
      <td>type3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NoResponse</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>-3.94</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>type2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>-3.79</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>type2</td>
    </tr>
  </tbody>
</table>
</div>



Now let's plot the trials again but group them by "trial_label".


```python
trials.plot_trials(group_on='trial_label')
```




    
![png](Introduction_files/Introduction_53_0.png)
    



We can clearly see that the different trial outcomes have different signal architectures. Also note, how the peaks after "trial_cue" are aligned in the "NoResponse" trial type, but spread out in the other trial types, this is because the "NoResponse" trials are aligned to "trial_cue" but the others are aligned to lever presses - meaning there are variable latencies between "trial_cue" and lever presses.

We can quantitativly compare the signal architectures by calculating summary metrics like area under the curve (AUC).


```python
# lets calculate the AUC between 0 and 4.9 seconds
#   but first take the absolute value of the signal before calculation
# this is to capture the notable negative deviation in "type3" trails
trials.obs['AUC_abs'] = trials.area_under_curve(
    centers=0.0,
    bounds=(0, 4.9),
    transformation=np.abs,
)

trials.obs.head(5)
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
      <th>trial_cue</th>
      <th>lever1</th>
      <th>lever2</th>
      <th>shock</th>
      <th>trial_label</th>
      <th>AUC_abs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>-3.52</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>type2</td>
      <td>0.074713</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>-2.71</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.22</td>
      <td>type3</td>
      <td>0.150258</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NoResponse</td>
      <td>0.030606</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>-3.94</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>type2</td>
      <td>0.083858</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>-3.79</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>type2</td>
      <td>0.087954</td>
    </tr>
  </tbody>
</table>
</div>




```python
# quick plot using pandas to compare differences in AUC_abs
trials.obs.plot(
    x='trial_label',
    y='AUC_abs',
    kind='scatter',
)
plt.show()
```


    
![png](Introduction_files/Introduction_56_0.png)
    


We can directly run an ANOVA with ``.ANOVA()`` to see if the differences are significant.


```python
trials.ANOVA(
    dependent_var='AUC_abs',
    between='trial_label',
)
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
      <th>Source</th>
      <th>SS</th>
      <th>DF</th>
      <th>MS</th>
      <th>F</th>
      <th>p_unc</th>
      <th>np2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>trial_label</td>
      <td>0.016438</td>
      <td>3</td>
      <td>0.005479</td>
      <td>51.521322</td>
      <td>1.916450e-08</td>
      <td>0.906194</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Within</td>
      <td>0.001702</td>
      <td>16</td>
      <td>0.000106</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



``PhotometryData`` also provides methods to filter out specific trials and average across groups.

We can pass a boolean mask or a sequence of numeric indexes to ``.filter_rows()`` to select the desired trials.

Using ``.collapse`` we can average across groups in ``.obs`` resulting in a new ``PhotometryData`` object.


```python
# filter out "NoResponse" trials
filt_trials = trials.filter_rows(
    trials.obs['trial_label'] != 'NoResponse',
)

# average across trial types
avg = filt_trials.collapse(
    # what .obs columns to group on
    group_on=['trial_label'],

    # what metrics other than mean to compute
    metrics={'std' : np.std},

    # what columns in .obs to also perform the aggregation on
    data_cols=['trial_cue', 'AUC_abs'],

    # what to call the count column
    count_col='n_trials',
)

avg.obs
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
      <th>trial_label</th>
      <th>n_trials</th>
      <th>trial_cue</th>
      <th>AUC_abs</th>
      <th>trial_cue_std</th>
      <th>AUC_abs_std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>type2</td>
      <td>6</td>
      <td>-3.358333</td>
      <td>0.080484</td>
      <td>0.644190</td>
      <td>0.006149</td>
    </tr>
    <tr>
      <th>1</th>
      <td>type3</td>
      <td>3</td>
      <td>-2.680000</td>
      <td>0.128538</td>
      <td>0.225536</td>
      <td>0.017163</td>
    </tr>
    <tr>
      <th>2</th>
      <td>type1</td>
      <td>6</td>
      <td>-3.030000</td>
      <td>0.063665</td>
      <td>0.514101</td>
      <td>0.009130</td>
    </tr>
  </tbody>
</table>
</div>




```python
trials.filter_rows(
    trials.obs['trial_label'].str.startswith('No')
)
```




    Photometry dataset with 5 trials, 1598 timepoints, and 7 observations.



The averaged object now only has 3 trials since we filtered out "NoResponse" trials prior to averaging. ``.X`` is now the average signal and the other metrics that were calculated, like "std", are stored in the ``.layers`` attribute of the underlying adata structure. We can access them with ``.get_layer()``.


```python
avg.get_layer('std')
```




    array([[0.00244562, 0.00257221, 0.00269665, ..., 0.00746858, 0.00713986,
            0.00677318],
           [0.00462332, 0.00531349, 0.00596617, ..., 0.0047484 , 0.00522709,
            0.00571096],
           [0.00528095, 0.00476425, 0.00446005, ..., 0.01031396, 0.01048537,
            0.01053889]], shape=(3, 1598))



Now we can plot the average signal for each trial type, with labels and error bars.


```python
avg.plot_trials(
    label_with=['trial_label', 'n_trials'],
    err_layer='std',
)
```




    
![png](Introduction_files/Introduction_65_0.png)
    



What if we want to refocus on the subject's response to "trial_cue"? We can do this with the ``.window()`` method. 


```python
recentered = trials.window(
    # you can pass in a scalar or a sequence of timestamps as centers
    # in this case we are using an event timestamp column in ".obs"
    centers = trials.obs['trial_cue'],
    bounds = (-3, 3),
    # which columns in .obs to also recenter
    event_cols=['trial_cue', 'lever1', 'lever2', 'shock']
)

recentered_avg = recentered.collapse(
    # passing None means all the trials will be averaged
    group_on=None
)

recentered_avg.plot_trials(err_layer='std')
```




    
![png](Introduction_files/Introduction_67_0.png)
    



With windowing we can easily get a clean, aligned view of the "trial_cue" peak across trials.

# 5. Bulk Processing

Processing experiments one at a time is all well and good, but often times we need to process dozens to hundreds of experiments. The ``PhotometryPipeline`` class makes batch processing easy.

Take a look in the "data/pipeline/" directory. There you can see two cases of file layouts:

* Case 1: has data (CSV) and annotation (JSON) files seperated in different folders.

* Case 2: has all data (CSV) and annotation (JSON) files in the same directory.

First we will tackle case 1. To use ``PhotometryPipeline`` first you have to initialize the object then run it with specific processing parameters. 


```python
from PhoPro import PhotometryPipeline, CSVLoader

# initialize a pipeline object
pipeline1 = PhotometryPipeline(
    # point it to files within the data directory
    data_directory='data/pipeline/case1',
    target_type='file',

    # need to specify which loader to use
    loader_cls=CSVLoader,

    # needs to be recursive to search within subdirectory
    recursive=True,
    pattern='experiment_*.csv'
)

# see what inputs are discovered
pipeline1.discover_inputs()
```




    [PosixPath('data/pipeline/case1/experiment_1/experiment_1.csv'),
     PosixPath('data/pipeline/case1/experiment_3/experiment_3.csv'),
     PosixPath('data/pipeline/case1/experiment_2/experiment_2.csv')]



Now that we have confirmed the inputs are correct, we can run the pipeline by passing in keyword arguments for our selected ``PhotometryLoader``, ``PhotometryExperiment.preprocess_signal()``, and ``PhotometryExperiment.extract_trial_data()``.


```python
from pathlib import Path

# .preprocess_signal() arguments
preprocess_kwargs = dict(
    correction_method = 'dF/F',
    fit_using = 'OLS',
)

# .extract_trial_data() arguments
trial_extraction_kwargs = dict(
    align_to='trial_cue',
    center_on=['lever1', 'lever2'],
    trial_bounds=(-8, 8),
    trial_normalization='none',
    check_overlap=True,
    all_events=True,
)

# loader arguments
# we can pass in loader kwargs as a function that takes in the csv file path
#   so the pipeline can find the annotation files based on the csv path
def case1_loader_kwargs(csv: Path):
    annotation_file = csv.parent / 'annotation.json'
    return dict(
        time_col = 'time',
        signal_col = 'raw_signal',
        isosbestic_col = 'raw_isosbestic',
        event_cols = ['trial_cue', 'lever1', 'lever2', 'shock'],
        annotation_file = annotation_file,
        annotation_handler = 'json',
    )

# run our pipeline
trials: PhotometryData = pipeline1.run(
    # pass in keyword arguments
    loader_kwargs = case1_loader_kwargs,
    preprocess_kwargs = preprocess_kwargs,
    trial_extraction_kwargs = trial_extraction_kwargs,

    # do not automatically save
    output_dir = None,

    # log to a file
    log_file = 'output/example_pipeline_case1.log',

    # passdown metadata loaded from the annotations to .obs
    passdown_metadata = ['source', 'subject', 'sex', 'age']
)

# view output
print(trials)
trials.obs.head(5)
```

    Photometry dataset with 60 trials, 320 timepoints, and 11 observations.





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
      <th>source</th>
      <th>subject</th>
      <th>sex</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>15.50</td>
      <td>31.45</td>
      <td>-3.50</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>data/pipeline/case1/experiment_1/experiment_1.csv</td>
      <td>animal_1</td>
      <td>male</td>
      <td>young</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>65.20</td>
      <td>81.15</td>
      <td>-2.70</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.25</td>
      <td>data/pipeline/case1/experiment_1/experiment_1.csv</td>
      <td>animal_1</td>
      <td>male</td>
      <td>young</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>113.00</td>
      <td>128.95</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>data/pipeline/case1/experiment_1/experiment_1.csv</td>
      <td>animal_1</td>
      <td>male</td>
      <td>young</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>167.50</td>
      <td>183.45</td>
      <td>-3.95</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>data/pipeline/case1/experiment_1/experiment_1.csv</td>
      <td>animal_1</td>
      <td>male</td>
      <td>young</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>217.85</td>
      <td>233.80</td>
      <td>-3.80</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>data/pipeline/case1/experiment_1/experiment_1.csv</td>
      <td>animal_1</td>
      <td>male</td>
      <td>young</td>
    </tr>
  </tbody>
</table>
</div>



Dealing with case 2 is very similar; however, we do not need to make it recursive and we need to change the loader keyword arguments.


```python
# initialize a pipeline object
pipeline2 = PhotometryPipeline(
    data_directory='data/pipeline/case2',
    target_type='file',
    loader_cls=CSVLoader,

    # does not need to be recursive
    recursive=False,
    pattern='experiment_*.csv'
)

# .preprocess_signal() arguments
preprocess_kwargs = dict(
    correction_method = 'dF/F',
    fit_using = 'OLS',
)

# .extract_trial_data() arguments
trial_extraction_kwargs = dict(
    align_to='trial_cue',
    center_on=['lever1', 'lever2'],
    trial_bounds=(-8, 8),
    trial_normalization='none',
    check_overlap=True,
    all_events=True,
)

# loader arguments
# we can shorten the loader_kwargs function with lambda notation
case2_loader_kwargs = lambda csv: dict(
    time_col = 'time',
    signal_col = 'raw_signal',
    isosbestic_col = 'raw_isosbestic',
    event_cols = ['trial_cue', 'lever1', 'lever2', 'shock'],
    annotation_file = csv.with_suffix('.json'),
    annotation_handler = 'json',
)


# run our pipeline
trials: PhotometryData = pipeline2.run(
    loader_kwargs = case2_loader_kwargs,
    preprocess_kwargs = preprocess_kwargs,
    trial_extraction_kwargs = trial_extraction_kwargs,
    output_dir = None,
    log_file = 'output/example_pipeline_case2.log',
    passdown_metadata = ['source', 'subject', 'sex', 'age']
)

# view output
print(trials)
trials.obs.head(5)
```

    Photometry dataset with 60 trials, 320 timepoints, and 11 observations.





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
      <th>source</th>
      <th>subject</th>
      <th>sex</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>15.50</td>
      <td>31.45</td>
      <td>-3.50</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>data/pipeline/case2/experiment_1.csv</td>
      <td>animal_1</td>
      <td>male</td>
      <td>young</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>65.20</td>
      <td>81.15</td>
      <td>-2.70</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.25</td>
      <td>data/pipeline/case2/experiment_1.csv</td>
      <td>animal_1</td>
      <td>male</td>
      <td>young</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>113.00</td>
      <td>128.95</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>data/pipeline/case2/experiment_1.csv</td>
      <td>animal_1</td>
      <td>male</td>
      <td>young</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>167.50</td>
      <td>183.45</td>
      <td>-3.95</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>data/pipeline/case2/experiment_1.csv</td>
      <td>animal_1</td>
      <td>male</td>
      <td>young</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>217.85</td>
      <td>233.80</td>
      <td>-3.80</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>data/pipeline/case2/experiment_1.csv</td>
      <td>animal_1</td>
      <td>male</td>
      <td>young</td>
    </tr>
  </tbody>
</table>
</div>



We can also put in custom operations and an unique identifier builder function into the pipeline.


```python
# set up identifier builder
# it needs to be a function that takes in a PhotometryExperiment and returns a string
def build_id(exp: PhotometryExperiment) -> str:
    experiment_num = exp.metadata['source'].split('_')[-1].split('.')[0]
    id = (
        f"exp{experiment_num}_{exp.metadata['subject']}"
    )
    return id

# a pre-signal processing operation
#   that will trim the first ten time points
def post_load_func(exp: PhotometryExperiment) -> None:
    exp.trim_times_by_index(start_idx=10, stop_idx=None)

# a post trial extraction operation
#   that will assign trial labels
def post_extraction_func(exp: PhotometryExperiment) -> None:
    
    df = exp.trial_data.obs

    has_lever1 = ~df['lever1'].isna()
    has_lever2 = ~df['lever2'].isna()
    has_shock = ~df['shock'].isna()

    df['trial_label'] = pd.NA
    df.loc[has_lever2, 'trial_label'] = 'type1'
    df.loc[has_lever1 & ~has_shock, 'trial_label'] = 'type2'
    df.loc[has_lever1 & has_shock, 'trial_label'] = 'type3'
    df.loc[~has_lever1 & ~has_lever2, 'trial_label'] = 'NoResponse'

    exp.trial_data.obs = df

# run our pipeline
trials: PhotometryData = pipeline1.run(
    loader_kwargs = case1_loader_kwargs,
    preprocess_kwargs = preprocess_kwargs,
    trial_extraction_kwargs = trial_extraction_kwargs,
    output_dir = None,
    log_file = 'output/example_pipeline_case3.log',
    passdown_metadata = ['source', 'subject', 'sex', 'age'],

    # pass in functions
    id_builder=build_id,
    post_load_operation=post_load_func,
    post_trial_extraction_operation=post_extraction_func,
)

# view output
print(trials)
trials.obs.head(5)
```

    Photometry dataset with 60 trials, 320 timepoints, and 13 observations.





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
      <th>trial_label</th>
      <th>experiment_id</th>
      <th>source</th>
      <th>subject</th>
      <th>sex</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>15.50</td>
      <td>31.45</td>
      <td>-3.50</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>type2</td>
      <td>exp1_animal_1</td>
      <td>data/pipeline/case1/experiment_1/experiment_1.csv</td>
      <td>animal_1</td>
      <td>male</td>
      <td>young</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>65.20</td>
      <td>81.15</td>
      <td>-2.70</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.25</td>
      <td>type3</td>
      <td>exp1_animal_1</td>
      <td>data/pipeline/case1/experiment_1/experiment_1.csv</td>
      <td>animal_1</td>
      <td>male</td>
      <td>young</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>113.00</td>
      <td>128.95</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NoResponse</td>
      <td>exp1_animal_1</td>
      <td>data/pipeline/case1/experiment_1/experiment_1.csv</td>
      <td>animal_1</td>
      <td>male</td>
      <td>young</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>167.50</td>
      <td>183.45</td>
      <td>-3.95</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>type2</td>
      <td>exp1_animal_1</td>
      <td>data/pipeline/case1/experiment_1/experiment_1.csv</td>
      <td>animal_1</td>
      <td>male</td>
      <td>young</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>217.85</td>
      <td>233.80</td>
      <td>-3.80</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>type2</td>
      <td>exp1_animal_1</td>
      <td>data/pipeline/case1/experiment_1/experiment_1.csv</td>
      <td>animal_1</td>
      <td>male</td>
      <td>young</td>
    </tr>
  </tbody>
</table>
</div>



Note the new column "experiment_id" and "trial_label", and the "Running custom operations..." lines in the pipeline log.

# Summary

``PhoPro`` provides a comprehensive toolkit for processing, handling, and analyzing fiber photometry data. More in depth tutorials are avaliable on loading data, processing experiments, handling trial-wise data, and simulating photometry data.
