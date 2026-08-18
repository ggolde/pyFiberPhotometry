# Batch Processing

This tutorial gives an in depth overview of the ``PhotometryPipeline`` class. ``PhotometryPipeline`` runs the same loader, preprocessing, and trial-extraction workflow over many recordings and returns one combined ``PhotometryData`` object.

The pipeline is useful when you have a large batch of experiments that share processing and trial-windowing parameters. It handles:

* discovering input files or folders,

* constructing loaders,

* running ``PhotometryExperiment.preprocess_signal()``,

* running ``PhotometryExperiment.extract_trial_data()``,

* passing metadata into trial-level ``obs`` columns,

* combining trial data across experiments,

* saving logs, trial-data files, and optional dashboards,

* running custom functions at key points in the workflow.

# Setup


```python
from pathlib import Path

import numpy as np
import pandas as pd

from PhoPro import CSVLoader, PhotometryData, PhotometryExperiment, PhotometryPipeline
```

This tutorial goes over two batch layouts:

* ``data/pipeline/case1``: each experiment has its own folder containing a CSV file and an ``annotation.json`` file holding per-experiment metadata: subject ID, age, and sex.

* ``data/pipeline/case2``: all CSV files and their matching JSON annotations live in one directory and share a basename.


```python
# set up reusable paths
CASE1_DIR = Path('data/pipeline/case1')
CASE2_DIR = Path('data/pipeline/case2')
EVENT_COLS = ['trial_cue', 'lever1', 'lever2', 'shock']

print(CASE1_DIR)
print(CASE2_DIR)
```

    data/pipeline/case1
    data/pipeline/case2


We will use the same preprocessing and trial extraction settings in several examples.


```python
preprocess_kwargs = dict(
    cutoff_frequency=3,
    order=4,
    correction_method='dF/F',
    fit_using='OLS',
)

trial_extraction_kwargs = dict(
    align_to='trial_cue',
    center_on=['lever1', 'lever2'],
    trial_bounds=(-8, 8),
    event_tolerences={
        'lever1': (2, 4),
        'lever2': (2, 4),
        'shock': None,
    },
    trial_normalization='none',
    check_overlap=True,
    all_events=True,
    window_alignment='nearest',
)
```

# 1. Creating a Pipeline

A pipeline is configured with a data directory, a target kind, and a loader class. ``target_type='file'`` means each experiments data is held in a single file (like CSVs) while ``target_type='folder'`` means experiment data is held in folders (like the TDT storage format).


```python
pipeline1 = PhotometryPipeline(
    data_directory=CASE1_DIR,
    target_type='file',
    loader_cls=CSVLoader,
    recursive=True,
    pattern='experiment_*.csv',
)
```

``discover_inputs`` shows which inputs will become jobs. In case 1, the CSV files are inside experiment subdirectories, so ``recursive=True`` is needed.


```python
pipeline1.discover_inputs()
```




    [PosixPath('data/pipeline/case1/experiment_1/experiment_1.csv'),
     PosixPath('data/pipeline/case1/experiment_3/experiment_3.csv'),
     PosixPath('data/pipeline/case1/experiment_2/experiment_2.csv')]



Case 2 is flat, so the pipeline can search only the top-level directory and not in nested folders for target files.


```python
pipeline2 = PhotometryPipeline(
    data_directory=CASE2_DIR,
    target_type='file',
    loader_cls=CSVLoader,
    recursive=False,
    pattern='experiment_*.csv',
)

pipeline2.discover_inputs()
```




    [PosixPath('data/pipeline/case2/experiment_1.csv'),
     PosixPath('data/pipeline/case2/experiment_2.csv'),
     PosixPath('data/pipeline/case2/experiment_3.csv')]



# 2. Loader Keyword Arguments

``loader_kwargs`` are passed to the loader constructor for each discovered input. The discovered path itself is supplied positionally (as the first argument) by the pipeline, so ``loader_kwargs`` should contain the rest of the loader configuration.


```python
static_loader_kwargs = dict(
    time_col='time',
    signal_col='raw_signal',
    isosbestic_col='raw_isosbestic',
    event_cols=EVENT_COLS,
)

static_loader_kwargs
```




    {'time_col': 'time',
     'signal_col': 'raw_signal',
     'isosbestic_col': 'raw_isosbestic',
     'event_cols': ['trial_cue', 'lever1', 'lever2', 'shock']}



A static dictionary is enough when every input uses the same loader settings and no per-file annotation path is needed.


```python
static_trials = pipeline2.run(
    loader_kwargs=static_loader_kwargs,
    preprocess_kwargs=preprocess_kwargs,
    trial_extraction_kwargs=trial_extraction_kwargs,
    output_dir=None,
    log_file='output/pipeline_static_loader.log',
    passdown_metadata=['source'],
)

print(static_trials)
static_trials.obs.head()
```

    Photometry dataset with 60 trials, 320 timepoints, and 6 observations.





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
      <th>source</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>-3.50</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>data/pipeline/case2/experiment_1.csv</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>-2.70</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.25</td>
      <td>data/pipeline/case2/experiment_1.csv</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>data/pipeline/case2/experiment_1.csv</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>-3.95</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>data/pipeline/case2/experiment_1.csv</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>-3.80</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>data/pipeline/case2/experiment_1.csv</td>
    </tr>
  </tbody>
</table>
</div>



For loading paired annotation files, ``loader_kwargs`` can be a function. The function receives the discovered ``Path`` and returns the kwargs for that specific input.


```python
def case1_loader_kwargs(csv: Path) -> dict:
    return dict(
        time_col='time',
        signal_col='raw_signal',
        isosbestic_col='raw_isosbestic',
        event_cols=EVENT_COLS,
        annotation_file=csv.parent / 'annotation.json',
        annotation_handler='json',
    )

case1_loader_kwargs(pipeline1.discover_inputs()[0])
```




    {'time_col': 'time',
     'signal_col': 'raw_signal',
     'isosbestic_col': 'raw_isosbestic',
     'event_cols': ['trial_cue', 'lever1', 'lever2', 'shock'],
     'annotation_file': PosixPath('data/pipeline/case1/experiment_1/annotation.json'),
     'annotation_handler': 'json'}



That path-aware function lets the pipeline derive the correct annotation file for each CSV in the nested folder layout.


```python
case1_trials = pipeline1.run(
    loader_kwargs=case1_loader_kwargs,
    preprocess_kwargs=preprocess_kwargs,
    trial_extraction_kwargs=trial_extraction_kwargs,
    output_dir=None,
    log_file='output/pipeline_case1.log',
    passdown_metadata=['source', 'subject', 'sex', 'age'],
)

print(case1_trials)
case1_trials.obs.head()
```

    Photometry dataset with 60 trials, 320 timepoints, and 9 observations.





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



For the single folder case, the resolver can use ``Path.with_suffix``.


```python
def case2_loader_kwargs(csv: Path) -> dict:
    return dict(
        time_col='time',
        signal_col='raw_signal',
        isosbestic_col='raw_isosbestic',
        event_cols=EVENT_COLS,
        annotation_file=csv.with_suffix('.json'),
        annotation_handler='json',
    )

case2_trials: PhotometryData = pipeline2.run(
    loader_kwargs=case2_loader_kwargs,
    preprocess_kwargs=preprocess_kwargs,
    trial_extraction_kwargs=trial_extraction_kwargs,
    output_dir=None,
    log_file='output/pipeline_case2.log',
    passdown_metadata=['source', 'subject', 'sex', 'age'],
)

print(case2_trials)
case2_trials.obs.head()
```

    Photometry dataset with 60 trials, 320 timepoints, and 9 observations.





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



# 3. Multiple Jobs per Input

``loader_kwargs`` may also be a list of dictionaries. A list creates multiple jobs for each discovered input. This is useful when the same file should be loaded in more than one way. For example, loading multiple boxes from a TDT vault folder or multiple exerimental wavelengths from one CSV. 

Below is an example of loading a single experiment twice.


```python
single_file_pipeline = PhotometryPipeline(
    data_directory=CASE2_DIR,
    target_type='file',
    loader_cls=CSVLoader,
    recursive=False,
    # restrict to only experiment_1.csv
    pattern='experiment_1.csv',
)

loader_fanout = [
    dict(
        time_col='time',
        signal_col='raw_signal',
        isosbestic_col='raw_isosbestic',
        event_cols=['trial_cue', 'lever1', 'lever2'],
        annotation_file=CASE2_DIR / 'experiment_1.json',
        annotation_handler='json',
    ),
    dict(
        time_col='time',
        signal_col='raw_signal',
        isosbestic_col='raw_isosbestic',
        event_cols=EVENT_COLS,
        annotation_file=CASE2_DIR / 'experiment_1.json',
        annotation_handler='json',
    ),
]

jobs = single_file_pipeline._build_jobs(
    inputs=single_file_pipeline.discover_inputs(),
    loader_kwargs=loader_fanout,
    preprocess_kwargs=preprocess_kwargs,
    trial_extraction_kwargs={**trial_extraction_kwargs, 'event_tolerences': {'lever1': (2, 4), 'lever2': (2, 4)}},
)

len(jobs)
```




    2



A callable resolver may also return a list, giving per-input fan-out while still deriving paths from each input.


```python
def fanout_loader_kwargs(csv: Path) -> list[dict]:
    base = dict(
        time_col='time',
        signal_col='raw_signal',
        isosbestic_col='raw_isosbestic',
        annotation_file=csv.with_suffix('.json'),
        annotation_handler='json',
    )
    return [
        base | {'event_cols': ['trial_cue', 'lever1', 'lever2']},
        base | {'event_cols': EVENT_COLS},
    ]

fanout_trials = single_file_pipeline.run(
    loader_kwargs=fanout_loader_kwargs,
    preprocess_kwargs=preprocess_kwargs,
    trial_extraction_kwargs={**trial_extraction_kwargs, 'event_tolerences': {'lever1': (2, 4), 'lever2': (2, 4)}},
    output_dir=None,
    log_file='output/pipeline_fanout.log',
    passdown_metadata=['source', 'subject'],
)

print(fanout_trials)
fanout_trials.obs.head()
```

    Photometry dataset with 40 trials, 320 timepoints, and 6 observations.





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
      <th>source</th>
      <th>subject</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>-3.50</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>data/pipeline/case2/experiment_1.csv</td>
      <td>animal_1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>-2.70</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>data/pipeline/case2/experiment_1.csv</td>
      <td>animal_1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>data/pipeline/case2/experiment_1.csv</td>
      <td>animal_1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>-3.95</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>data/pipeline/case2/experiment_1.csv</td>
      <td>animal_1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>-3.80</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>data/pipeline/case2/experiment_1.csv</td>
      <td>animal_1</td>
    </tr>
  </tbody>
</table>
</div>



# 4. Validation

The pipeline validates static preprocessing, extraction, dashboard, and resolved loader kwargs before running jobs. This catches misspelled keyword arguments early.


```python
try:
    pipeline2.run(
        loader_kwargs=static_loader_kwargs | {'not_a_loader_argument': True},
        preprocess_kwargs=preprocess_kwargs,
        trial_extraction_kwargs=trial_extraction_kwargs,
        output_dir=None,
    )
except TypeError as err:
    print(err)
```

    Unexpected kwargs for CSVLoader.__init__(): ['not_a_loader_argument']


Callable ``loader_kwargs`` are validated after concrete jobs are built, because the pipeline cannot know what a resolver will return until it sees an input path.


```python
def bad_loader_kwargs(csv: Path) -> dict:
    return {'time_col': 'time', 'not_a_loader_argument': True}

try:
    pipeline2.run(
        loader_kwargs=bad_loader_kwargs,
        preprocess_kwargs=preprocess_kwargs,
        trial_extraction_kwargs=trial_extraction_kwargs,
        output_dir=None,
    )
except TypeError as err:
    print(err)
```

    Unexpected kwargs for CSVLoader.__init__(): ['not_a_loader_argument']


# 5. Passing Metadata Down to Trials

``passdown_metadata`` copies selected experiment-level metadata keys into every trial row for that experiment. This is usually how subject, sex, age, session, and source information enter ``PhotometryData.obs``.


```python
case2_trials.obs[['subject', 'sex', 'age', 'source']].head()
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
      <th>subject</th>
      <th>sex</th>
      <th>age</th>
      <th>source</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>animal_1</td>
      <td>male</td>
      <td>young</td>
      <td>data/pipeline/case2/experiment_1.csv</td>
    </tr>
    <tr>
      <th>1</th>
      <td>animal_1</td>
      <td>male</td>
      <td>young</td>
      <td>data/pipeline/case2/experiment_1.csv</td>
    </tr>
    <tr>
      <th>2</th>
      <td>animal_1</td>
      <td>male</td>
      <td>young</td>
      <td>data/pipeline/case2/experiment_1.csv</td>
    </tr>
    <tr>
      <th>3</th>
      <td>animal_1</td>
      <td>male</td>
      <td>young</td>
      <td>data/pipeline/case2/experiment_1.csv</td>
    </tr>
    <tr>
      <th>4</th>
      <td>animal_1</td>
      <td>male</td>
      <td>young</td>
      <td>data/pipeline/case2/experiment_1.csv</td>
    </tr>
  </tbody>
</table>
</div>



Set ``passdown_metadata=None`` if you do not want any experiment metadata copied into trial rows.


```python
no_metadata_trials = single_file_pipeline.run(
    loader_kwargs=case2_loader_kwargs,
    preprocess_kwargs=preprocess_kwargs,
    trial_extraction_kwargs=trial_extraction_kwargs,
    output_dir=None,
    log_file='output/pipeline_no_metadata.log',
    passdown_metadata=None,
)

no_metadata_trials.obs.head()
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
      <td>-3.50</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>-2.70</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.25</td>
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
      <td>-3.95</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>-3.80</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



# 6. Experiment IDs and Custom Hooks

``run`` accepts custom functions at several points in the workflow:

* ``id_builder`` runs after trial extraction, assigns an experiment ID, and passes that ID into the ``trial_data.obs``.

* ``post_load_operation`` runs immediately after loading.

* ``post_preprocess_operation`` runs immediately after preprocessing.

* ``post_trial_extraction_operation`` runs immediately after trial extraction.

The ``id_builder`` should accept the experiment object as an arguement and return a string. Custom operations should accept the experiment as the only arguement, mutate the experiment in place, and return ``None``.


```python
def build_id(exp: PhotometryExperiment) -> str:
    source = Path(exp.metadata['source'])
    return f"{exp.metadata.get('subject', 'unknown')}_{source.stem}"


def trim_first_second(exp: PhotometryExperiment) -> None:
    exp.trim_times_by_values(lower=1.0)


def add_processed_summary(exp: PhotometryExperiment) -> None:
    exp.metadata['processed_signal_mean'] = float(np.nanmean(exp.signal))


def add_trial_labels(exp: PhotometryExperiment) -> None:
    obs = exp.trial_data.obs.copy()
    has_lever1 = obs['lever1'].notna()
    has_lever2 = obs['lever2'].notna()
    has_shock = obs['shock'].notna()

    obs['trial_label'] = 'no_response'
    obs.loc[has_lever2, 'trial_label'] = 'small_reward'
    obs.loc[has_lever1 & ~has_shock, 'trial_label'] = 'large_reward_safe'
    obs.loc[has_lever1 & has_shock, 'trial_label'] = 'large_reward_shock'
    exp.trial_data.obs = obs
```


```python
hook_trials: PhotometryData = pipeline1.run(
    loader_kwargs=case1_loader_kwargs,
    preprocess_kwargs=preprocess_kwargs,
    trial_extraction_kwargs=trial_extraction_kwargs,
    output_dir=None,
    log_file='output/pipeline_hooks.log',
    passdown_metadata=['source', 'subject', 'processed_signal_mean'],
    id_builder=build_id,
    post_load_operation=trim_first_second,
    post_preprocess_operation=add_processed_summary,
    post_trial_extraction_operation=add_trial_labels,
)

hook_trials.obs[['experiment_id', 'subject', 'processed_signal_mean', 'trial_label']].head()
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
      <th>experiment_id</th>
      <th>subject</th>
      <th>processed_signal_mean</th>
      <th>trial_label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>animal_1_experiment_1</td>
      <td>animal_1</td>
      <td>0.000039</td>
      <td>large_reward_safe</td>
    </tr>
    <tr>
      <th>1</th>
      <td>animal_1_experiment_1</td>
      <td>animal_1</td>
      <td>0.000039</td>
      <td>large_reward_shock</td>
    </tr>
    <tr>
      <th>2</th>
      <td>animal_1_experiment_1</td>
      <td>animal_1</td>
      <td>0.000039</td>
      <td>no_response</td>
    </tr>
    <tr>
      <th>3</th>
      <td>animal_1_experiment_1</td>
      <td>animal_1</td>
      <td>0.000039</td>
      <td>large_reward_safe</td>
    </tr>
    <tr>
      <th>4</th>
      <td>animal_1_experiment_1</td>
      <td>animal_1</td>
      <td>0.000039</td>
      <td>large_reward_safe</td>
    </tr>
  </tbody>
</table>
</div>



# 7. Writing Outputs

When ``output_dir`` is supplied, the final combined trial data are written to ``output_dir / trial_output_file``. If ``log_file`` is supplied, pipeline progress and job errors are written to that file.


```python
output_trials = single_file_pipeline.run(
    loader_kwargs=case2_loader_kwargs,
    preprocess_kwargs=preprocess_kwargs,
    trial_extraction_kwargs=trial_extraction_kwargs,
    output_dir='output/pipeline_output_demo',
    log_file='output/pipeline_output_demo.log',
    trial_output_file='combined_trials.h5ad',
    passdown_metadata=['source', 'subject', 'sex', 'age'],
)

print(output_trials)
print(Path('output/pipeline_output_demo/combined_trials.h5ad').exists())
print(Path('output/pipeline_output_demo.log').exists())
```

    ... storing 'source' as categorical
    ... storing 'subject' as categorical
    ... storing 'sex' as categorical
    ... storing 'age' as categorical


    Photometry dataset with 20 trials, 320 timepoints, and 9 observations.
    True
    True


The saved result can be loaded back as a normal ``PhotometryData`` object.


```python
loaded_output = PhotometryData.read_h5ad('output/pipeline_output_demo/combined_trials.h5ad')
loaded_output
```




    Photometry dataset with 20 trials, 320 timepoints, and 9 observations.



# 8. Saving Dashboards

Set ``save_dashboards=True`` to save one processed dashboard per experiment. The dashboards are written into ``output_dir / 'dashboards'``. Dashboard keyword arguments are passed to ``PhotometryExperiment.plot_dashboard``.


```python
dashboard_trials = single_file_pipeline.run(
    loader_kwargs=case2_loader_kwargs,
    preprocess_kwargs=preprocess_kwargs,
    trial_extraction_kwargs=trial_extraction_kwargs,
    output_dir='output/pipeline_dashboard_demo',
    log_file='output/pipeline_dashboard_demo.log',
    trial_output_file='dashboard_trials.h5ad',
    passdown_metadata=['source', 'subject'],
    save_dashboards=True,
    dashboard_kwargs={'raw': False, 'downsample': 30},
    id_builder=build_id,
)

sorted(Path('output/pipeline_dashboard_demo/dashboards').glob('*.svg'))
```

    ... storing 'experiment_id' as categorical
    ... storing 'source' as categorical
    ... storing 'subject' as categorical





    [PosixPath('output/pipeline_dashboard_demo/dashboards/animal_1_experiment_1.svg')]



# 9. Low-memory Mode

By default, the pipeline retains each experiment's ``PhotometryData`` in memory and concatenates all results once at the end. ``low_memory_mode=True`` writes each experiment's trials to a temporary ``.h5ad`` shard, concatenates all shards on disk once after the jobs finish, then loads the final result.

Low-memory mode requires a fresh output file. This is intentional, because appending into a pre-existing file could accidentally mix old and new runs.


```python
low_memory_output_dir = Path('output') / f"pipeline_low_memory_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S_%f')}"

low_memory_trials = pipeline2.run(
    loader_kwargs=case2_loader_kwargs,
    preprocess_kwargs=preprocess_kwargs,
    trial_extraction_kwargs=trial_extraction_kwargs,
    output_dir=low_memory_output_dir,
    log_file='output/pipeline_low_memory.log',
    trial_output_file='trials.h5ad',
    low_memory_mode=True,
    passdown_metadata=['source', 'subject', 'sex', 'age'],
)

print(low_memory_trials)
print((low_memory_output_dir / 'trials.h5ad').exists())
```

    ... storing 'source' as categorical


    ... storing 'subject' as categorical
    ... storing 'sex' as categorical
    ... storing 'age' as categorical
    ... storing 'source' as categorical
    ... storing 'subject' as categorical
    ... storing 'sex' as categorical
    ... storing 'age' as categorical
    ... storing 'source' as categorical
    ... storing 'subject' as categorical
    ... storing 'sex' as categorical
    ... storing 'age' as categorical


    Photometry dataset with 60 trials, 320 timepoints, and 9 observations.
    True


# 10. Error Handling and Logs

The pipeline logs errors for individual jobs and continues to later jobs. If every job fails, finalization raises because there is no trial data to return. For configuration problems, validation usually raises before any job starts.


```python
with open('output/pipeline_case2.log', 'r') as f:
    log_preview = ''.join(f.readlines()[:12])

print(log_preview)
```

    INFO:PhoPro.core.PhotometryPipeline:Beginning pipeline
    INFO:PhoPro.core.PhotometryPipeline:Validating static inputs...
    INFO:PhoPro.core.PhotometryPipeline:Discovering inputs...
    INFO:PhoPro.core.PhotometryPipeline:Building jobs...
    INFO:PhoPro.core.PhotometryPipeline:Validating loader_kwargs per-job...
    INFO:PhoPro.core.PhotometryPipeline:Iterating over 3 jobs...
    
    INFO:PhoPro.core.PhotometryPipeline:Processing job data/pipeline/case2/experiment_1.csv (1/3)
    INFO:PhoPro.core.PhotometryPipeline:Loading expriment...
    INFO:PhoPro.core.PhotometryPipeline:Preprocessing signal...
    INFO:PhoPro.core.PhotometryPipeline:Extracting trial data...
    INFO:PhoPro.core.PhotometryPipeline:Passing down experiment metadata as columns...
    


# 11. Choosing Pipeline Settings

A good pipeline setup separates three kinds of decisions:

* Loader kwargs describe how to read the raw file or folder.

* Preprocess kwargs describe how to turn raw continuous traces into processed signals.

* Trial extraction kwargs describe how to slice the processed signal into trials.

Keeping those dictionaries separate makes it easier to reuse the same preprocessing with different trial definitions, or the same trial definition across different file layouts.


```python
def run_case2_with_trial_bounds(bounds: tuple[float, float]) -> PhotometryData:
    extraction = trial_extraction_kwargs | {'trial_bounds': bounds}
    out: PhotometryData = pipeline2.run(
        loader_kwargs=case2_loader_kwargs,
        preprocess_kwargs=preprocess_kwargs,
        trial_extraction_kwargs=extraction,
        output_dir=None,
        log_file=f"output/pipeline_bounds_{bounds[0]}_{bounds[1]}.log",
        passdown_metadata=['source', 'subject'],
    )
    return out

short_trials = run_case2_with_trial_bounds((-4, 4))
short_trials
```




    Photometry dataset with 60 trials, 160 timepoints, and 7 observations.



# Summary

``PhotometryPipeline`` is designed for reproducible batch workflows. The main pattern is:

1. Configure a pipeline with a data directory, target type, loader class, recursion setting, and glob pattern.

2. Use ``discover_inputs`` to confirm the files or folders that will be processed.

3. Provide ``loader_kwargs`` as a static dictionary, a list of dictionaries, or a path-aware function.

4. Provide ``preprocess_kwargs`` and ``trial_extraction_kwargs`` for the experiment processing workflow.

5. Use ``passdown_metadata``, hooks, logs, output files, dashboards, and low-memory mode as needed.

The result is a single ``PhotometryData`` object containing trial data from every successful job.

# AI Use Disclaimer

Generative AI was used to assist in the creation of this tutorial. I plan to replace it in the future with a more polished version.
