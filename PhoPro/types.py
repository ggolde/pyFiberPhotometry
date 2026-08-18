from collections.abc import Callable, Mapping
from typing import Literal, TypeAlias, Hashable
import numpy as np
from numpy.typing import ArrayLike

from scipy.signal import resample_poly

#####################
#region --- UTILS ---
#####################

DownsampleMethods: TypeAlias = (
    Literal[
        'resample',
        'mean',
    ]
)
'''
Methods for time series downsampling.

- ``resample``: uses ``scipy.signal.resample_poly`` to perform polyphase filtering
- ``mean``: downsamples using mean pooling
'''

#endregion

##################################
#region --- EXP SIG PROCESSING ---
##################################


ChannelMode: TypeAlias = (
    Literal[
        'auto',
        'dual',
        'single'
    ]
)
'''
The workflow used to prepreprocess the signal.

- ``auto``: auto detect channel mode from the prescence or abscense of the isosbestic
- ``dual``: process the signal with isosbestic fitting and correction
- ``single``: process the experimental signal with photobleaching detrending
'''

IsoFitMethod: TypeAlias = (
    Literal[
        'OLS', 
        'IRLS',
    ]
    | Callable[
        [np.ndarray, np.ndarray],
        np.ndarray,
    ]
)
'''
The method used to fit the isosbestic to experimnetal signal in dual-channel processing.

- ``OLS``: ordinary least squares
- ``IRLS``: iteratively reweighted least squares (recommended default)
- A custom function that accepts ``(signal: np.ndarray, isosbestic: np.ndarray)`` 
  positionally and returns the fitted isosbestic signal as a ``np.ndarray``
'''

CorrectionMethod: TypeAlias = (
    Literal[
        'dF/F', 
        'dF', 
        'dB/B', 
        'dB', 
        'none',
    ]
    | Callable[
        [np.ndarray, np.ndarray],
        np.ndarray,
    ]
)
'''
How to use the fitted reference signal (the isosbestic for dual-channel and 
the fit photobleaching curve for single-channel) to correct the experimental signal.

For dual-channel experiments:
- ``dF/F``: ``(experiment - fit isosbestic) / fit isosbestic`` 
  (corrects for photobleaching attenuation)
- ``dF``: ``(experiment - fit isosbestic)``,
  (does NOT correct for photobleaching attenuation on its own)

For single-channel experiments:
- ``dB/B``: ``(experiment - fit photobleaching) / fit photobleaching``,
  (corrects for photobleaching attenuation)
- ``dB``: ``(experiment - fit photobleaching)``,
  (does NOT correct for photobleaching attenuation on its own)

For both:
- A custom function that accepts ``(signal: np.ndarray, fitted reference: np.ndarray)``
  positionally and returns the corrected signal as a ``np.ndarray``
- ``none``: no transformation based on the fitted reference signal is performed,
  used mainly for debugging.
'''

ExpNormMethod: TypeAlias = (
    Literal[
        'zscore',
        'nullZ',
        'none',
    ]
    | Callable[
        [np.ndarray],
        np.ndarray
    ]
)
'''
The method used for whole-experiment normalization.

- ``zscore``: the traditional Z-score
- ``nullZ``: division by the signals root-mean-square deviation from zero without centering
- ``none``: no whole-experiment normalization
'''

#endregion

#################################
#region --- EXP TRIAL EXTRACT ---
#################################

TrialNormMethod: TypeAlias = (
    Literal[
        'zscore', 
        'zero', 
        'mad', 
        'amp', 
        'none'
    ]
    | Callable[
        [np.ndarray, np.ndarray],
        np.ndarray,
    ]
)
'''
The method used for trial-wise normalization based on a specified baseline region.

- ``zscore``: the traditional Z-score using the standard deviation and mean
  of the baseline region
- ``zero``: centers trials by the mean of the baseline
- ``mad``: robust Z-score using the median absolute deviation and median
  of the baseline region
- ``amp``: scales the trials by their absolute maximum
- ``none``: no trial-wise normalization
- A custom function that accepts ``(signal: np.ndarray, baselines: np.ndarray)``
  and returns the normalized signals as a 2D ``np.ndarray`` of the same shape of 
  ``signal``
'''

WindowMethod: TypeAlias = (
    Literal[
        'nearest', 
        'interp'
    ]
)
'''
The method used to window time-series.

- ``nearest``: snaps events to the nearest sampled timepoint and slices
  windows based to the nearest time point fitting within the window
- ``interp``: exactly center windows to events and use liner interpolation to
  interpolate the signal at the exact time grid built around the center
'''

InvalidWindowPolicy: TypeAlias = (
    Literal[
        'drop', 
        'error'
    ]
)
'''
How to handle windows with bounds that extend outside of the time-series
being windowed.

- ``drop``: drops the invalid windows without raising an error
- ``error``: raises a ``ValueError`` if there are invalid windows

'''

EventSelectionLogic: TypeAlias = (
    Literal[
        'first',
        'last',
        'all',
        'mean',
    ]
)
'''
Rule used to select timestamp(s) if multiple timestamps for the same 
event label fall inside a trial annotation window. 

- ``first``: only selects the first occurence of the event
- ``last``: only selects the last occurence of the event
- ``all``: keep all occurrences, but relabels them, with the first occurrence
  mantaining the base label and subsequent ones being relabeled as 
  ``f'{base_label}_occurrence_{n}'`` 
- ``mean``: uses the average timestamp of the multiple occurences
'''

#endregion

##############################
#region --- PEAK DETECTION ---
##############################

PeakCenterMethod: TypeAlias = (
    Literal[
        'median',
        'mean',
        'zeros',
    ]
    | Callable[
        [np.ndarray, int],
        np.ndarray
    ]
)
'''
Method used to estimate baseline or rolling-window center.

-``median``: ``np.nanmedian``
-``mean``: ``np.nanmean``
-``zeros``: a full zero-valued baseline
- A custom function that accepts ``(baseline / window: np.ndarray, axis: int = 1)``
  and returns a ``np.ndarray`` of shape ``baseline.shape[0]``.
'''

PeakScaleMethod: TypeAlias = (
    Literal[
        'mad',
        'std',
        'ones',
    ]
    | Callable[
        [np.ndarray, int],
        np.ndarray
    ]
)
'''
Method used to estimate each trial's baseline or rolling-window scale.

- ``mad``: median absolute deviation
- ``std``: ``np.std``
- ``zeros``: a full one-valued scale
- A custom function that accepts ``(baseline / window: np.ndarray, axis: int = 1)``
  and returns a ``np.ndarray`` of shape ``baseline.shape[0]``.
'''

PeakDirection: TypeAlias = (
    Literal[
        'positive',
        'negative',
        'both'
    ]
)
'''
What direction of peaks should be detected.

- ``positive``: only detect peaks that are above the baseline
- ``negative``: only detect peaks that go below the baseline
- ``both``: detect both positive and negative peaks
'''

#endregion

####################
#region --- DATA ---
####################

GroupKey: TypeAlias = (
    Hashable
    | tuple[Hashable, ...]
)

#endregion

######################
#region --- LOADER ---
######################

EventDecoderCallable: TypeAlias = Callable[
    [Mapping[str, ArrayLike], np.ndarray | None],
    Mapping[str, ArrayLike],
]

EventEncoding: TypeAlias = Literal['binary', 'timestamp', 'long'] | EventDecoderCallable
'''
Encoding of raw event streams.

- ``binary``: true-false like values indicating whether the event
occured at the timestamp of an aligned columns containing sampling times
- ``timestamps``: timestamp values of when the event occured relative to
the experiments time series with columns serving as event names
- ``long``: like ``timestamps`` but in long format, where the time column serves
as the timestamps and a single event column contains event labels
- A custom function that accepts a mapping of event labels to raw arrays and an
optional shared time array, and returns an event-label to timestamps mapping.
'''

#endregion
