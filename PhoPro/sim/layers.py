"""Composable layers used to render simulated photometry traces."""

from typing import Any, Callable, Protocol, TypeAlias, runtime_checkable
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from scipy.signal import butter, sosfiltfilt, resample_poly

########################
#region --- SAMPLERS ---
########################

@runtime_checkable
class ParameterSampler(Protocol):
    """Protocol for values sampled independently for each event onset."""

    def sample(
            self,
            rng: np.random.Generator,
            size: int,
            ) -> np.ndarray:
        ...

@dataclass(frozen=True)
class SamplerUniform:
    """Sample event parameters from a continuous uniform distribution."""

    low: float
    high: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.low) or not np.isfinite(self.high):
            raise ValueError('Uniform sampler bounds must be finite.')
        if self.low > self.high:
            raise ValueError(
                f'Uniform sampler low ({self.low}) must be <= high ({self.high}).'
            )

    def sample(
            self,
            rng: np.random.Generator,
            size: int,
            ) -> np.ndarray:
        return rng.uniform(self.low, self.high, size=size)

@dataclass(frozen=True)
class SamplerNormal:
    """Sample event parameters from a normal distribution."""

    mean: float
    std: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.mean) or not np.isfinite(self.std):
            raise ValueError('Normal sampler parameters must be finite.')
        if self.std < 0:
            raise ValueError(
                f'Normal sampler std must be non-negative, is {self.std}.'
            )

    def sample(
            self,
            rng: np.random.Generator,
            size: int,
            ) -> np.ndarray:
        return rng.normal(self.mean, self.std, size=size)

ParameterValue: TypeAlias = (
    float | ParameterSampler
)

#endregion

########################
#region --- TIMEBASE ---
########################
@dataclass
class TimeBase:
    """Time axis specification for simulated traces."""

    time_sec: float
    frequency: float
    start_time: float = 0.0

    def __post_init__(self) -> None:
        """Validate timebase parameters."""
        if self.time_sec <= 0:
            raise ValueError(f'time_sec must be positive, is {self.time_sec}')
        if self.frequency <= 0:
            raise ValueError(f'frequency must be positive, is {self.frequency}')

    def evenly_spaced_timestamps(self, n_events: int | None, buffer_left_sec: float | None, buffer_right_sec: float | None) -> np.ndarray:
        """Create evenly spaced timestamps within the rendered timebase.

        Parameters
        ----------
        n_events : int or None
            Number of timestamps to generate. ``None`` and ``0`` return an
            empty array.
        buffer_left_sec : float or None
            Minimum buffer from the first rendered time point.
        buffer_right_sec : float or None
            Minimum buffer from the last rendered time point.

        Returns
        -------
        np.ndarray
            Evenly spaced timestamps sampled from the rendered timebase.
        """
        if n_events in (None, 0, 0.0):
            return np.empty(0)

        time = self.render()
        lo_idx = 0 if buffer_left_sec is None else np.searchsorted(time, time[0] + buffer_left_sec, side='left')
        hi_idx = time.size - 1 if buffer_right_sec is None else np.searchsorted(time, time[-1] - buffer_right_sec, side='left')

        stamp_idxs = np.linspace(lo_idx, hi_idx, n_events, dtype=int)
        timestamps = time[stamp_idxs]
        return timestamps

    def render(self) -> np.ndarray:
        """Render the timebase.

        Returns
        -------
        np.ndarray
            One-dimensional array of sampled time points.
        """
        n_points = round(self.time_sec * self.frequency)
        return (np.arange(n_points) / self.frequency) + self.start_time

#endregion

##############################
#region --- PHOTOBLEACHING ---
##############################

@dataclass
class PhotobleachingLayer:
    """Bi-exponential photobleaching baseline layer."""

    alpha1: float
    tau1: float
    alpha2: float
    tau2: float
    B_floor: float
    scale: float | None = None
    offset: float | None = None

    def __post_init__(self) -> None:
        """Validate photobleaching parameters and apply scale and/or offset."""
        self._apply_scale_offset()

        if self.tau1 <= 0:
            raise ValueError(f'tau1 must be positive, is {self.tau1}')
        if self.tau2 <= 0:
            raise ValueError(f'tau1 must be positive, is {self.tau2}')
        if self.B_floor < 0:
            raise ValueError(f'B_floor must be >= 0, is {self.B_floor}')
        if self.alpha1 + self.alpha2 <= 0:
            raise ValueError(f'alpha1 + alpha2 must be positive, is {self.alpha1 + self.alpha2}')
        
    def _apply_scale_offset(self) -> None:
        """Apply scale and offset once."""
        if self.scale is not None:
            self.alpha1 *= self.scale
            self.alpha2 *= self.scale
            self.B_floor *= self.scale
        if self.offset is not None:
            self.B_floor += self.offset

    def render(self, time: np.ndarray) -> np.ndarray:
        """Render the photobleaching baseline.

        Parameters
        ----------
        time : np.ndarray
            Time points, in seconds.

        Returns
        -------
        np.ndarray
            Strictly positive baseline values.

        Raises
        ------
        ValueError
            If the rendered baseline contains non-positive values.
        """
        exp1 = self.alpha1 * np.exp(-time / self.tau1)
        exp2 = self.alpha2 * np.exp(-time / self.tau2)
        B = exp1 + exp2 + self.B_floor

        if np.any(B <= 0):
            raise ValueError("Photobleaching baseline B must be strictly positive.")
        return B

@dataclass
class PhotobleachingLayer_alt:
    """Alternative bi-exponential photobleaching baseline layer."""

    B0: float
    B_floor: float
    tau1: float
    tau2: float
    weight1: float

    def render(self, time: np.ndarray) -> np.ndarray:
        """Render the alternative photobleaching baseline.

        Parameters
        ----------
        time : np.ndarray
            Time points, in seconds.

        Returns
        -------
        np.ndarray
            Strictly positive baseline values.

        Raises
        ------
        ValueError
            If the rendered baseline contains non-positive values.
        """
        B = self.B_floor + (self.B0 - self.B_floor) * (
            self.weight1 * np.exp(-time / self.tau1)
            + (1.0 - self.weight1) * np.exp(-time / self.tau2)
        )
        if np.any(B <= 0):
            raise ValueError("Photobleaching baseline B must be strictly positive.")
        return B

#endregion

#####################
#region --- NOISE ---
#####################

@dataclass
class NoiseMultiplicativeLayer:
    """Multiplicative noise layer.
    
    Creates multiplicative noise with the variation given by:

        std(t) = Nm * sqrt(C_norm ^ p)

    Where C_norm is the normalized clean trace, Nm is the magnitude, 
    and p is the proportionality exponent. p = 1 gives Poisson-like shot noise.

        p = 0   : constant variance (additive noise)
        p = 1   : shot-noise variance (Poissonian-like, assuming large photon count)
        p = 2   : multiplicative/fractional-noise-like variance (dynamic-like)
        p < 1   : sublinear signal-dependent noise
        p > 1   : superlinear signal-dependent noise
    """

    magnitude: float | None
    exponent: float = 1

    def __post_init__(self) -> None:
        """Validate multiplicative noise parameters."""
        if (self.magnitude is not None) and (self.magnitude < 0):
            raise ValueError(f'magnitude must be >= 0, is {self.magnitude}')
        if (self.exponent < 0):
            raise ValueError(f'proportionality must be >= 0, is {self.exponent}')
        

    def render(self, clean_trace: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Render multiplicative noise for a clean trace.

        Parameters
        ----------
        clean_trace : np.ndarray
            Clean fluorescence trace used to set expected photon counts.
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        np.ndarray
            Shot-noise trace in fluorescence units.
        """
        if self.magnitude is None:
            return np.zeros_like(clean_trace)
        
        norm_trace = np.maximum( clean_trace / clean_trace[0], 0.0)
        std_t = self.magnitude * np.sqrt( np.power(norm_trace, self.exponent) )
        noise = rng.normal(0, scale=std_t, size=clean_trace.size)
        
        return noise

@dataclass
class NoiseGaussianLayer:
    """Additive Gaussian noise layer."""

    sigma: float | None

    def __post_init__(self) -> None:
        """Validate Gaussian-noise parameters."""
        if (self.sigma is not None) and (self.sigma < 0):
            raise ValueError(f'sigma must be >= 0, is {self.sigma}')

    def render(self, time: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Render Gaussian noise.

        Parameters
        ----------
        time : np.ndarray
            Time points used to determine output length.
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        np.ndarray
            Gaussian noise trace.
        """
        if self.sigma is None:
            return np.zeros_like(time)

        return rng.normal(0.0, scale=self.sigma, size=time.size)
    
@dataclass
class NeuralDynamicNoiseLayer:
    amplitude: float | None
    center: float | None
    dynamic_frequency: float | None

    def render(self, time: np.ndarray, frequency: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self.amplitude is None:
            return np.zeros_like(time)
        
        noise_timebase = TimeBase(time[-1] - time[0], self.dynamic_frequency, time[0])
        noise_time = noise_timebase.render()
        low_rate_noise = rng.normal(self.center, self.amplitude, size=noise_time.size)

        ratio = Fraction(frequency / self.dynamic_frequency).limit_denominator(10000)
        high_rate_noise = resample_poly(
            low_rate_noise,
            up=ratio.numerator,
            down=ratio.denominator,
        )
        return high_rate_noise[:time.size]

#endregion

#########################
#region --- ARTIFACTS ---
#########################

@dataclass
class MovementAttenuationLayer:
    """Low-frequency multiplicative movement attenuation layer."""

    attenuation: float | None = 0.5
    cutoff_hz: float = 0.1
    filter_order: int = 4
    margin_sec: float | None = None
    scale: float | None = None
    center: float | None = None

    def __post_init__(self) -> None:
        """Validate movement attenuation parameters."""
        if self.attenuation is not None:
            if (self.attenuation < 0) or (self.attenuation > 1):
                raise ValueError(f'attenuation must be between 0 and 1, is {self.attenuation}')

    def _attenuation_keevers(
            self,
            n_samples: int,
            frequency: float,
            rng: np.random.Generator
            ) -> np.ndarray:
        """Generate a low-pass random attenuation trace."""
        # add padding
        margin_sec = 10.0 / self.cutoff_hz if self.margin_sec is None else self.margin_sec
        margin_samples = int(np.ceil(margin_sec * frequency))
        extended_length = n_samples + 2 * margin_samples

        # generate noise and lowpass it
        raw = rng.uniform(0.0, 1.0, size=extended_length)
        sos = butter(self.filter_order, self.cutoff_hz, btype='lowpass', fs=frequency, output='sos')
        filtered = sosfiltfilt(sos, raw, padtype='odd')

        # remove padding and return
        filtered = filtered[margin_samples : margin_samples + n_samples]
        return 1.0 - self.attenuation * filtered

    def render(self, time: np.ndarray, frequency: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Render movement attenuation.

        Parameters
        ----------
        time : np.ndarray
            Time points used to determine output length.
        frequency : float
            Sampling frequency in Hz.
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        np.ndarray
            Multiplicative attenuation trace.
        """
        if self.attenuation in (None, 0):
            return np.ones_like(time)

        trace = self._attenuation_keevers(
            n_samples=time.size,
            frequency=frequency,
            rng=rng
        )

        if self.scale is not None:
            trace *= self.scale

        if self.center is not None:
            trace += self.center - np.mean(trace)

        return trace


@dataclass
class ArtifactSpikeLayer:
    """Multiplicative transient spike artifact layer."""

    n_spikes: int
    amplitude_range: tuple[float, float]
    tau_range: tuple[float, float]
    tail_cutoff: float = 1e-3 # relative cutoff

    def __post_init__(self) -> None:
        """Validate spike artifact parameters."""
        if self.n_spikes < 0:
            raise ValueError(f'n_spikes must be >= 0, is {self.n_spikes}')

    @staticmethod
    def _spike_artifact(
            frequency: float,
            amplitude: float,
            tau_sec: float,
            cutoff: float,
            ) -> np.ndarray:
        """Render one exponential spike artifact."""
        # find time at cutoff
        t_cutoff = -1 * tau_sec * np.log(cutoff)

        # build time base to point after cutoff
        timebase = TimeBase(time_sec=t_cutoff, frequency=frequency)
        time = timebase.render()

        # return spike
        return amplitude * np.exp( -1 * time / tau_sec )

    def _add_spike_at(
        self,
        trace: np.ndarray,
        spike: np.ndarray,
        start_idx: int,
        ) -> None:
        """Add one spike into a trace with boundary clipping."""
        start_idx = int(start_idx)
        stop_idx = start_idx + spike.size

        trace_start = max(start_idx, 0)
        trace_stop = min(stop_idx, trace.size)

        if trace_start >= trace_stop:
            return

        kernel_start = trace_start - start_idx
        kernel_stop = kernel_start + (trace_stop - trace_start)

        trace[trace_start:trace_stop] += spike[kernel_start:kernel_stop]

    def render(self, time: np.ndarray, frequency: float, rng: np.random.Generator) -> np.ndarray:
        """Render multiplicative spike artifacts.

        Parameters
        ----------
        time : np.ndarray
            Time points used to determine output length.
        frequency : float
            Sampling frequency in Hz.
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        np.ndarray
            Multiplicative spike artifact trace initialized around one.
        """
        # set trace = 1
        trace = np.ones_like(time)
        if self.n_spikes == 0: return trace

        # estimate max spike size
        max_spike_size = max(int(5 * max(self.tau_range) * frequency), 3)

        # select params
        onset_idxs = rng.choice(np.arange(0, trace.size - max_spike_size), size=self.n_spikes, replace=False)
        amplitudes = rng.uniform(*self.amplitude_range, size=self.n_spikes)
        taus = rng.uniform(*self.tau_range, size=self.n_spikes)

        # add spikes to trace and return
        for i in range(self.n_spikes):
            spike = self._spike_artifact(frequency, amplitudes[i], taus[i], self.tail_cutoff)
            self._add_spike_at(trace, spike, onset_idxs[i])
        return trace

@dataclass
class ArtifactJumpLayer:
    """Multiplicative step-like jump artifact layer."""

    n_jumps: int
    amplitude_range: tuple[float, float]
    duration_range: tuple[float, float]

    def __post_init__(self) -> None:
        """Validate jump artifact parameters."""
        if self.n_jumps < 0:
            raise ValueError(f'n_jumps must be >= 0, is {self.n_jumps}')

    def render(self, time: np.ndarray, frequency: float, rng: np.random.Generator) -> np.ndarray:
        """Render multiplicative jump artifacts.

        Parameters
        ----------
        time : np.ndarray
            Time points used to determine output length.
        frequency : float
            Sampling frequency in Hz.
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        np.ndarray
            Multiplicative jump artifact trace initialized around one.
        """
        # set trace = 1
        trace = np.ones_like(time)
        if self.n_jumps == 0: return trace

        # convert durations into idxs
        min_size = round(min(self.duration_range) * frequency)
        max_size = round(max(self.duration_range) * frequency)

        # select params
        onset_idxs = rng.choice(np.arange(0, trace.size - max_size), size=self.n_jumps, replace=False)
        amplitudes = rng.uniform(*self.amplitude_range, size=self.n_jumps)
        sizes = rng.choice(np.arange(min_size, max_size), size=self.n_jumps, replace=True)

        # add jumps to trace and return
        for i in range(self.n_jumps):
            onset = onset_idxs[i]
            offset = onset + sizes[i]
            trace[onset : offset] += amplitudes[i]
        return trace

#endregion

######################
#region --- EVENTS ---
######################

@dataclass
class EventSpec:
    """Specification for one simulated event type."""

    onsets: np.ndarray
    amplitude: ParameterValue
    kernel_func: Callable[..., np.ndarray]
    kernel_params: dict[str, ParameterValue] = field(default_factory=dict)

    @property
    def has_fixed_params(self) -> bool:
        """Whether all parameters have fixed scalar values."""
        fixed_amp = not isinstance(self.amplitude, ParameterSampler)
        fixed_other = not any(
            isinstance(param, ParameterSampler) for param in self.kernel_params.values()
        )
        return fixed_amp and fixed_other

    def get_onset_idxs(
            self,
            time: np.ndarray,
            ) -> np.ndarray:
        """Convert event onset timestamps to indexes in a rendered timebase."""
        onsets = np.asarray(self.onsets)
        if onsets.size == 0:
            return np.empty(0, dtype=int)

        too_low = onsets < time[0]
        too_high = onsets > time[-1]
        if np.any(too_low) or np.any(too_high):
            raise ValueError(
                f'Event onset out of bounds at times '
                f'({onsets[too_low | too_high]}).'
            )

        return np.searchsorted(time, onsets, side='left')

    def _sample_param(
            self,
            value: ParameterValue,
            size: int,
            rng: np.random.Generator,
            ) -> np.ndarray:
        """Resolve one fixed or sampled parameter to a one-dimensional array."""
        if isinstance(value, ParameterSampler):
            sampled = value.sample(rng, size)
        else:
            sampled = np.full(size, value, dtype=float)

        sampled = np.asarray(sampled, dtype=float)

        if sampled.shape != (size,):
            raise ValueError(
                f'Sampled parameter must have shape ({size},), got '
                f'{sampled.shape}.'
            )
        if not np.isfinite(sampled).all():
            raise ValueError('Sampled parameter contains non-finite values.')

        return sampled

    def sample_parameters(
            self,
            size: int,
            rng: np.random.Generator,
            ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Sample amplitude and kernel parameters for event occurrences."""
        amp = self._sample_param(self.amplitude, size, rng)
        params = {
            label : self._sample_param(value, size, rng)
            for label, value in self.kernel_params.items()
        }
        return amp, params

@dataclass
class EventLayer:
    """Layer that renders event kernels at labeled onset timestamps."""

    specs: dict[str, EventSpec] = field(default_factory=dict)
    tail_cutoff: float = 1e-4
    max_duration_sec: float = 1000.0
    relative_cutoff: bool = True

    # --- event handling ---
    @property
    def event_labels(self) -> list[str]:
        """Event labels currently stored in this layer."""
        return list(self.specs.keys())

    def add_event(
            self,
            label: str,
            onsets: np.ndarray,
            amplitude: ParameterValue,
            kernel_func: Callable[..., np.ndarray],
            kernel_params: dict[str, ParameterValue],
            ) -> None:
        """Add an event specification from individual fields.

        Parameters
        ----------
        label : str
            Event label.
        onsets : np.ndarray
            Event onset times, in seconds.
        amplitude : float or ParameterSampler
            Fixed amplitude or per-onset sampler passed to ``kernel_func``.
        kernel_func : Callable[..., np.ndarray]
            Kernel function evaluated for this event.
        kernel_params : dict[str, float or ParameterSampler]
            Fixed or per-onset keyword arguments passed to ``kernel_func``.
        """
        self.specs[label] = EventSpec(
            onsets=onsets,
            amplitude=amplitude,
            kernel_func=kernel_func,
            kernel_params=kernel_params
        )

    def add_event_from_spec(self, label: str, spec: EventSpec) -> None:
        """Add an event specification object.

        Parameters
        ----------
        label : str
            Event label.
        spec : EventSpec
            Event specification to store.
        """
        self.specs[label] = spec

    def remove_event(self, label: str) -> None:
        """Remove an event specification.

        Parameters
        ----------
        label : str
            Event label to remove.

        Raises
        ------
        KeyError
            If ``label`` is not present.
        """
        if label not in self.event_labels:
            raise KeyError(f'Event {label} not found.')
        self.specs.pop(label)

    def onsets_to_dict(self) -> dict[str, np.ndarray]:
        """Return event onsets as a dictionary.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping from event labels to onset arrays.
        """
        return {label : spec.onsets for label, spec in self.specs.items()}

    # --- export ---
    def timestamps_to_dict(self) -> dict[str, np.ndarray]:
        """Return event timestamps as a dictionary.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping from event labels to onset arrays.
        """
        return {k: spec.onsets for k, spec in self.specs.items()}

    # --- trace building ---
    def _get_finite_kernel(
            self,
            amplitude: float,
            kernel_func: Callable[..., np.ndarray],
            kernel_params: dict[str, Any],
            frequency: float,
            ) -> np.ndarray:
        """Render one finite event kernel."""
        # build time
        timebase = TimeBase(self.max_duration_sec, frequency, 0.0)
        time = timebase.render()

        # make kernel and find peak
        values = np.asarray(kernel_func(time, amplitude, **kernel_params))
        magnitude = np.abs(values)
        peak_idx = int(np.argmax(magnitude))
        peak = magnitude[peak_idx]

        if peak == 0: return np.zeros(1)

        # find when it goes below threshold
        threshold = (peak * self.tail_cutoff) if self.relative_cutoff else self.tail_cutoff
        above_cutoff = np.flatnonzero(magnitude[peak_idx:] > threshold)

        if above_cutoff.size == 0:
            stop_idx = peak_idx + 1
        else:
            last_above = peak_idx + above_cutoff[-1]
            stop_idx = min(last_above + 2, values.size)

        if stop_idx == values.size and magnitude[-1] > threshold:
            raise ValueError("Kernel did not reach tail_cutoff before max_duration_sec.")

        # trim to idx where kernel < threshold
        finite_kernel = values[:stop_idx].copy()
        finite_kernel[-1] = 0.0
        return finite_kernel

    def _add_kernel_to_trace(
            self,
            trace: np.ndarray,
            kernel: np.ndarray,
            start_idxs: np.ndarray,
            ) -> None:
        """Add a finite kernel at start indexes using overlap-safe accumulation."""
        start_idxs = np.atleast_1d(np.asarray(start_idxs, dtype=int))

        # check for no onsets
        if start_idxs.size == 0:
            return

        indexes = start_idxs[:, None] + np.arange(kernel.size)[None, :]
        values = np.broadcast_to(kernel, indexes.shape)

        valid = (indexes >= 0) & (indexes < trace.size)
        np.add.at(trace, indexes[valid], values[valid])


    def render(
            self, 
            time: np.ndarray, 
            frequency: float,
            rng: np.random.Generator | None = None,
            ) -> np.ndarray:
        """Render all event specifications into one event trace.

        Parameters
        ----------
        time : np.ndarray
            Time points for the output trace.
        frequency : float
            Sampling frequency in Hz.
        rng : numpy.random.Generator or None, default=None
            Random generator used for per-onset parameter sampling. A fresh
            generator is created when omitted.

        Returns
        -------
        np.ndarray
            Event trace containing all rendered kernels.

        Raises
        ------
        ValueError
            If an event onset is outside ``time`` or a kernel does not decay
            below ``tail_cutoff`` before ``max_duration_sec``.
        """
        # init trace
        trace = np.zeros_like(time)
        rng = np.random.default_rng() if rng is None else rng

        for spec in self.specs.values():
            # init onset idxs
            onset_idxs = spec.get_onset_idxs(time)

            # fixed param case
            if spec.has_fixed_params:
                # uses vectorized workflow
                kernel = self._get_finite_kernel(
                    amplitude=spec.amplitude,
                    kernel_func=spec.kernel_func,
                    kernel_params=spec.kernel_params,
                    frequency=frequency,
                )
                self._add_kernel_to_trace(trace, kernel, onset_idxs)

            # per-onset parameter distributions
            else:
                # Different parameter values require one kernel per onset.
                sampled_amps, sampled_params = spec.sample_parameters(
                    onset_idxs.size,
                    rng,
                )

                for i, onset_idx in enumerate(onset_idxs):
                    amp_i = sampled_amps[i]
                    params_i = {
                        name : values[i]
                        for name, values in sampled_params.items()
                    }

                    kernel = self._get_finite_kernel(
                        amplitude=amp_i,
                        kernel_func=spec.kernel_func,
                        kernel_params=params_i,
                        frequency=frequency
                    )

                    self._add_kernel_to_trace(trace, kernel, [onset_idx])

        return trace

#endregion
