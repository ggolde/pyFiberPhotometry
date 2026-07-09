"""Simulation tools for synthetic fiber photometry recordings."""

from .SimulatedPhotometry import SimulatedPhotometry
from .SimulatedLibrary import SimulatedLibrary

from .layers import (
    TimeBase,
    PhotobleachingLayer,
    EventLayer, EventSpec,
    NoiseMultiplicativeLayer, NoiseGaussianLayer,
    MovementAttenuationLayer,
    ArtifactSpikeLayer, ArtifactJumpLayer,
)

__all__ = [
    "SimulatedPhotometry",
    "SimulatedLibrary",
    "TimeBase",
    "PhotobleachingLayer",
    "EventLayer",
    "EventSpec",
    "NoiseMultiplicativeLayer",
    "NoiseGaussianLayer",
    "MovementAttenuationLayer",
    "ArtifactSpikeLayer",
    "ArtifactJumpLayer",
]
