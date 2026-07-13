# PhoPro: object-oriented photometry processing in Python

*A comprehensive toolkit for the processing, analysis, and simulation of fiber photometry data ([GitHub](https://github.com/ggolde/PhoPro)).*

## Overview
This package provides extensive functionaly for processing, handling, analyzing, and bulk processing fiber photometry datasets while remaining highly customizable. The high level APIs only require basic programming experience. It is built around 4 cores modules:

* **[PhotometryLoader](PhoPro/core/PhotometryLoader.md)** - loads various photometry data formats.

* **[PhotometryExperiment](PhoPro/core/PhotometryExperiment.md)** - processes and windows photometry experiments.

* **[PhotometryData](PhoPro/core/PhotometryData.md)** - handles trial-wise signals and metadata with advanced filtering, averaging, windowing, and analysis functionality.

* **[PhotometryPipeline](PhoPro/core/PhotometryPipeline.md)** - handles highly-customizable bulk processing of photometry data.


With an additional module for rigorously simulating complex photometry data:

* **[SimulatedPhotometry](PhoPro/sim/SimulatedPhotometry.md)** - simulates photometry traces with realistic photobleaching, movement artifacts, and custom event dynamics.

## Installation
You can install directly from GitHub using the command:
```
pip install git+https://github.com/ggolde/PhoPro.git
```

See the [Installation](installation.md) page for more details.

## Tutorials
The currently avaliable tutorials are:

* [Introduction](tutorials/Introduction.md) - explains the basic usage of the packages core features.

* [Simulating Photometry](tutorials/Simulating%20Photometry.md) - provides a detailed explanation of the components of fiber photometry signals and how to simulate them with the package.

* [Data Handling and Analysis](tutorials/Data%20Handling%20and%20Analysis.md) - detailed walkthrough of the ``PhotometryData`` class and its functionality (WIP).

* [Signal Processing](tutorials/Signal%20Processing.md) - detailed walkthrough of the ``PhotometryExperiment`` class (WIP).

* [Batch Processing](tutorials/Batch%20Processing.md) - detailed guide on how to use the ``PhotometryPipeline`` class for bulk processing (WIP).

They are also avaliable as interactive Jupyter Notebooks [here](https://github.com/ggolde/PhoPro/tree/main/tutorials). Some tutorials are still a work-in-progress and will be updated.

## License

MIT License.

## Authors
This package is developed and used internally at the [Bizon-Setlow Lab](https://burke.neuroscience.ufl.edu/profile/bizon-jennifer/) at the University of Florida.

**Griffin Golde**: University of Florida, contact: **[ggolde@ufl.edu](mailto:ggolde@ufl.edu)**

## Citations

If you use this package in your work, and want to support the package, use the citation below (a proper publication is hopefully on the horizon):

* Golde G. *PhoPro: a comprehensive toolkit for the processing, analysis, and simulation of fiber photometry data*.
GitHub repository. Version 0.6.0. Available at: https://github.com/ggolde/PhoPro.

Additionally, if you used the ``analysis.FMM`` module, please review the [recommended reference(s)](https://github.com/gloewing/photometry_FLMM/blob/main/README.md#references) for the ``fast-fmm-rpy2`` package.

## AI Use Disclaimer

Generative AI tools were used in support of developing this package mainly by:

* Generating and updating docstrings
* Helping prototype new features
* Debugging and integration
* Optimization of coputation heavy operations
* Assisting in the creation some tutorials