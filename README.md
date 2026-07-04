# **PhoPro**

*A comprehensive toolkit for the processing, analysis, and simulation of fiber photometry data*

## Overview
This package provides extensive functionaly for processing, handling, analyzing, and bulk processing fiber photometry datasets while remaining highly customizable. The high level APIs only require basic programming experience. It is built around 4 cores modules:

* **[PhotometryLoader](https://ggolde.github.io/PhoPro/api/PhotometryLoader/)** - loads various photometry data formats.

* **[PhotometryExperiment](https://ggolde.github.io/PhoPro/api/PhotometryExperiment/)** - processes and windows photometry experiments.

* **[PhotometryData](https://ggolde.github.io/PhoPro/api/PhotometryData/)** - handles trial-wise signals and metadata with advanced filtering, averaging, windowing, and analysis functionality.

* **[PhotometryPipeline](https://ggolde.github.io/PhoPro/api/PhotometryPipeline/)** - handles highly-customizable bulk processing of photometry data.


With an additional module for rigorously simulating complex photometry data:

* **[SimulatedPhotometry](https://ggolde.github.io/PhoPro/sim/SimulatedPhotometry)** - simulates photometry traces with realistic photobleaching, movement artifacts, and custom event dynamics.

---

## [Documentation](https://ggolde.github.io/PhoPro/)

Documentation for this package can be found [here](https://ggolde.github.io/PhoPro/). It is built with mkdocs-material and hosted on GitHub pages.

---

## Installation
You can now install through PyPi:
```
pip install PhoPro
```

Or you can install directly from GitHub using the command:
```
pip install git+https://github.com/ggolde/PhoPro.git
```

See the [Installation](https://ggolde.github.io/PhoPro/installation/) page of the docs for more details.

---

## Planned Features
This package is in ongoing development, please suggest any features you would like to see implemented on the [issues page](https://github.com/ggolde/PhoPro/issues). Below are a few planned features:
* [ ] Implement import support for more formats.
* [ ] More methods for artifact detection and correction.
* [x] Easier APIs for the cluster based permutation test and functional mixed modeling.
* [x] Phase out "variants" modules.
* [x] Improve windowing in PhotometryData.

## License

MIT License.

## Authors
This package is developed and used internally at the [Bizon-Setlow Lab](https://burke.neuroscience.ufl.edu/profile/bizon-jennifer/) at the University of Florida.

**Griffin Golde**: University of Florida, contact: **[ggolde@ufl.edu](mailto:ggolde@ufl.edu)**

## Citations

If you use this package in your work, and want to support the package, use the citation below (a proper publication is hopefully on the horizon):

* Golde G. *PhoPro: Python toolkit for simulating, processing, handling, and analyzing fiber photometry data*.
GitHub repository. Version 0.5.5. Available at: https://github.com/ggolde/PhoPro.

Additionally, if you used the FMM module, please review the [recommended reference(s)](https://github.com/gloewing/photometry_FLMM/blob/main/README.md#references) for the ``fast-fmm-rpy2`` package.