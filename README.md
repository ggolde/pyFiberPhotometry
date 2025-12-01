# **pyFiberPhotometry**

**Python package for processing TDT fiber photometry data**

`pyFiberPhotometry` provides a framework for loading, preprocessing, annotating, and analyzing behavior-coupled fiber photometry datasets collected using Tucker-Davis Technologies (TDT) acquisition systems.
 Built around two core classes that are subclassed for specific pipeline implementations.
 PhotometryData holds trial-wise data in an AnnData format.
 PhotometryExperiment extracts and processes signals and event timestamps from TDT folders, yeilding PhotometryData.
 Used internally in the **[Bizon-Setlow lab](https://neuroscience.ufl.edu/profile/bizon-jennifer/)** at the University of Florida.

---

## **Installation**

Install from GitHub:

```bash
pip install git+https://github.com/ggolde/pyFiberPhotometry.git
```

---

## **Quick Start**

### **Load and preprocess a TDT fiber photometry session**

```python
from pyFiberPhotometry import RDT_PhotometryExperiment

exp = RDT_PhotometryExperiment(
    data_folder="path/to/TDT/block",
    box="A",
)

exp.run_pipeline(
    downsample=10,
    preprocess_method="dF/F",
)

trials = exp.trials
baselines = exp.baselines
```

### **Plot a trial**

```python
trials.plot_line(0, label_with=["trial_label", "block"])
```

---

## **License**

MIT License.

---

## **Author**

**Griffin Golde**: University of Florida,
contact: **[ggolde@ufl.edu](mailto:ggolde@ufl.edu)**