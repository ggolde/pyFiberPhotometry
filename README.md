# **pyFiberPhotometry**

**Python package for processing TDT fiber photometry data**

`pyFiberPhotometry` provides a framework for loading, preprocessing, annotating, and analyzing behavior-coupled fiber photometry datasets collected using Tucker-Davis Technologies (TDT) acquisition systems. Mainly for use internally at the Bizon-Setlow lab.

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

trials = exp.trials     # trial-aligned signals
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
Contact: **[ggolde@ufl.edu](mailto:ggolde@ufl.edu)**