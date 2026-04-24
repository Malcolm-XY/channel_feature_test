# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 19:07:53 2026

@author: usouu
"""

"""
---
By this description, give me a table or figure (python code), to show your imagination:

All classification experiments followed an experiment-dependent evaluation protocol. 
Specifically, for each experiment, training and testing were conducted using data from the same EEG recording, 
without mixing data across different experiments or subjects. For each recording, 
different combinations of connectivity methods and subnetwork extraction with varying NRRs were applied. 
Each combination was then evaluated independently using a 5-fold cross-validation scheme. Under this setting, 
classification performance was summarized by reporting the mean accuracy and standard deviation 
across the recordings for each method and NRR.

---
Yes. Based on your description, the most natural main figure is:

x-axis: NRR
y-axis: mean accuracy
one line per method
error bars or shaded band: standard deviation across recordings

That directly matches:

“mean accuracy and standard deviation across the recordings for each method and NRR”

Below is a clean Python example using placeholder data to show the intended structure.
"""

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------
# Example structure of summarized results
# rows: NRR levels
# cols: methods
# values: mean accuracy (%) across recordings
# -----------------------------------

nrrs = np.array([100, 50, 40, 30, 20, 10])

methods = ["PCC", "PLV", "Additive", "Multiplicative", "Splicing", "PG-AC"]

mean_acc = {
    "PCC":            np.array([82.1, 80.4, 79.6, 78.8, 76.9, 73.5]),
    "PLV":            np.array([83.4, 82.0, 81.2, 80.3, 78.7, 75.4]),
    "Additive":       np.array([83.0, 81.5, 80.8, 79.9, 77.8, 74.6]),
    "Multiplicative": np.array([83.8, 82.4, 81.7, 80.9, 79.3, 76.1]),
    "Splicing":       np.array([82.7, 81.2, 80.3, 79.1, 77.0, 73.8]),
    "PG-AC":          np.array([85.2, 84.6, 84.0, 83.1, 81.7, 79.8]),
}

std_acc = {
    "PCC":            np.array([2.1, 2.4, 2.3, 2.6, 2.8, 3.1]),
    "PLV":            np.array([2.0, 2.1, 2.2, 2.4, 2.6, 2.9]),
    "Additive":       np.array([2.2, 2.3, 2.4, 2.5, 2.7, 3.0]),
    "Multiplicative": np.array([2.0, 2.2, 2.1, 2.3, 2.5, 2.7]),
    "Splicing":       np.array([2.3, 2.5, 2.6, 2.7, 2.9, 3.2]),
    "PG-AC":          np.array([1.8, 1.9, 2.0, 2.1, 2.3, 2.5]),
}

# -----------------------------------
# Plot
# -----------------------------------
plt.figure(figsize=(9, 6))

for method in methods:
    y = mean_acc[method]
    s = std_acc[method]
    plt.plot(nrrs, y, marker='o', linewidth=2, label=method)
    plt.fill_between(nrrs, y - s, y + s, alpha=0.15)

plt.xlabel("Node Retention Rate (NRR, %)")
plt.ylabel("Accuracy (%)")
plt.title("Classification Performance Across Methods and NRRs")
plt.xticks(nrrs)
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(frameon=True)
plt.tight_layout()
plt.show()