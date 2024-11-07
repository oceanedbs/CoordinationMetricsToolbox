# CoordinationMetricsToolbox
# Coordination Metrics Toolbox

The Coordination Metrics Toolbox is a comprehensive suite of tools designed to measure and analyze coordination metrics in various projects.

## Installation

To install the toolbox, clone the repository and install the dependencies:

```bash
git clone https://github.com/yourusername/CoordinationMetricsToolbox.git
cd CoordinationMetricsToolbox
pip install -r requirements.txt
```

# Informations

This work comes from the article : Dubois, O., Roby-Brami, A., Parry, R. et al. A guide to inter-joint coordination characterization for discrete movements: a comparative study. J NeuroEngineering Rehabil 20, 132 (2023). https://doi.org/10.1186/s12984-023-01252-2

Not all metrics have been implemented since some of them needs additional informations suche as joints position over time. 

2 more metrics have been added : Joints contribution variation based on Principal Component Analysis (JcvPCA) and Joint Synchronization Variation based on Continuous Relative Phase (JsvCRP). These metrics are under publication for PLOS-One

# Tutorial
An exemple with simulated data is available in test_coord_metric.py. 
CSV files containing a time column and joint angles trajectories can also be loaded to build a CoodinationMetric object form which different metrics can be computed. 

The format expected for the data is the following one : 
| time | joint_i | joint_j | joint_k | _ee_x_ | _ee_y_ | _ee_z_ |
|------|---------|---------|---------|--------|--------|--------|
|      |         |         |         |        |        |        |
|      |         |         |         |        |        |        |
|      |         |         |         |        |        |        |

Columns in italic (ee_x, ee_y, ee_z) which are the position of the end-effector are optional. However, without thoses columns, the number of metrics that can be computed is limited.

The file generate_testing_dataset.py can also be used to generate different datasets based on sinusoids. You can personalize this file to generate your own datasets. 

