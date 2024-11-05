# CoordinationMetricsToolbox
# Coordination Metrics Toolbox

The Coordination Metrics Toolbox is a comprehensive suite of tools designed to measure and analyze coordination metrics in various projects.

## Features

- **Metric Calculation**: Automatically calculate various coordination metrics.
- **Data Visualization**: Generate visual representations of coordination data.


## Installation

To install the toolbox, clone the repository and install the dependencies:

```bash
git clone https://github.com/yourusername/CoordinationMetricsToolbox.git
cd CoordinationMetricsToolbox
pip install -r requirements.txt
```

## Usage

An exemple with simulated data is available in test.py. 
CSV files containing a time column and joint angles trajectories can also be loaded to build a CoodinationMetric object form which different metrics can be computed. 

The file generate_testing_dataset.py generates different datasets based on sinusoids. You can personalize this file to generate your own datasets. 
