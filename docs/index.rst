.. CoordinationMetricsToolbox documentation master file, created by
   sphinx-quickstart on Wed Nov 13 09:21:42 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

CoordinationMetricsToolbox documentation
========================================

This coordination metrics toolbox is from the work of __Dubois, O., Roby-Brami, A., Parry, R. et al. A guide to inter-joint coordination characterization for discrete movements: a comparative study. J NeuroEngineering Rehabil 20, 132 (2023). https://doi.org/10.1186/s12984-023-01252-2__
Not all metrics are implemented since some of them requires joints position trajectories. 
In addition, the metrics developped in the paper [add PLOS paper once published], have been added to quantify differences in inter-joint coordination.

To use this toolbox, joints angle trajectories are required in the following format :

The format expected for the data is the following one : 
| time | joint_i | joint_j | joint_k | _ee_x_ | _ee_y_ | _ee_z_ |
|------|---------|---------|---------|--------|--------|--------|
|      |         |         |         |        |        |        |
|      |         |         |         |        |        |        |
|      |         |         |         |        |        |        |

Columns in italic (ee_x, ee_y, ee_z) which are the position of the end-effector are optional. However, without thoses columns, the number of metrics that can be computed is limited.


.. toctree::
   :maxdepth: 3
   :caption: Contents:

   modules

