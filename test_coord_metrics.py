import numpy as np
import matplotlib.pyplot as plt
import coordination_metrics_toolbox as cm
import os
import random
import pandas as pd

if __name__ == "__main__":   

    """
    Main program to perform coordination metrics analysis on test data.

    This program reads files from test datasets, performs coordination metrics analysis,
    and optionally plots the data.
    """

    # Get all the files as a list of file names contained in test_dataset1
    test_dataset_path = "tests/test_dataset1"
    file_list = [os.path.join(test_dataset_path, f) for f in os.listdir(test_dataset_path) if os.path.isfile(os.path.join(test_dataset_path, f))]
    # Create coordination metrics object on test data
    m1 = cm.CoordinationMetrics(file_list, end_effector=True)

    # Get all the files as a list of file names contained in test_dataset2
    test_dataset_path = "tests/test_dataset2"
    file_list = [os.path.join(test_dataset_path, f) for f in os.listdir(test_dataset_path) if os.path.isfile(os.path.join(test_dataset_path, f))]
    # Perform coordination metrics analysis on test data 2
    m2 = cm.CoordinationMetrics(file_list, end_effector=True)


    # Plot the data
    m1.plot_joints_angles(-1)
    m1.plot_joints_angular_velocity(4)


    # Compute inter-joint coordination metrics
    res_crp = m1.compute_continuous_relative_phase(trial=4, plot=True)
    res_angle_angle = m1.compute_angle_angle_plot(trial=None)
    res_pca = m1.compute_principal_component_analysis(trial=-1, plot=True, n_components=1)
    res_crosscorr = m1.compute_cross_correlation(trial=-1, plot=True, normalize=True)
    res_ici = m1.compute_interjoint_coupling_interval(trial=-1, plot=True)
    dist_pca = m1.compute_distance_between_PCs(m2, plot=True)
    res_correlation = m1.compute_correlation(trial=2, plot=True, type='pearson')
    res_angles_ratio = m1.compute_angle_ratio(trial=None, plot=True)
    res_temporal_coord = m1.compute_temporal_coordination_index(trial=None, plot=True)
    res_zero_crossing = m1.compute_zero_crossing(trial=None, plot=True)
    res_dtw = m1.compute_dynamic_time_warping(trial= None, plot=True)
    res_jcvpca = m1.compute_jcvpca(m2, plot=True, n_pca=2)
    res_jsvcrp = m1.compute_jsvcrp(m2, plot=True)