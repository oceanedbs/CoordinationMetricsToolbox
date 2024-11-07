"""
This toolbox computes different metrics of interjoint coordination using CSV files containing time and joint angle data. 
A list of files can be provided, with one file per trial. The toolbox computes the following metrics: 
- Continuous relative phase (CRP)
- Angle-angle plots
- Angle ratio
- Cross-correlation
- Dynamic time warping
- FADA
- Interjoint Coupling Interval (ICI)
- PCA
- PCA distance
- JcvPCA
- JsvCRP
- RJAC
- different statistical tests for inter-joint coordination
- Time Zero Crossing
- Temporal coordination index
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import math
from fastdtw import fastdtw
#from .utils import generate_palette


def generate_palette(n):
    """
    Generate a palette with n colors using the viridis colormap.
    
    Parameters:
    n (int): Number of colors to generate.
    
    Returns:
    list: List of colors in RGB format.
    """
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i / n) for i in range(n)]
    return colors


class CoordinationMetrics():

    def __init__(self, list_files_angles, list_name_angles=None, name=None, end_effector=False,  deg=True, freq=None):
        """
        Initialize the CoordinationMetricsToolbox.
        Parameters:
        list_files_angles (list): List of file paths containing angle data.
        list_name_angles (list, optional): List of names corresponding to the angles. Defaults to None. If no names are passed, it is supposed that the first line containes the time + the name of the joints angles.
        name (str, optional): Name of the dataset instance. Will be printed as a header of plots. Defaults to None.
        end_effector (bool, optional): Flag indicating if the data contains end-effector data. Defaults to False. If end effector data are contained it is supposed that there are 3 columns : x, y, z.
        deg (bool, optional): Flag indicating if the angles are in degrees. Defaults to True.
        freq (float, optional): Frequency of the data. If None is passed, the sampling frequency will be computed based on the time column. Defaults to None.
        """

        self.list_files_angles = list_files_angles
        self.list_name_angles = list_name_angles
        if name is not None:
            self.name = name
        else:
            self.name = "Dataset"
        self.end_effector = end_effector
        self.deg = deg
        self.freq = freq

        self.load_csv_files()

        self.set_angle_names()
        self.set_velocities_names() 
        self.set_angles_combinations()

        self.rename_time_column()

        self.set_n_dof()

        if not deg:
            self.convert_angles_to_radians()

        if self.end_effector:
            self.rename_end_effector_columns()  
            self.compute_end_effector_velocity()
        

        self.compute_joints_angular_velocity()  

        return None

#%% Load data and initialize fields correctly 

    def load_csv_files(self):
        """
        Loads CSV files from a specified directory.

        This method reads all CSV files in the self.list_files_angles and processes them
        into a usable format for further analysis.

        Returns:
            list: A list of dataframes, each containing the data from a CSV file.
        """        
        self.data_joints_angles = []

        for f in self.list_files_angles:
            if not os.path.exists(f):
                raise FileNotFoundError(f"File {f} not found.")
            if not f.endswith(".csv"):
                raise ValueError(f"File {f} is not a CSV file.")
            self.data_joints_angles.append(pd.read_csv(f, sep=",", header=[0]))



    def set_angle_names(self):
        """
        Sets the names of the angles if they are not already set.
        This method checks if the attribute `list_name_angles` is None. If it is,
        it assigns the names of the angles from the first row of the `data_joints_angles`
        DataFrame, excluding the first column that should be the time.
        Attributes:
            list_name_angles (list or None): A list to store the names of the angles.
            data_joints_angles (list of DataFrames): A list containing DataFrames with joint angle data.
        """

        if self.list_name_angles is None and not self.end_effector:
            self.list_name_angles = self.data_joints_angles[0].columns[1:]
        elif self.list_name_angles is None and self.end_effector: 
            self.list_name_angles= self.data_joints_angles[0].columns[1: -3]

    def set_velocities_names(self):
        """
        Sets the names of the velocities of the angles.
        This method sets the names of the velocities of the angles by appending "_velocity"
        to the names of the angles.
        Attributes:
            list_name_angles (list): A list containing the names of the angles.
            list_name_velocities (list): A list containing the names of the velocities of the angles.
        """

        self.list_name_velocities = [f"{angle}_velocity" for angle in self.list_name_angles]

    def set_angles_combinations(self):
        """
        Sets the combinations of angles for computing inter-joint coordination metrics.
        This method generates all possible combinations of angles from the list of angle names
        and stores them in the `angles_combinations` attribute.
        Attributes:
            list_name_angles (list): A list containing the names of the angles.
            angles_combinations (list): A list containing all possible combinations of angles.
        """

        self.angles_combinations = list(itertools.combinations(self.list_name_angles, 2))

    def rename_time_column(self):
        """
        Renames the first column of each DataFrame in the `data_joints_angles` attribute to "time".

        This method iterates over the list of DataFrames stored in the `data_joints_angles` attribute
        and renames the first column of each DataFrame to "time".

        Returns:
            None
        """

        for df in self.data_joints_angles:
            df.rename(columns={df.columns[0]: "time"}, inplace=True)

    def rename_end_effector_columns(self):
        """
        Renames the columns of the end-effector data.

        This method renames the columns of the end-effector data to "x", "y", and "z".

        Returns:
            None
        """

        for df in self.data_joints_angles:
            df.rename(columns={df.columns[-3]: "ee_x", df.columns[-2]: "ee_y", df.columns[-1]: "ee_z"}, inplace=True)

    def compute_end_effector_velocity(self):
        """
        Computes the velocity of the end-effector data.

        This method computes the velocity of the end-effector data by taking the derivative of the
        "ee_x", "ee_y", and "ee_z" columns. The velocity is stored in the "ee_x_velocity",
        "ee_y_velocity", and "ee_z_velocity" columns.

        Returns:
            None
        """

        for df in self.data_joints_angles:
            for col, vel_col in zip(["ee_x", "ee_y", "ee_z"], ["ee_x_velocity", "ee_y_velocity", "ee_z_velocity"]):
                df[vel_col] = df[col].diff()/df["time"].diff()
            df["ee_velocity"] = np.sqrt(df["ee_x_velocity"]**2 + df["ee_y_velocity"]**2 + df["ee_z_velocity"]**2)

    def set_n_dof(self):
        """
        Sets the number of degrees of freedom (n_dof) for the object.
        This method calculates the number of degrees of freedom by determining the length of the 
        list_name_angles attribute and assigns this value to the n_dof attribute.
        Attributes:
            n_dof (int): The number of degrees of freedom.
            list_name_angles (list): A list containing the names of the angles.
        """

        self.n_dof = len(self.list_name_angles)

    def convert_angles_to_radians(self):
        """
        Converts the joint angles from degrees to radians.
        This method converts the joint angles from degrees to radians by multiplying the values
        in the `data_joints_angles` attribute by the conversion factor pi/180.
        Attributes:
            data_joints_angles (list of DataFrames): A list containing DataFrames with joint angle data.
        """

        for df in self.data_joints_angles:
            for col in self.list_name_angles:
                df[col] = df[col] * (np.pi / 180)


#%% Compute angular velocity of the joints

    def compute_joints_angular_velocity(self):
        """
        Computes the angular velocity of the joints.
        This method computes the angular velocity of the joints by taking the derivative of the joint angles.
        The angular velocity is stored in the attribute `data_joints_angular_velocity`.
        Attributes:
            data_joints_angles (list of DataFrames): A list containing DataFrames with joint angle data.
            data_joints_angular_velocity (list of DataFrames): A list containing DataFrames with joint angular velocity data.
        """

        for df in self.data_joints_angles:
            for col, vel_col in zip(self.list_name_angles, self.list_name_velocities):
                df[vel_col] = df[col].diff()/df["time"].diff()


    #%% Plotting functions

    def plot_joints_angles(self, trial=None):
        """
        Plots the joint angles for the specified trial.
        Parameters:
        trial (int): The index of the trial to plot. If None, plots all trials. If -1  plots the mean of all trials.
        Raises:
        ValueError: If the trial index is out of range.
        Returns:
        None
        """

        if trial == None:
            data = self.data_joints_angles
            title = "All trials"
        elif trial >= len(self.data_joints_angles) or trial < -1:
            raise ValueError(f"Trial index {trial} out of range. Only {len(self.data_joints_angles)} trials available.")
        elif trial == -1:
            data = [self.get_mean_data()]
            title = "Mean of all trials"
        else : 
            data = [self.data_joints_angles[trial]]
            title = f"Trial {trial}"

        fig, ax = plt.subplots()
        c = ut.generate_palette(self.n_dof)
        #plot all trials
        for df in data:
            for i, angle in enumerate(self.list_name_angles):
                    df.plot(x="time", y=angle, ax=ax, color=c[i])
            ax.legend(self.list_name_angles)
            ax.set_xlabel("Time")
            if self.deg:
                ax.set_ylabel("Angle (degrees)")
            else:
                ax.set_ylabel("Angle (radians)")
            if self.name is not None:
                fig.suptitle(f"Joint angles for {self.name}")
            else:
                fig.suptitle("Joint angles \n"+title + '\n' + self.name)    
        plt.show()
    
    def plot_joints_angular_velocity(self, trial=None):    
        """
        Plots the joint angular velocities for the specified trial.
        Parameters:
        trial (int): The index of the trial to plot. If None, plots all trials. If -1  plots the mean of all trials.
        Raises:
        ValueError: If the trial index is out of range.
        Returns:
        None
        """
        if trial == None:
            data = self.data_joints_angles
            title = "All trials"
        elif trial >= len(self.data_joints_angles) or trial < -1:
            raise ValueError(f"Trial index {trial} out of range. Only {len(self.data_joints_angles)} trials available.")
        elif trial == -1:
            data = [self.get_mean_data()]
            title = "Mean of all trials"
        else:
            data = [self.data_joints_angles[trial]]
            title = f"Trial {trial}"

        fig, ax = plt.subplots()
        c = ut.generate_palette(self.n_dof)
        # plot all trials
        for df in data:
            for i, angle in enumerate(self.list_name_angles):
                df.plot(x="time", y=f"{angle}_velocity", ax=ax, color=c[i])
        ax.legend(self.list_name_angles)
        ax.set_xlabel("Time")
        if self.deg:
            ax.set_ylabel("Angular Velocity (degrees/s)")
        else:
            ax.set_ylabel("Angular Velocity (radians/s)")
      
        fig.suptitle("Joint angular velocities \n"+title + '\n' + self.name)
        plt.show()

   #%% Inter-joint coordination metrics

    def compute_continuous_relative_phase(self, trial=None, plot=False):
        """
        Computes the continuous relative phase (CRP) between pairs of joints.
        Parameters:
        trial (int): The index of the trial to compute the CRP for. None = Compute the overall CRP. If -1, uses the mean joints data
        plot (bool): Flag to indicate whether to plot the CRP. Default is False.
        Raises:
        ValueError: If the trial index is out of range.
        Returns:
        dict: A dataframe containing the CRP values for each pair of joints, one row per trial.
        """

        if trial == None: 
            data = self.data_joints_angles
            title = "All trials"
        elif trial >= len(self.data_joints_angles) or trial < -1:
            raise ValueError(f"Trial index {trial} out of range. Only {len(self.data_joints_angles)} trials available.")
        elif trial == -1:
            data = [self.get_mean_data()]
            title = "Mean of all trials"
        else:
            data = [self.data_joints_angles[trial]]
            title = f"Trial {trial}"    
        #compute the phase for each joint
        for d in data:
            for angle, angle_vel in zip(self.list_name_angles, self.list_name_velocities):
                d[f"{angle}_phase"] = np.arctan2(d[angle_vel],d[angle])

            for a1, a2 in self.angles_combinations:
                print(a1, a2)
                #create column and fill with NaN
                d['CRP_'+a1+'_'+a2] = np.NaN
                #compute relative phase, without considering the first row that is NaN
                d['CRP_'+a1+'_'+a2].iloc[1:] = np.unwrap(d[a1+'_phase'].iloc[1:] - d[a2+'_phase'].iloc[1:])

        #plot the CRP
        if plot :      
            for a1, a2 in self.angles_combinations:
                fig, ax = plt.subplots()
                for i, d in enumerate(data):
                    d.plot(x='time', y='CRP_'+a1+'_'+a2, ax=ax, label="Trial "+str(i))
                    ax.set_title('Continuous Relative Phase '+a1+'-'+a2 + '\n'+title + '\n' + self.name)
                    ax.set_xlabel('Time')
                    ax.set_ylabel('CRP (radians)')
                plt.show()
        return data


    def compute_angle_angle_plot(self, trial=None):
        """
        Generates an angle-angle plot for the specified trial.
        Parameters:
        trial (int, optional): The index of the trial to plot. If trial is None, 
                       concatenated data from all trials is used. 
                       If trial is -1, mean data from all trials is used. 
                       Defaults to None.
        Raises:
        ValueError: If the trial index is out of range.
        Returns:
        None: This function does not return any value. It displays a plot.
        """
        
        if trial == None:
            data = self.get_concatenate_data()
            title = "All trials"
        elif trial >= len(self.data_joints_angles) or trial < -1:
            raise ValueError(f"Trial index {trial} out of range. Only {len(self.data_joints_angles)} trials available.")
        elif trial==-1:
            data = self.get_mean_data()
            title = "Mean of all trials"
        else:
            data = self.data_joints_angles[trial]
            title = f"Trial {trial}"

        a = sns.pairplot(data, vars=self.list_name_angles, kind='scatter', corner=True, diag_kind='kde', plot_kws={'alpha':0.5})
        a.fig.suptitle("Angle-Angle plot \n"+title + '\n' + self.name)  
        plt.show()

    def compute_principal_component_analysis(self, trial=None, plot=False, n_components=2):
        """
        Computes the principal component analysis (PCA) for the joint angles.
        Parameters:
        trial (int): The index of the trial to compute the PCA for. Default is None and uses all the trials. If -1, uses the mean joints data
        plot (bool): Flag to indicate whether to plot the PCA. Default is False.
        Raises:
        ValueError: If the trial index is out of range.
        Returns:
        dict: A dataframe containing the PCA values for each pair of joints, one row per trial.
        """
        if trial == None: 
            data = self.get_concatenate_data()
            title = "All trials"
        elif trial >= len(self.data_joints_angles) or trial < -1:
            raise ValueError(f"Trial index {trial} out of range. Only {len(self.data_joints_angles)} trials available.")
        elif trial == -1:
            data = self.get_mean_data()
            title = "Mean of all trials"
        else:
            data = self.data_joints_angles[trial]
            title = f"Trial {trial}"
        
        #standardize the data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)

        #compute the PCA for each joint
        pca = PCA(n_components=n_components)
        pca.fit(data[self.list_name_angles])

        #plot the PCA
        if plot:      
            fig, ax = plt.subplots(n_components, 1)
            if n_components == 1:
                ax = [ax]   
            for n in range(n_components):
                ax[n].bar(self.list_name_angles, pca.components_[n])
                ax[n].set_title(f'Principal Component {n+1} \n'+title + '\n' + self.name)
            plt.show()
    
        return pca

    def compute_cross_correlation(self, trial=None, plot=False, normalize=False):
        """
        Computes the cross-correlation between pairs of joints.
        Parameters:
        trial (int): The index of the trial to compute the cross-correlation for. Default is None and uses all the data. If -1, uses the mean joints data
        plot (bool): Flag to indicate whether to plot the cross-correlation. Default is False.
        Raises:
        ValueError: If the trial index is out of range.
        Returns:
        dict: A dataframe containing the cross-correlation values for each pair of joints, one row per trial.
        """

        if trial == None: 
            data = self.get_data_joints_angles()
            title = "All trials"
        elif trial >= len(self.data_joints_angles) or trial < -1:
            raise ValueError(f"Trial index {trial} out of range. Only {len(self.data_joints_angles)} trials available.")
        elif trial == -1:
            data = [self.get_mean_data()]
            title = "Mean of all trials"
        else:
            data = [self.data_joints_angles[trial]]
            title = f"Trial {trial}"

        # Normalize the data
        if normalize:
            for d in data:
                for col in self.list_name_angles:
                    d[col] = (d[col] - d[col].mean()) / d[col].std()

        for a1, a2 in self.angles_combinations:
            for d in data:
                #create column and fill with NaN
                d['CrossCorr_'+a1+'_'+a2] = np.NaN
                #compute cross-correlation
                d['CrossCorr_'+a1+'_'+a2] = np.correlate(d[a1],d[a2], mode='same')
                d['CrossCorr_Lag']=np.arange(-len(d)//2,len(d)//2)
                
        if plot:
            for a1, a2 in self.angles_combinations:
                fig, ax = plt.subplots()
                for i, d in enumerate(data):
                    d.plot(x='CrossCorr_Lag', y='CrossCorr_'+a1+'_'+a2, ax=ax, label="Trial "+str(i))
                ax.set_title(f'Cross-correlation {a1}-{a2} \n'+title + '\n' + self.name)
                ax.set_xlabel('Lag Step')
                ax.set_ylabel('Cross-correlation')
                plt.show()

        return data
    
    def compute_interjoint_coupling_interval(self, trial=None, plot=False):
        """
        Computes the Interjoint Coupling Interval (ICI) between pairs of joints.
        Parameters:
        trial (int): The index of the trial to compute the ICI for. Default is None and uses all the data. If -1, uses the mean joints data
        plot (bool): Flag to indicate whether to plot the ICI. Default is False.
        Raises:
        ValueError: If the trial index is out of range.
        Returns:
        dict: A dataframe containing the ICI values for each pair of joints, one row per trial.
        """
        if trial == None: 
            data = self.get_data_joints_angles()
            title = "All trials"
        elif trial >= len(self.data_joints_angles) or trial < -1:
            raise ValueError(f"Trial index {trial} out of range. Only {len(self.data_joints_angles)} trials available.")
        elif trial == -1:
            data = [self.get_mean_data()]
            title = "Mean of all trials"
        else:
            data = [self.data_joints_angles[trial]]
            title = f"Trial {trial}"
        
        # Create an empty confusion matrix with the joint angles as columns and rows
        ici_results = pd.DataFrame(columns=['trial', 'joints', 'ICI'])
       
        print(ici_results.columns)
        for i,d in enumerate(data) :
            for a1, a2 in self.angles_combinations:
                #compute ICI
                end_of_movement1 = d[(d[a1+'_velocity'] < 0.05 * d[a1+'_velocity'].max())]
                end_of_movement2 = d[(d[a2+'_velocity'] < 0.05 * d[a2+'_velocity'].max())]

                # Find the intervals where the indices are consecutive and selet the last block for the end of the movement
                #Then select the first element of the last block as the deactivation time of the joint
                end_of_movement1 = np.split(end_of_movement1, np.where(np.diff(end_of_movement1.index) != 1)[0] + 1)[-1].head(1)
                end_of_movement2 = np.split(end_of_movement2, np.where(np.diff(end_of_movement2.index) != 1)[0] + 1)[-1].head(1)

                #Compute the ICI
                ici_results.loc[len(ici_results)] = ({'trial': i, 'joints': f'{a1}_{a2}', 'ICI': end_of_movement2['time'].values[0] - end_of_movement1['time'].values[0]})

                
            print(ici_results)

        if plot:
            fig, ax = plt.subplots()
            sns.barplot(ici_results, x='joints', y='ICI', ax=ax)
            ax.set_title(f'Interjoint Coupling Interval {a1}-{a2} \n'+title + '\n' + self.name)
            ax.set_xlabel('Time')
            ax.set_ylabel('ICI')
            plt.show()

        return ici_results
    
    def get_pca_subspace(self):
        """
        Return the subspace defined by the eigenvectors of the covariance matrix.

        Returns
        -------
        eigenvectors : TYPE
            DESCRIPTION.

        """
        data = self.get_concatenate_data()
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data[self.list_name_angles])
        pca = PCA()
        pca.fit(data_scaled)
        return pca.components_
    
    def compute_distance_between_PCs(self, cm2,trialA=None, trialB=None, plot=False):
        """
        Computes the distance between the principal components (PCs) of the joint angles.
        From Bockemühl Till, Troje NF, Dürr V. Inter‑joint coupling and joint angle synergies of human catching movements. Hum Mov Sci. 2010;29(1):73–93.
        https:// doi. org/ 10. 1016/j. humov. 2009. 03. 003.
        Distance between PCs is defined as dist(U, V ) = np.sqrt(1 − s^2), with s being the minimum singular value of the matrix W = min(U^T V ).
        Parameters:
        trial (int): The index of the trial to compute the distance between PCs for. Default is None and uses all the data. If -1, uses the mean joints data
        plot (bool): Flag to indicate whether to plot the distance between PCs. Default is False.
        Raises:
        ValueError: If the trial index is out of range.
        Returns:
        res_dist_pca: A dataframe containing the distance between PCs for each pair of joints, one row per trial.
        """

        #Compute PCA on both datasets that will define U and V
        res_dist_pca = pd.DataFrame(columns=['datasetA', 'datasetB', 'distance', 'angle'])
        subspaceA = self.get_pca_subspace() # U
        subspaceB = cm2.get_pca_subspace() # V

        print(subspaceA)
        print(subspaceB)

        # Compute U^T V
        W = np.dot(subspaceA.T, subspaceB)

        print(W)
        # Compute the singular values of W
        u, s, v = np.linalg.svd(np.dot(subspaceA, np.transpose(subspaceB)))
        print(s)
        #find the minimum singular value
        min_singular_value = np.min(s)
        # print(s_min)
        if min_singular_value > 1:
            min_singular_value = 1

        #comute the distance
        distance = np.real(np.sqrt(1 - min_singular_value**2))
        angle = math.asin(distance)*180/math.pi

        res_dist_pca.loc[len(res_dist_pca)] = {'datasetA': self.get_name(), 'datasetB': cm2.get_name(), 'distance': distance, 'angle': angle}
        
    def compute_correlation(self, trial=None, plot=False, type='pearson'):
        """
        Computes the correlation between pairs of joints.
        Parameters:
        trial (int): The index of the trial to compute the correlation for. Default is None and uses all the data. If -1, uses the mean joints data
        plot (bool): Flag to indicate whether to plot the correlation. Default is False.
        type (str): Type of correlation to compute. Default is 'pearson'. Other options are 'spearman' and 'kendall'.
        Raises:
        ValueError: If the trial index is out of range.
        Returns:
        dict: A dataframe containing the correlation values for each pair of joints, one row per trial.
        """
        if trial == None:
            data = self.get_data_joints_angles()
            title = "All trials"
        elif trial >= len(self.data_joints_angles) or trial < -1:
            raise ValueError(f"Trial index {trial} out of range. Only {len(self.data_joints_angles)} trials available.")
        elif trial == -1:
            data = [self.get_mean_data()]
            title = "Mean of all trials"
        else:
            data = [self.data_joints_angles[trial]]
            title = f"Trial {trial}"
        
        # Create an empty confusion matrix with the joint angles as columns and rows
        correlation_results = pd.DataFrame(columns=['trial', 'joints', 'correlation'])

        for i,d in enumerate(data) :
            for a1, a2 in self.angles_combinations:
                #compute correlation
                correlation_results.loc[len(correlation_results)] = ({'trial': i, 'joints': f'{a1}_{a2}', 'correlation': d[a1].corr(d[a2], method=type)})

        if plot :
            fig, ax = plt.subplots()
            sns.barplot(correlation_results, x='joints', y='correlation', ax=ax)
            ax.set_title(f'Correlation {a1}-{a2} \n'+title + '\n' + self.name)
            ax.set_xlabel('Time')
            ax.set_ylabel('Correlation')
            plt.show()

        return correlation_results
    
    def compute_angle_ratio(self, trial=None, plot=False):
        """
        Computes the angle ratio of pairs of joints at the maximum velocity of the end-effector time point.
        Parameters:
        trial (int): The index of the trial to compute the angle ratio for. Default is None and uses all the data. If -1, uses the mean joints data
        plot (bool): Flag to indicate whether to plot the angle ratio. Default is False.
        Raises:
        ValueError: If the trial index is out of range.
        Returns:
        dict: A dataframe containing the angle ratio values for each pair of joints, one row per trial.
        """
        if not self.end_effector:
            raise ValueError("This metric can only be computed if end-effector data is available.")
        if trial == None:
            data = self.get_data_joints_angles()
            title = "All trials"
        elif trial >= len(self.data_joints_angles) or trial < -1:
            raise ValueError(f"Trial index {trial} out of range. Only {len(self.data_joints_angles)} trials available.")
        elif trial == -1:
            data = [self.get_mean_data()]
            title = "Mean of all trials"
        else:
            data = [self.data_joints_angles[trial]]
            title = f"Trial {trial}"

        angle_ratio_results = pd.DataFrame(columns=['trial', 'joints', 'angle_ratio'])

        for i, d in enumerate(data):
            max_vel_time = d['time'][d['ee_velocity'].idxmax()]
            for a1, a2 in self.angles_combinations:
                angle1 = d.loc[d['time'] == max_vel_time, a1].values[0]
                angle2 = d.loc[d['time'] == max_vel_time, a2].values[0]
                angle_ratio = angle1 / angle2 if angle2 != 0 else np.nan
                angle_ratio_results.loc[len(angle_ratio_results)] = {'trial': i, 'joints': f'{a1}_{a2}', 'angle_ratio': angle_ratio}

        if plot:
            fig, ax = plt.subplots()
            sns.barplot(angle_ratio_results, x='joints', y='angle_ratio', ax=ax)
            ax.set_title(f'Angle Ratio at Max Velocity Time Point \n'+title + '\n' + self.name)
            ax.set_xlabel('Joint Pairs')
            ax.set_ylabel('Angle Ratio')
            plt.show()

        return angle_ratio_results
    
 
    
    def compute_temporal_coordination_index(self, trial=None, plot=False):
        """
        Compute the Temporal Coordination Index (TCI) for the given trial(s).
        This metric calculates the time difference between the start of the end-effector movement
        and the start of each joint's movement. The start of the movement is defined as the point
        where the velocity exceeds 5% of its maximum value.
        Parameters:
        -----------
        trial : int, optional
            The index of the trial to compute the TCI for. If None, computes TCI for all trials.
            If -1, computes TCI for the mean of all trials. Default is None.
        plot : bool, optional
            If True, plots the TCI results using a bar plot. Default is False.
        Returns:
        --------
        tci_results : pandas.DataFrame
            A DataFrame containing the TCI results with columns 'trial', 'joints', and 'tci'.
        Raises:
        -------
        ValueError
            If end-effector data is not available.
            If the trial index is out of range.
        """
        
        if not self.end_effector:
            raise ValueError("This metric can only be computed if end-effector data is available.")
        if trial == None:
            data = self.get_data_joints_angles()
            title = "All trials"
        elif trial >= len(self.data_joints_angles) or trial < -1:
            raise ValueError(f"Trial index {trial} out of range. Only {len(self.data_joints_angles)} trials available.")
        elif trial == -1:
            data = [self.get_mean_data()]
            title = "Mean of all trials"


        # Create an empty confusion matrix with the joint angles as columns and rows    
        tci_results = pd.DataFrame(columns=['trial', 'joints', 'tci'])
        for i,d in enumerate(data) :
            #get start of the movement as 5% of the maximum velocity of the end-effector
            start_of_movement1 = d[(d['ee_velocity'] > 0.05 * d['ee_velocity'].max())]['time'].head(1)
            
            for a in self.list_name_angles:
                #find the start of joint movement as 5% of the maximum velocity of the joint
                start_joint = d[(d[a+'_velocity'] < 0.05 * d[a+'_velocity'].max())]['time'].head(1)
                #Compute the TCI
                tci_results.loc[len(tci_results)] = ({'trial': i, 'joints': f'{a}', 'tci': start_joint.values[0] - start_of_movement1.values[0]})
        if plot:
            fig, ax = plt.subplots()
            sns.barplot(tci_results, x='joints', y='tci', ax=ax)
            ax.set_title(f'Temporal Coordination Index \n'+title + '\n' + self.name)
            ax.set_xlabel('Time')
            ax.set_ylabel('TCI')
            plt.show()

        return tci_results
    
    def compute_zero_crossing(self, trial=None, plot=False):
            if not self.end_effector:
                raise ValueError("This metric can only be computed if end-effector data is available.")
            if trial == None:
                data = self.get_data_joints_angles()
                title = "All trials"
            elif trial >= len(self.data_joints_angles) or trial < -1:
                raise ValueError(f"Trial index {trial} out of range. Only {len(self.data_joints_angles)} trials available.")
            elif trial == -1:
                data = [self.get_mean_data()]
                title = "Mean of all trials"
            else:
                data = [self.data_joints_angles[trial]]
                title = f"Trial {trial}"

            zero_crossing_results = pd.DataFrame(columns=['trial', 'joints', 'zero_crossing'])

            for i, d in enumerate(data):
                start_of_movement = d[(d['ee_velocity'] > 0.05 * d['ee_velocity'].max())]['time'].head(1).values[0]
                for a in self.list_name_angles:
                    deactivation_time = d[(d[f'{a}_velocity'] < 0.05 * d[f'{a}_velocity'].max())]['time'].tail(1).values[0]
                    zero_crossing_results.loc[len(zero_crossing_results)] = {'trial': i, 'joints': a, 'zero_crossing': deactivation_time - start_of_movement}

            if plot:
                fig, ax = plt.subplots()
                sns.barplot(data=zero_crossing_results, x='joints', y='zero_crossing', ax=ax)
                ax.set_title(f'Zero Crossing Time Delay \n{title} \n{self.name}')
                ax.set_xlabel('Joint')
                ax.set_ylabel('Time Delay (s)')
                plt.show()

            return zero_crossing_results
    
    def compute_dynamic_time_warping(self, trial=None, plot=False):
        """
        Computes the Dynamic Time Warping (DTW) between pairs of joints.
        Parameters:
        trial (int): The index of the trial to compute the DTW for. Default is None and uses all the data. If -1, uses the mean joints data
        plot (bool): Flag to indicate whether to plot the DTW. Default is False.
        Raises:
        ValueError: If the trial index is out of range.
        Returns:
        dict: A dataframe containing the DTW values for each pair of joints, one row per trial.
        Analysis :
        The lower the value is the more coordinated are the joints
        """
        if trial == None:
            data = self.get_data_joints_angles()
            title = "All trials"
        elif trial >= len(self.data_joints_angles) or trial < -1:
            raise ValueError(f"Trial index {trial} out of range. Only {len(self.data_joints_angles)} trials available.")
        elif trial == -1:
            data = [self.get_mean_data()]
            title = "Mean of all trials"
        else:
            data = [self.data_joints_angles[trial]]
            title = f"Trial {trial}"
        
        # Create an empty confusion matrix with the joint angles as columns and rows
        dtw_results = pd.DataFrame(columns=['trial', 'joints', 'dtw'])

        for i,d in enumerate(data) :
            for a1, a2 in self.angles_combinations:
                #compute DTW
                dtw_results.loc[len(dtw_results)] = ({'trial': i, 'joints': f'{a1}_{a2}', 'dtw': fastdtw(d[a1], d[a2])[0]})

        if plot:
            fig, ax = plt.subplots()
            sns.barplot(dtw_results, x='joints', y='dtw', ax=ax)
            ax.set_title(f'Dynamic Time Warping {a1}-{a2} \n'+title + '\n' + self.name)
            ax.set_xlabel('Time')
            ax.set_ylabel('DTW')
            plt.show()
        
        return dtw_results
   
    #%% Getter functions 

    def get_data_joints_angles(self):
        """
        Getter function for the data_joints_angles attribute.
        This function returns the data_joints_angles attribute.
        Returns:
            list: A list of DataFrames containing joint angle data.
        """
        return self.data_joints_angles

    def get_n_dof(self):
        """
        Getter function for the n_dof attribute.
        This function returns the n_dof attribute.
        Returns:
            int: The number of degrees of freedom.
        """
        return self.n_dof
    
    def get_name(self):
        """
        Getter function for the name attribute.
        This function returns the name attribute.
        Returns:
            str: The name of the dataset instance.
        """
        return self.name

    def get_list_name_angles(self):
        """
        Getter function for the list_name_angles attribute.
        This function returns the list_name_angles attribute.
        Returns:
            list: A list containing the names of the angles.
        """
        return self.list_name_angles

    def get_list_name_velocities(self):
        """
        Getter function for the list_name_velocities attribute.
        This function returns the list_name_velocities attribute.
        Returns:
            list: A list containing the names of the velocities of the angles.
        """
        return self.list_name_velocities
    
    def get_angles_combinations(self):
        """
        Getter function for the angles_combinations attribute.
        This function returns the angles_combinations attribute.
        Returns:
            list: A list containing all possible combinations of angles.
        """
        return self.angles_combinations 
    
    def get_mean_data(self):
        """
        Returns the mean of the data for all trials.
        Returns:
            DataFrame: A DataFrame containing the mean of the data for all trials.
        """
        return pd.concat(self.data_joints_angles).groupby('time').mean().reset_index()
    
    def get_concatenate_data(self):
        """
        Concatenate all joints data in a single DataFrame.
        Returns:
            DataFrame: A DataFrame containing all joints data.
        """
        return pd.concat(self.data_joints_angles)