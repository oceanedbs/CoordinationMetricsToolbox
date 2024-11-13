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

def generate_palette(n):
    """
    Generate a color palette with `n` colors using the 'viridis' colormap.

    Parameters
        n : int
            The number of colors to generate.

    Returns
        colors : list
            A list of RGBA color tuples.
    """
    
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i / n) for i in range(n)]
    return colors

def get_pca_frame(data, n_components=2):
    """
    Perform Principal Component Analysis (PCA) on the given data and return the PCA frame.

    Parameters
        data : array-like, shape (n_samples, n_features)
            The input data to perform PCA on.
        n_components : int, optional, default=2
            The number of principal components to compute.

    Returns
        components : array, shape (n_components, n_features)
            The principal components of the data.
        explained_variance_ratio : array, shape (n_components,)
            The amount of variance explained by each of the selected components.

    Notes
        The function normalizes the data by subtracting the mean before performing PCA.
    """

    data_norm = data - data.mean()
    pca = PCA(n_components=n_components)
    pca_data = pca.fit(data)

    # pca.components_[0,:] contains the first vector of the PCA frame
    # pca.components_[1,:] contains the second vector of the PCA frame

    components = pca.components_
    explained_variance_ratio = pca.explained_variance_ratio_
    return components, explained_variance_ratio




class CoordinationMetrics():
    """
    This class provides a toolbox for computing various coordination metrics from joint angle data. 

    It includes methods for loading data, setting angle names, computing velocities, and various 
    coordination metrics such as continuous relative phase, principal component analysis, 
    cross-correlation, and more.

    Attributes
    ----------
    list_files_angles : list
        List of file paths to the CSV files containing joint angle data.
    list_name_angles : list 
        List of names corresponding to the angles.
    name : str  
        Name of the dataset instance.
    end_effector : bool
        Flag indicating if the data contains end-effector data.
    deg : bool
        Flag indicating if the angles are in degrees.
    freq : float
        Frequency of the data.
    data_joints_angles : list
        List of pandas DataFrames containing joint angle data.
    list_name_velocities : list
        List of names of the velocities of the angles.
    angles_combinations : list
        List of all possible combinations of angles.
    n_dof : int
        Number of degrees of freedom.
        
    Methods
    -------
    load_csv_files()
        Loads joint angle data from a list of CSV files.
    set_angle_names()
        Sets the names of the angles based on the data provided.
    set_velocities_names()
        Sets the names of the velocities of the angles.
    set_angles_combinations()
        Sets the combinations of angles for computing inter-joint coordination metrics.
    rename_time_column()
        Renames the first column of each DataFrame in the `data_joints_angles` attribute to "time".
    rename_end_effector_columns()
        Renames the columns of the end-effector data.
    compute_end_effector_velocity()
        Computes the velocity of the end-effector data.
    set_n_dof()
        Sets the number of degrees of freedom (n_dof) for the object.
    convert_angles_to_radians()
        Converts the joint angles from degrees to radians.
    compute_joints_angular_velocity()
        Computes the angular velocity for each joint angle in the dataset.
    plot_joints_angles(trial=None)
        Plots the joint angles for the specified trial, all trials, or mean trial.
    plot_joints_angular_velocity(trial=None)
        Plots the joint angular velocities for the specified trial.
    compute_continuous_relative_phase(trial=None, plot=False)   
        Compute the Continuous Relative Phase (CRP) for joint angles.
    compute_angle_angle_plot(trial=None)
        Generates an angle-angle plot for joint angles data.
    compute_principal_component_analysis(trial=None, plot=False, n_components=2)
        Compute Principal Component Analysis (PCA) on joint angle data.
    compute_cross_correlation(trial=None, plot=False, normalize=False)
        Compute the cross-correlation between joint angles for a given trial or all trials.
    compute_interjoint_coupling_interval(trial=None, plot=False)
        Computes the Interjoint Coupling Interval (ICI) for the given trial or all trials.
    get_pca_subspace(n_components=None)
        Computes the PCA (Principal Component Analysis) subspace for the given data.
    

    Examples
    --------
    >>> list_files_angles = ["data/angles1.csv", "data/angles2.csv"]
    >>> list_name_angles = ["angle1", "angle2", "angle3"]
    >>> m = CoordinationMetrics(list_files_angles, list_name_angles, "Test Data", end_effector = False, deg=False)
    >>> m.compute_continuous_relative_phase(trial=4, plot=True)  
    """
    

    def __init__(self, list_files_angles, list_name_angles=None, name=None, end_effector=False,  deg=True, freq=None):
        """
        Initialize the CoordinationMetrics object.

        Parameters
        list_files_angles : list
            List of file paths to the CSV files containing joint angle data.
        list_name_angles : list, optional
            List of names corresponding to the angles. Defaults to None.
        name : str, optional
            Name of the dataset instance. Defaults to None.
        end_effector : bool, optional
            Flag indicating if the data contains end-effector data. Defaults to False.
        deg : bool, optional
            Flag indicating if the angles are in degrees. Defaults to True.
        freq : float, optional
            Frequency of the data. Defaults to None.

        Returns
        None
        """


        self.list_files_angles = list_files_angles
        self.list_name_angles = list_name_angles
        if name is not None:
            self.name = name
        else:
            # Default name
            self.name = "Dataset"
        self.end_effector = end_effector
        self.deg = deg
        self.freq = freq

        # Load data and initialize fields correctly
        self.load_csv_files()
        self.set_angle_names()
        self.set_velocities_names() 
        self.set_angles_combinations()
        self.rename_time_column()
        self.set_n_dof()

        # Convert angles to radians if necessary
        if not deg:
            self.convert_angles_to_radians()

        # If end effector is provided, rename columns and compute velocity
        if self.end_effector:
            self.rename_end_effector_columns()  
            self.compute_end_effector_velocity()

        # Compute angular velocity of the joints
        self.compute_joints_angular_velocity()  

        return None


    def load_csv_files(self):
        """
        Loads joint angle data from a list of CSV files.

        This method reads CSV files specified in `self.list_files_angles` and 
        stores the data in `self.data_joints_angles`. Each file is expected to 
        have a header row.

        Raises:
            FileNotFoundError: If any file in `self.list_files_angles` does not exist.
            ValueError: If any file in `self.list_files_angles` is not a CSV file.
        """
        # Create an empty list to store the data
        self.data_joints_angles = []

        # Load the data from each file
        for f in self.list_files_angles:
            if not os.path.exists(f):
                raise FileNotFoundError(f"File {f} not found.")
            if not f.endswith(".csv"):
                raise ValueError(f"File {f} is not a CSV file.")
            self.data_joints_angles.append(pd.read_csv(f, sep=",", header=[0]))



    def set_angle_names(self):
        """
        Sets the names of the angles based on the data provided.

        This method assigns the list of angle names to the `list_name_angles` attribute.
        If `list_name_angles` is None and `end_effector` is False, it sets `list_name_angles`
        to all columns of `data_joints_angles[0]` except the first one.
        If `list_name_angles` is None and `end_effector` is True, it sets `list_name_angles`
        to all columns of `data_joints_angles[0]` except the first one and the last three.

        Parameters
            list_name_angles : list 
                The list of angle names.
            data_joints_angles : list
                A list containing data frames with joint angles.
            end_effector : bool
                A flag indicating whether the end effector is considered.
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

        Parameters:
            list_name_angles (list): A list containing the names of the angles.
            list_name_velocities (list): A list containing the names of the velocities of the angles.
        """

        self.list_name_velocities = [f"{angle}_velocity" for angle in self.list_name_angles]

    def set_angles_combinations(self):
        """
        Sets the combinations of angles for computing inter-joint coordination metrics.
        This method generates all possible combinations of angles from the list of angle names
        and stores them in the `angles_combinations` attribute.

        Parameters:
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

        This method renames the columns of the end-effector data to "ee_x", "ee_y", and "ee_z".
        The function iterates over all dataframes containing joints angles.

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
        "ee_y_velocity", and "ee_z_velocity" columns. The global end-effector velocity is also computed
        and stored in the "ee_velocity" column. 

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

        Parameters:
            n_dof (int): The number of degrees of freedom.
            list_name_angles (list): A list containing the names of the angles.

        Returns:
            None
        """

        self.n_dof = len(self.list_name_angles)

    def convert_angles_to_radians(self):
        """
        Converts the joint angles from degrees to radians.

        This method converts the joint angles from degrees to radians by multiplying the values
        in the `data_joints_angles` attribute by the conversion factor pi/180.

        Parameters:
            data_joints_angles (list of DataFrames): A list containing DataFrames with joint angle data.

        Returns:
            None
        """

        for df in self.data_joints_angles:
            for col in self.list_name_angles:
                df[col] = df[col] * (np.pi / 180)


#%% Compute angular velocity of the joints

    def compute_joints_angular_velocity(self):
        """
        Computes the angular velocity for each joint angle in the dataset.

        This method calculates the angular velocity for each joint angle by taking the difference
        between consecutive angle values and dividing by the time difference. The resulting angular
        velocities are stored in new columns with the suffix '_velocity'.

        Parameters:
            None

        Returns:
            None
            """
        

        for df in self.data_joints_angles:
            for col, vel_col in zip(self.list_name_angles, self.list_name_velocities):
                df[vel_col] = df[col].diff()/df["time"].diff()


    #%% Plotting functions

    def plot_joints_angles(self, trial=None):
        """
        Plots the joint angles for the specified trial, all trials, or mean trial.

        Parameters:
            trial (int, optional): The index of the trial to plot. If None, plots all trials. If -1, plots the mean of all trials. Defaults to None.
            
        Raises:
            ValueError: If the trial index is out of range.
            The plot will display the joint angles over time for the specified trial(s).
            The x-axis represents time, and the y-axis represents the joint angles.
            The y-axis label will indicate whether the angles are in degrees or radians.
            The plot title will include the name of the trial(s) and, if available, the name of the dataset.
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
        c = generate_palette(self.n_dof)
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
        c = generate_palette(self.n_dof)
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
        Compute the Continuous Relative Phase (CRP) for joint angles.

        Parameters:
            trial : int, optional
                Index of the trial to compute the CRP for. If None, computes CRP for all trials.
                If -1, computes CRP for the mean of all trials. Default is None.
            plot : bool, optional
            If True, plots the CRP for the specified trial(s). Default is False.

        Returns:
            data : list of pandas.DataFrame
                List of DataFrames containing the joint angles and their computed phases, 
                as well as the CRP for the specified trial(s).

        Raises:
            ValueError
                If the specified trial index is out of range.

        Notes:
            The function computes the phase for each joint using the arctangent of the 
            joint angle velocity and the joint angle. It then computes the CRP between 
            specified pairs of joints (angles_combinations) by unwrapping the difference 
            between their phases. If plotting is enabled, it generates plots of the CRP 
            over time for the specified trial(s).
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
                #create column and fill with NaN
                d['CRP_'+a1+'_'+a2] = np.NaN
                #compute relative phase, without considering the first row that is NaN
                d.loc[1:, 'CRP_'+a1+'_'+a2] = np.unwrap(d.loc[1:, a1+'_phase'] - d.loc[1:, a2+'_phase'])

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
        Generates an angle-angle plot for joint angles data.

        Parameters:
            trial (int, optional): The index of the trial to plot. If None, all trials are concatenated and plotted.
                        If -1, the mean of all trials is plotted. Defaults to None.

        Returns:
            DataFrame: The data used for plotting.

        Raises:
            ValueError: If the trial index is out of range.
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
    
        return data

    def compute_principal_component_analysis(self, trial=None, plot=False, n_components=2):
        """
        Compute Principal Component Analysis (PCA) on joint angle data.

        Parameters:
            trial : int, optional
                Index of the trial to analyze. If None, all trials are concatenated.
                If -1, the mean of all trials is used. Default is None.
            plot : bool, optional
                If True, plots the PCA components. Default is False.
            n_components : int, optional
                Number of principal components to compute. Default is 2.

        Returns:
            pca : PCA object
                Fitted PCA object containing the principal components.

        Raises:
            ValueError
                If the trial index is out of range.
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
        Compute the cross-correlation between joint angles for a given trial or all trials.

        Parameters:
            trial : int, optional 
                Index of the trial to compute cross-correlation for. If None, computes for all trials. If -1, computes for the mean of all trials.  Defaults to None.
            plot : bool, optional
                If True, plots the cross-correlation results. Defaults to False.
            normalize : bool, optional
                If True, normalizes the data before computing cross-correlation. Defaults to False.

        Returns:
            data : list
                A list of DataFrames containing the cross-correlation results for each trial.

        Raises:
            ValueError: If the trial index is out of range.
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
        Computes the Interjoint Coupling Interval (ICI) for the given trial or all trials.

        Parameters:
            trial : int, optional
                The index of the trial to compute the ICI for. If None, computes the ICI for all trials.
                If -1, computes the ICI for the mean of all trials. Default is None.
            plot : bool, optional
                If True, plots the ICI results using a bar plot. Default is False.
        
        Returns:
            pd.DataFrame
                A DataFrame containing the ICI results with columns 'trial', 'joints', and 'ICI'.
        
        Raises:
            ValueError
                If the trial index is out of range.
        
        Notes:
            The ICI is computed as the difference in deactivation times between two joints.
            The deactivation time is defined as the first element of the last block of consecutive indices
            where the joint's velocity is less than 5% of its maximum velocity.
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
    
    def get_pca_subspace(self, n_components=None):
        """
        Computes the PCA (Principal Component Analysis) subspace for the given data.
        
        Parameters:
            n_components (int, optional): Number of principal components to keep. 
                If None, the number of components will be set to the number of degrees of freedom (self.n_dof).
        
        Returns:
            DataFrame: A DataFrame containing the PCA-transformed data with the specified number of components.
        """
        
        
        if n_components is None:
            n_components = self.n_dof
        data = self.get_concatenate_data()
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data[self.list_name_angles])

        return get_pca_frame(data_scaled, n_components)
    
    def compute_distance_between_PCs(self, cm2, n_components = None, plot=False):
        
        """
        Computes the distance between the principal components (PCs) of the joint angles.
        
        From Bockemühl Till, Troje NF, Dürr V. Inter‑joint coupling and joint angle synergies of human catching movements. Hum Mov Sci. 2010;29(1):73–93.
        https:// doi. org/ 10. 1016/j. humov. 2009. 03. 003.
        Distance between PCs is defined as dist(U, V ) = np.sqrt(1 − s^2), with s being the minimum singular value of the matrix W = min(U^T V ).
        
        Parameters:
            cm2 : CoordinationMetrics
                The CoordinationMetrics object to compare the distance between PCs with.
            trial : int
                The index of the trial to compute the distance between PCs for. Default is None and uses all the data. If -1, uses the mean joints data
            plot : bool
                Flag to indicate whether to plot the distance between PCs. Default is False.
        
        Raises:
            ValueError: If the trial index is out of range.
        
        Returns:
            res_dist_pca: A dataframe containing the distance between PCs for each pair of joints, one row per trial.
        """

        #Compute PCA on both datasets that will define U and V
        res_dist_pca = pd.DataFrame(columns=['datasetA', 'datasetB', 'distance', 'angle'])
        subspaceA, _ = self.get_pca_subspace(n_components=n_components) # U
        subspaceB, _ = cm2.get_pca_subspace(n_components=n_components) # V

        U = subspaceA
        V = subspaceB

        print(subspaceA)
        print(subspaceB)

        # Verify dimension subspaces
        if U.shape[1] > U.shape[0]:
            U = U.T
        if V.shape[1] > V.shape[0]:
            V = V.T

        # Singular value decomposition of the matrix U^T V
        S = np.linalg.svd(np.dot(U.T, V), compute_uv=False)

        # Compute the minimum of SVD
        Smin = np.min(S)

        # Compute the distance
        d = np.real(np.sqrt(1 - Smin * Smin))
        angle = np.arcsin(d) * 180 / np.pi       
        
        res_dist_pca.loc[len(res_dist_pca)] = {'datasetA': self.get_name(), 'datasetB': cm2.get_name(), 'distance': d, 'angle': angle}

        if plot:
            fig, ax = plt.subplots()
            sns.barplot(res_dist_pca, x='datasetA', y='distance', ax=ax)
            ax.set_title('Distance between PCs')
            ax.set_ylim(0, 1)   
            ax.set_xlabel('Datasets')
            ax.set_ylabel('Distance')
            plt.show()


        return res_dist_pca

        
    def compute_correlation(self, trial=None, plot=False, type='pearson'):
        """
        Compute the correlation between joint angles for a given trial or all trials.

        Parameters:
            trial : int, optional
                The index of the trial to compute the correlation for. If None, computes the correlation for all trials.
                If -1, computes the correlation for the mean of all trials. Default is None.
            plot : bool, optional
                If True, plots the correlation results. Default is False.
            type : str, optional
                The type of correlation to compute. Options are 'pearson', 'kendall', or 'spearman'. Default is 'pearson'.
        
        Returns:
            pd.DataFrame
                A DataFrame containing the correlation results with columns ['trial', 'joints', 'correlation'].
        
        Raises:
            ValueError
                If the trial index is out of range.
        
        Notes:
            The method computes the correlation between all combinations of joint angles specified in `self.angles_combinations`.
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
        Compute the ratio of joint angles at the time of maximum end-effector velocity.

        Parameters:
            trial : int, optional
                The index of the trial to compute the angle ratio for. If None, computes for all trials.
                If -1, computes for the mean of all trials. Default is None.
            plot : bool, optional
                If True, plots the angle ratios. Default is False.
        
        Returns:
            pd.DataFrame
                A DataFrame containing the trial index, joint pairs, and their corresponding angle ratios.
        
        Raises:
            ValueError
                If end-effector data is not available or if the trial index is out of range.
       
        Notes:
            The angle ratio is computed as the ratio of the angles of two joints at the time point where the end-effector velocity is maximum.
            If the angle of the second joint is zero, the ratio is set to NaN.
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
            trial : int, optional
                The index of the trial to compute the TCI for. If None, computes TCI for all trials. If -1, computes TCI for the mean of all trials. Default is None.
            plot : bool, optional
                If True, plots the TCI results using a bar plot. Default is False.

        Returns:
            tci_results : pandas.DataFrame
                A DataFrame containing the TCI results with columns 'trial', 'joints', and 'tci'.
        
        Raises:
            ValueError
                If end-effector data is not available. If the trial index is out of range.
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
            """
            Compute the zero crossing time delay for joint angles.
            
            This method calculates the time delay between the start of movement and the 
            deactivation time for each joint angle. The start of movement is defined as 
            the time when the end-effector velocity exceeds 5% of its maximum value. The 
            deactivation time for each joint is defined as the time when the joint's 
            velocity drops below 5% of its maximum value.

            Parameters:
                trial : int, optional
                    The index of the trial to compute the zero crossing for. If None, computes for all trials. If -1, computes for the mean of all trials. Defaults to None.
                plot : bool, optional
                    If True, plots the zero crossing time delay for each joint. Defaults to False.
            
            Returns:
                pd.DataFrame: A DataFrame containing the zero crossing time delay for each joint 
                          in each trial. The DataFrame has columns 'trial', 'joints', and 
                          'zero_crossing'.

            Raises:
                ValueError: If end-effector data is not available or if the trial index is out of range.
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
        Compute the Dynamic Time Warping (DTW) distance between joint angles for a given trial or all trials.

        Parameters:
            trial : int, optional
                The index of the trial to compute DTW for. If None, computes DTW for all trials. If -1, computes DTW for the mean of all trials. Defaults to None.
            plot : bool, optional
                If True, plots the DTW results. Defaults to False.

        Returns:
            pd.DataFrame: A DataFrame containing the DTW results with columns 'trial', 'joints', and 'dtw'.
        
        Raises:
            ValueError: If the trial index is out of range.
        
        Notes:
            - The DTW distance is computed for each combination of joint angles.
            - If plot is True, a bar plot of the DTW results is displayed.
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
    
    def compute_jcvpca(self, coord_metric, plot=True, n_pca=None):
        """
        Compute Joint Contribution Variation using PCA (JCVPCA).

        Parameters:
            coord_metric : object
                An object that provides the method `get_concatenate_data()` to retrieve the data for PCA.
            plot : bool, optional
                If True, the function will generate and display plots of the PCA results. Default is True.
            n_pca : int, optional
                Number of principal components to compute. If None, it defaults to the number of degrees of freedom (self.n_dof).
        
        Returns:
            subspaceA : numpy.ndarray
                The PCA subspace of the reference dataset.
            res : numpy.ndarray
                The absolute values of the projection of the second dataset in the PCA subspace.
            sub : numpy.ndarray
                The difference between the absolute values of the PCA subspaces of the two datasets.
        """
        

        if n_pca is None:
            n_pca = self.n_dof

        # 1) Compute PCA on datasetA, the reference 
        subspaceA, varA = self.get_pca_subspace(n_components=n_pca)

        # 2) Project 2nd dataset in the PCA subspace
        dataB = coord_metric.get_concatenate_data()
        dataB_transformed = np.matmul(dataB[self.list_name_angles].to_numpy(), subspaceA.T)

       
        
        # 3) Compute a PCA on these transformed data

        subspaceB, varB = get_pca_frame(dataB_transformed)
   

        # 4) Express depening on the joints
        res = np.absolute(np.matmul(subspaceB, subspaceA))
        sub = res - np.absolute(subspaceA)

        # 5) Compute the difference reported to the explained variance
        res_prop = np.array([sub[0, :] * varA[0], sub[1, :] * varA[1]]).flatten()



        if plot:
            fig, ax = plt.subplots(7)
            plt.suptitle('Absolute values')
    
            ax[0].set_title(" PC1 %.2f" % varA[0])
            ax[1].set_title(" PC2 %.2f" % varA[1])
            ax[0].set_ylim([-0.5, 1.1])
            ax[1].set_ylim([-0.5, 1.1])

            ax[0].set_ylabel(coord_metric.get_name())
   
            ax[0].bar(np.arange(len(subspaceA[0, :])), np.absolute(subspaceA[0, :]))
            ax[0].set_xticks(np.arange(len(self.list_name_angles)))
            ax[0].set_xticklabels(self.list_name_angles)

            ax[1].bar(np.arange(len(subspaceA[1, :])), np.absolute(subspaceA[1, :]))
            ax[1].set_xticks(np.arange(len(self.list_name_angles)))
            ax[1].set_xticklabels(self.list_name_angles)

            ax[2].set_title('PC1')
            ax[2].set_ylabel(coord_metric.get_name())
            ax[2].bar(np.arange(len(res[0, :])), res[0, :])
            ax[2].set_xticks(np.arange(len(self.list_name_angles)))
            ax[2].set_xticklabels(self.list_name_angles)
            ax[3].set_title('PC2')
            ax[3].bar(np.arange(len(res[1, :])), res[1, :])
            ax[3].set_xticks(np.arange(len(self.list_name_angles)))
            ax[3].set_xticklabels(self.list_name_angles)
            ax[2].set_ylim([-0.5, 1.1])
            ax[3].set_ylim([-0.5, 1.1])

            # print(res)
            ax[4].set_title('PC1')
            ax[4].set_ylabel('Diff ' + coord_metric.get_name() +
                        '- ' + self.get_name())

            ax[4].bar(np.arange(len(sub[0, :])), sub[0, :], color='orange')
            ax[4].set_xticks(np.arange(len(self.list_name_angles)))
            ax[4].set_xticklabels(self.list_name_angles)

            ax[5].set_title('PC2')
            ax[5].bar(np.arange(len(sub[1, :])), sub[1, :], color='orange')
            ax[5].set_xticks(np.arange(len(self.list_name_angles)))
            ax[5].set_xticklabels(self.list_name_angles)

            # Create a list of strings that contains n times 'PC1_n' and n times 'PC2_n'
            bins = [f'PC1_{i}' for i in self.list_name_angles] + [f'PC2_{i}' for i in self.list_name_angles]
            ax[6].bar(bins, res_prop)
            ax[6].set_title('Difference reported to the explained variance')
            ax[4].set_ylim([-0.5, 1.1])
            ax[5].set_ylim([-0.5, 1.1])
            ax[6].set_ylim([-0.5, 1.1])

            max_x =max_y= min_x= min_y = 0
            for axs in ax:
                lim_x, lim_y = max(axs.get_xlim()), max(axs.get_ylim())
                if lim_y > max_y:
                    max_y = lim_y
                lim_x, lim_y = min(axs.get_xlim()), min(axs.get_ylim())
                if lim_y < min_y:
                    min_y = lim_y
            plt.setp(fig.get_axes(), ylim=(min_y - 0.2, max_y + 0.2))

            plt.show()

        return subspaceA, res, sub
    
    def compute_jsvcrp(self, coord_metric, plot=False):
        """
        Compute the Joint Synchronization Variation based on Continuous Relative Phase (JSvCRP) between two datasets.
        
        Parameters:
            coord_metric : object 
                An object containing the second dataset with a method to compute continuous relative phase.
            plot :bool
                If True, a heatmap of the JSvCRP will be plotted. Default is False.
        
        Returns:
            pd.DataFrame: A DataFrame containing the JSvCRP values for each pair of angles.
        
        Notes:
            - The method computes the Continuous Relative Phase (CRP) for both datasets.
            - The JSvCRP is calculated as the integral of the absolute difference between the CRP values of the two datasets.
            - The resulting JSvCRP values are stored in a DataFrame with angles as both rows and columns.
            - If `plot` is True, a heatmap of the JSvCRP values is displayed.
        """
        

        # Compute the CRP for both datasets
        crp1 = self.compute_continuous_relative_phase()
        crp2 = coord_metric.compute_continuous_relative_phase()
        
        crp1=pd.concat(crp1)
        crp2=pd.concat(crp2)

        
        # Compute the JSvCRP
        res_jsvcrp = pd.DataFrame(index=self.list_name_angles, columns=self.list_name_angles)
        for a1, a2 in self.angles_combinations:
            crp1_a = crp1['CRP_'+a1+'_'+a2].dropna()
            crp2_a = crp2['CRP_'+a1+'_'+a2].dropna()
            jsvcrp = np.trapz(np.abs(crp1_a - crp2_a))
            res_jsvcrp.loc[a1, a2] = jsvcrp
            res_jsvcrp.loc[a2, a1] = jsvcrp
        res_jsvcrp = res_jsvcrp.replace(np.nan, 0)
        print(res_jsvcrp)
        if plot:
            fig, ax = plt.subplots()
            # Generate a mask for the upper triangle
            mask = np.triu(np.ones_like(res_jsvcrp, dtype=bool))
            sns.heatmap(data=res_jsvcrp, ax=ax, mask=mask, annot=True, cmap='coolwarm')
            ax.set_title('Joint Synchronization Variation based on Continuous Relative Phase')
            ax.set_xlabel('Joint Pairs')
            ax.set_ylabel('JSvCRP')
            plt.show()

        return res_jsvcrp

   
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