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

    def __init__(self, list_files_angles, list_name_angles=None, name=None, deg=True, freq=None):
        """
        Initialize the CoordinationMetricsToolbox.
        Parameters:
        list_files_angles (list): List of file paths containing angle data.
        list_name_angles (list, optional): List of names corresponding to the angles. Defaults to None. If no names are passed, it is supposed that the first line containes the time + the name of the joints angles.
        name (str, optional): Name of the dataset instance. Will be printed as a header of plots. Defaults to None.
        deg (bool, optional): Flag indicating if the angles are in degrees. Defaults to True.
        freq (float, optional): Frequency of the data. If None is passed, the sampling frequency will be computed based on the time column. Defaults to None.
        """

        self.list_files_angles = list_files_angles
        self.list_name_angles = list_name_angles
        self.name = name
        self.deg = deg
        self.freq = freq

        self.load_csv_files()

        self.set_angle_names()
        self.set_velocities_names() 

        self.rename_time_column()

        self.set_n_dof()

        if not deg:
            self.convert_angles_to_radians()

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

        if self.list_name_angles is None:
            self.list_name_angles = self.data_joints_angles[0].columns[1:]

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

    def plot_joints_angles(self, trial=0):
        """
        Plots the joint angles for the specified trial.
        Parameters:
        trial (int): The index of the trial to plot. If 0, plots all trials. Default is 0.
        Raises:
        ValueError: If the trial index is out of range.
        Returns:
        None
        """

        if trial >= len(self.data_joints_angles) or trial < 0:
            raise ValueError(f"Trial index {trial} out of range. Only {len(self.data_joints_angles)} trials available.")
        elif trial ==0 :
            fig, ax = plt.subplots()
            c = generate_palette(self.n_dof)
            #plot all trials
            for df in self.data_joints_angles:
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
                fig.suptitle("Joint angles")
            

        else:
            c = generate_palette(self.n_dof)
            # plot only the selected trial
            ax = self.data_joints_angles[trial].plot(x="time", y=self.list_name_angles, color=c)
            ax.legend(self.list_name_angles)
            ax.set_xlabel("Time")
            if self.deg:
                ax.set_ylabel("Angle (degrees)")
            else:
                ax.set_ylabel("Angle (radians)")
            if self.name is not None:
                ax.set_title(f"Trial n°{trial} Joint angles for {self.name}")
            else:
                ax.set_title(f"Trial n°{trial} Joint angles")
        plt.show()
    
    def plot_joints_angular_velocity(self, trial=0):    
        """
        Plots the joint angular velocities for the specified trial.
        Parameters:
        trial (int): The index of the trial to plot. If 0, plots all trials. Default is 0.
        Raises:
        ValueError: If the trial index is out of range.
        Returns:
        None
        """
        if trial >= len(self.data_joints_angles) or trial < 0:
            raise ValueError(f"Trial index {trial} out of range. Only {len(self.data_joints_angles)} trials available.")
        elif trial == 0:
            fig, ax = plt.subplots()
            c = generate_palette(self.n_dof)
            # plot all trials
            for df in self.data_joints_angles:
                for i, angle in enumerate(self.list_name_angles):
                    df.plot(x="time", y=f"{angle}_velocity", ax=ax, color=c[i])
            ax.legend(self.list_name_angles)
            ax.set_xlabel("Time")
            if self.deg:
                ax.set_ylabel("Angular Velocity (degrees/s)")
            else:
                ax.set_ylabel("Angular Velocity (radians/s)")
            if self.name is not None:
                fig.suptitle(f"Joint angular velocities for {self.name}")
            else:
                fig.suptitle("Joint angular velocities")
        else:
            c = generate_palette(self.n_dof)
            # plot only the selected trial
            ax = self.data_joints_angles[trial].plot(x="time", y=[f"{angle}_velocity" for angle in self.list_name_angles], color=c)
            ax.legend(self.list_name_angles)
            ax.set_xlabel("Time")
            if self.deg:
                ax.set_ylabel("Angular Velocity (degrees/s)")
            else:
                ax.set_ylabel("Angular Velocity (radians/s)")
            if self.name is not None:
                ax.set_title(f"Trial n°{trial} Joint angular velocities for {self.name}")
            else:
                ax.set_title(f"Trial n°{trial} Joint angular velocities")
        plt.show()

   #%% Inter-joint coordination metrics

    def compute_continuous_relative_phase(self, trial=0, plot=False):
        """
        Computes the continuous relative phase (CRP) between pairs of joints.
        Parameters:
        trial (int): The index of the trial to compute the CRP for. Default is 0.
        plot (bool): Flag to indicate whether to plot the CRP. Default is False.
        Raises:
        ValueError: If the trial index is out of range.
        Returns:
        dict: A dataframe containing the CRP values for each pair of joints, one row per trial.
        """
        if trial >= len(self.data_joints_angles) or trial < 0:
            raise ValueError(f"Trial index {trial} out of range. Only {len(self.data_joints_angles)} trials available.")
        angles_combinations = itertools.combinations(self.list_name_angles, 2)  

        crp_result = pd.DataFrame(columns=['trial']+list(angles_combinations))

        #compute the phase for each joint
        for d in self.data_joints_angles:
            for angle, angle_vel in zip(self.list_name_angles, self.list_name_velocities):
                d[f"{angle}_phase"] = np.arctan2(d[angle_vel],d[angle])

            for a1, a2 in angles_combinations:
                d['CRP_'+a1+'_'+a2] = d[a1+'_phase'] - d[a2+'_phase']

        if plot:                    
            for a1, a2 in angles_combinations:
                fig, ax = plt.subplots()
                for d in self.data_joints_angles:
                    d.plot(x='time', y='CRP_'+a1+'_'+a2, ax=ax)
                    ax.set_title('Continuous Relative Phase')
                    ax.set_xlabel('Time')
                    ax.set_ylabel('CRP (radians)')
                    plt.show()
            
   
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

    def get_list_name_angles(self):
        """
        Getter function for the list_name_angles attribute.
        This function returns the list_name_angles attribute.
        Returns:
            list: A list containing the names of the angles.
        """
        return self.list_name_angles