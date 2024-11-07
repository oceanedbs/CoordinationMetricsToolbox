import matplotlib.pyplot as plt
import os
import random
import numpy as np
import pandas as pd

list_file_paths = []
noise_freq = 0.05
noise_phase = 0.1

L1, L2, L3 = 1.0, 1.0, 1.0  # Link lengths

def dh_transform(a, alpha, d, theta):
    """
    Create the Denavit-Hartenberg transformation matrix.
    
    Parameters:
    - a: Link length
    - alpha: Link twist
    - d: Link offset
    - theta: Joint angle
    
    Returns:
    - Transformation matrix as a 4x4 numpy array.
    """
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0,              np.sin(alpha),                np.cos(alpha),               d],
        [0,              0,                            0,                           1]
    ])


def forward_kinematics_3_joints(theta1, theta2, theta3, L1, L2, L3):
        """
        Compute the forward kinematics for a 3-joint rotational robot arm over a trajectory.
        
        Parameters:
        - theta1, theta2, theta3: Arrays of joint angles in radians
        - L1, L2, L3: Link lengths
        
        Returns:
        - The positions of the end effector as an array of shape (n, 3) where n is the number of time points.
        """
        positions = []
        for t1, t2, t3 in zip(theta1, theta2, theta3):
            pos = forward_kinematics_3_joint(t1, t2, t3, L1, L2, L3)
            positions.append(pos)
        return np.array(positions)

def forward_kinematics_3_joint(theta1, theta2, theta3, L1, L2, L3):
    """
    Compute the forward kinematics for a 3-joint rotational robot arm.
    
    Parameters:
    - theta1, theta2, theta3: Joint angles in radians
    - L1, L2, L3: Link lengths
    
    Returns:
    - The position of the end effector as (x, y, z).
    """
    # DH parameters for each joint
    # For this example, assume all alpha and d values are 0 (planar arm)
    T1 = dh_transform(L1, 0, 0, theta1)
    T2 = dh_transform(L2, 0, 0, theta2)
    T3 = dh_transform(L3, 0, 0, theta3)

    # Calculate the transformation matrix from the base to the end effector
    T = T1 @ T2 @ T3

    # The position of the end effector is given by the first three elements of the last column
    end_effector_pos = T[:3, 3]
    
    return end_effector_pos



# Generate n samples files for testing
def generate_test_data(n=1, plot = False, folder_name = 'test_dataset'): 
    
    # Write the data to a file in a folder test_dataset
    # Create the directory if it doesn't exist
    os.makedirs('tests/'+folder_name, exist_ok=True) 
    np.random.seed(42)  # For reproducibility
    t = np.linspace(0, 10, 1000)  # 100 time points from 0 to 10
    
    for i in range(n):
        file_path = os.path.join('tests', folder_name, 'test_data_'+str(i)+'.csv')

        # Generate random data for testing
        frequencies = [0.1+random.uniform(-noise_freq, noise_freq), 0.2+random.uniform(-noise_freq, noise_freq), 0.3+random.uniform(-noise_freq, noise_freq)]  # Different frequencies for the 3 joints
        phases = [np.pi/2+random.uniform(-noise_phase, noise_phase), np.pi/2+random.uniform(-noise_phase, noise_phase), np.pi/4+random.uniform(-noise_phase, noise_phase)]  # Different phases for the 3 joints
        
        #generate sinusoids
        data = np.array([np.sin(f * t + p) for f, p in zip(frequencies, phases)]).T
        #add time column
        data = np.concatenate((t.reshape(-1, 1), data), axis=1)  # Add time column
        #compute end effector position
        theta1, theta2, theta3 = data[:, 1], data[:, 2], data[:, 3]
        end_effector_pos = forward_kinematics_3_joints(theta1, theta2, theta3, L1, L2, L3)
        data = np.concatenate((data, end_effector_pos), axis=1)  # Add time column
        
        #plot data
        if plot:
            plt.plot(data[:,1:4])
            plt.show()

        # Convert numpy array to pandas DataFrame
        df = pd.DataFrame(data, columns=['time'] + [f'joint_{i}' for i in range(1,4)]+['x', 'y', 'z'])
        # Save the DataFrame to a CSV file
        df.to_csv(file_path, index=False)

        list_file_paths.append(file_path)   

    return list_file_paths

if __name__ == "__main__":
    file_list = generate_test_data(5, False, 'test_dataset2')