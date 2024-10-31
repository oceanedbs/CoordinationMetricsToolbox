import numpy as np
import matplotlib.pyplot as plt
import coordination_metrics_toolbox as cm
import os
import random
import pandas as pd

list_file_paths = []
noise_freq = 0.05
noise_phase = 0.1

# Generate n samples files for testing
def generate_test_data(n=1, plot = False): 
    
    # Write the data to a file in a folder test_dataset
    # Create the directory if it doesn't exist
    os.makedirs('test_dataset', exist_ok=True) 
    np.random.seed(42)  # For reproducibility
    t = np.linspace(0, 10, 1000)  # 100 time points from 0 to 10
    
    for i in range(n):
        file_path = os.path.join('test_dataset', 'test_data_'+str(i)+'.csv')

        # Generate random data for testing
        frequencies = [0.2+random.uniform(-noise_freq, noise_freq), 0.4+random.uniform(-noise_freq, noise_freq), 0.6+random.uniform(-noise_freq, noise_freq)]  # Different frequencies for the 3 joints
        phases = [0+random.uniform(-noise_phase, noise_phase), np.pi/4+random.uniform(-noise_phase, noise_phase), np.pi/2+random.uniform(-noise_phase, noise_phase)]  # Different phases for the 3 joints
        
        #generate sinusoids
        data = np.array([np.sin(f * t + p) for f, p in zip(frequencies, phases)]).T
        #add time column
        data = np.concatenate((t.reshape(-1, 1), data), axis=1)  # Add time column
        
        #plot data
        if plot:
            plt.plot(data[:,1:])
            plt.show()

        # Convert numpy array to pandas DataFrame
        df = pd.DataFrame(data, columns=['time'] + [f'joint_{i}' for i in range(1, data.shape[1])])
        # Save the DataFrame to a CSV file
        df.to_csv(file_path, index=False)

        list_file_paths.append(file_path)   

    return list_file_paths

if __name__ == "__main__":
    file_list = generate_test_data(5, False)
   
    # Perform coordination metrics analysis on test data
    m = cm.CoordinationMetrics(file_list)

    # Plot the data
    # m.plot_joints_angles(2)
    # m.plot_joints_angular_velocity(2)


    #Compute inter-joint coordination metrics
    m.compute_continuous_relative_phase(plot=True)

