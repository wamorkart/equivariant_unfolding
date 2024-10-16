import h5py
import numpy as np

def concatenate_h5_files(file1_path, file2_path, output_file_path, num_entries=50000):
    """
    Concatenates the first `num_entries` from two HDF5 files and saves to a new HDF5 file.
    
    Parameters:
    - file1_path: Path to the first HDF5 file.
    - file2_path: Path to the second HDF5 file.
    - output_file_path: Path where the concatenated output HDF5 file will be saved.
    - num_entries: Number of entries to read from each file (default: 50000).
    """
    
    # Open both files for reading
    with h5py.File(file1_path, 'r') as file1, h5py.File(file2_path, 'r') as file2:
        
        # Create output file to save concatenated data
        with h5py.File(output_file_path, 'w') as output_file:
            
            # Iterate over all the keys in the first file
            for key in file1.keys():
                # Read the first `num_entries` from both files for the current key
                data1 = file1[key][:num_entries]
                data2 = file2[key][:num_entries]
                
                # Concatenate the data along the first axis (i.e., stacking entries)
                concatenated_data = np.concatenate([data1, data2], axis=0)
                
                # Save concatenated data to the output file
                output_file.create_dataset(key, data=concatenated_data)
                
                print(f"Concatenated {key}: {data1.shape[0]} + {data2.shape[0]} = {concatenated_data.shape[0]} entries")
    
    print(f"Data concatenation complete. Saved to {output_file_path}")

# Example usage:
file1_path = 'file1.h5'
file2_path = 'file2.h5'
output_file_path = 'concatenated_output.h5'

inputdirectory = "h5files/"
outputdirectory = "datasets/samedata/"
concatenate_h5_files(inputdirectory+"pd_muonlevel1_background.h5", inputdirectory+"pd_muonlevel1_signal.h5", outputdirectory+"valid.h5", num_entries=50000)
concatenate_h5_files(inputdirectory+"trainmc_muonlevel1_background.h5", inputdirectory+"trainmc_muonlevel1_signal.h5", outputdirectory+"train.h5", num_entries=50000)
concatenate_h5_files(inputdirectory+"testmc_muonlevel1_background.h5", inputdirectory+"testmc_muonlevel1_signal.h5", outputdirectory+"test.h5", num_entries=50000)
