import h5py
import numpy as np

def pad_to_max(array_list, max_len):
    """Pad an array of arrays to ensure they all have the same length."""
    return np.array([np.pad(arr, ((0, max_len - len(arr)), (0, 0)), 'constant') if len(arr) < max_len else arr for arr in array_list])

def calculate_fractions(is_signal, dataset_name):
    """Calculate and print the fraction of MC and PD in a dataset."""
    total = len(is_signal)
    mc_count = np.sum(is_signal == 1)  # Assuming '1' indicates MC (signal)
    pd_count = np.sum(is_signal == 0)  # Assuming '0' indicates PD (non-signal)
    
    mc_fraction = mc_count / total
    pd_fraction = pd_count / total

    print(f"Dataset: {dataset_name}")
    print(f"Total events: {total}")
    print(f"MC fraction: {mc_fraction:.4f}")
    print(f"PD fraction: {pd_fraction:.4f}\n")

def pad_labels(labels, max_length):
    """Pad label arrays to ensure they all have the same length."""
    padded_labels = []
    for label in labels:
        padded_labels.append(np.pad(label, (0, max_length - len(label)), 'constant', constant_values=-1))  # Pad with -1 (beam particles)
    return np.array(padded_labels)

def split_h5_datasets(testmc_file, trainmc_file, pd_file, output_train_file, output_val_file, output_test_file):
    # Load datasets
    with h5py.File(testmc_file, 'r') as f:
        Pmu_testmc = f['Pmu'][:]
        Nobj_testmc = f['Nobj'][:]
        is_signal_testmc = f['is_signal'][:]
        label_testmc = f['label'][:]  # Load label for test MC

    with h5py.File(trainmc_file, 'r') as f:
        Pmu_trainmc = f['Pmu'][:]
        Nobj_trainmc = f['Nobj'][:]
        is_signal_trainmc = f['is_signal'][:]
        label_trainmc = f['label'][:]  # Load label for train MC

    with h5py.File(pd_file, 'r') as f:
        Pmu_pd = f['Pmu'][:]
        Nobj_pd = f['Nobj'][:]
        is_signal_pd = f['is_signal'][:]
        label_pd = f['label'][:]  # Load label for pseudodata

    # Check shapes before padding
    print(f"Pmu_trainmc shape: {Pmu_trainmc.shape}")
    print(f"Pmu_pd shape: {Pmu_pd.shape}")

    # Determine max particles for padding
    max_particles = max(Pmu_trainmc.shape[1], Pmu_pd.shape[1], Pmu_testmc.shape[1])

    # Pad all datasets to the same size
    Pmu_trainmc_padded = pad_to_max(Pmu_trainmc, max_particles)
    Pmu_pd_padded = pad_to_max(Pmu_pd, max_particles)
    Pmu_testmc_padded = pad_to_max(Pmu_testmc, max_particles)

    # Split data
    trainmc_80 = int(len(Pmu_trainmc_padded) * 0.8)
    pd_60 = int(len(Pmu_pd_padded) * 0.6)
    pd_20 = int(len(Pmu_pd_padded) * 0.2)

    # Pad labels to match shapes
    max_label_length = max(label_trainmc.shape[1] if len(label_trainmc.shape) > 1 else label_trainmc.shape[0],
                            label_pd.shape[1] if len(label_pd.shape) > 1 else label_pd.shape[0],
                            label_testmc.shape[1] if len(label_testmc.shape) > 1 else label_testmc.shape[0])
                            
    label_trainmc_padded = pad_labels(label_trainmc, max_label_length)
    label_pd_padded = pad_labels(label_pd, max_label_length)
    label_testmc_padded = pad_labels(label_testmc, max_label_length)

    # Training data: 80% from trainmc + 60% from pseudodata
    Pmu_train = np.concatenate([Pmu_trainmc_padded[:trainmc_80], Pmu_pd_padded[:pd_60]], axis=0)
    Nobj_train = np.concatenate([Nobj_trainmc[:trainmc_80], Nobj_pd[:pd_60]])
    is_signal_train = np.concatenate([is_signal_trainmc[:trainmc_80], is_signal_pd[:pd_60]])
    label_train = np.concatenate([label_trainmc_padded[:trainmc_80], label_pd_padded[:pd_60]], axis=0)  # Concatenate labels

    # Validation data: 20% from trainmc + 20% from pseudodata
    Pmu_val = np.concatenate([Pmu_trainmc_padded[trainmc_80:], Pmu_pd_padded[-pd_20:]], axis=0)
    Nobj_val = np.concatenate([Nobj_trainmc[trainmc_80:], Nobj_pd[-pd_20:]])
    is_signal_val = np.concatenate([is_signal_trainmc[trainmc_80:], is_signal_pd[-pd_20:]])
    label_val = np.concatenate([label_trainmc_padded[trainmc_80:], label_pd_padded[-pd_20:]], axis=0)  # Concatenate labels

    # Test data: 100% from testmc + 20% from pseudodata
    Pmu_test = np.concatenate([Pmu_testmc_padded, Pmu_pd_padded[-pd_20:]], axis=0)
    Nobj_test = np.concatenate([Nobj_testmc, Nobj_pd[-pd_20:]])
    is_signal_test = np.concatenate([is_signal_testmc, is_signal_pd[-pd_20:]])
    label_test = np.concatenate([label_testmc_padded, label_pd_padded[-pd_20:]], axis=0)  # Concatenate labels

    # Calculate and print fractions
    calculate_fractions(is_signal_test, "Test")
    calculate_fractions(is_signal_train, "Train")
    calculate_fractions(is_signal_val, "Valid")

    # Save the datasets
    with h5py.File(output_train_file, 'w') as f:
        f.create_dataset('Pmu', data=Pmu_train)
        f.create_dataset('Nobj', data=Nobj_train)
        f.create_dataset('is_signal', data=is_signal_train)
        f.create_dataset('label', data=label_train)  # Save labels

    with h5py.File(output_val_file, 'w') as f:
        f.create_dataset('Pmu', data=Pmu_val)
        f.create_dataset('Nobj', data=Nobj_val)
        f.create_dataset('is_signal', data=is_signal_val)
        f.create_dataset('label', data=label_val)  # Save labels

    with h5py.File(output_test_file, 'w') as f:
        f.create_dataset('Pmu', data=Pmu_test)
        f.create_dataset('Nobj', data=Nobj_test)
        f.create_dataset('is_signal', data=is_signal_test)
        f.create_dataset('label', data=label_test)  # Save labels

    print(f"Datasets successfully written to {output_train_file}, {output_val_file}, {output_test_file}.")

# Call the function with your file paths
split_h5_datasets('test/testmc_beams_label.h5', 'test/trainmc_beams_label.h5', 'test/pd_beams_label.h5', 'test/train_val_test_label/train.h5', 'test/train_val_test_label/valid.h5', 'test/train_val_test_label/test.h5')
