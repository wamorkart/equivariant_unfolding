import h5py
import numpy as np

def pad_to_max(array_list, max_len):
    """Pad an array of arrays to ensure they all have the same length."""
    return np.array([np.pad(arr, ((0, max_len - len(arr)), (0, 0)), 'constant') if len(arr) < max_len else arr for arr in array_list])

def calculate_fractions(is_signal, dataset_name):
    total = len(is_signal)
    mc_count = np.sum(is_signal == 1)
    pd_count = np.sum(is_signal == 0)
    
    mc_fraction = mc_count / total
    pd_fraction = pd_count / total

    print(f"Dataset: {dataset_name}")
    print(f"Total events: {total}")
    print(f"MC fraction: {mc_fraction:.4f}")
    print(f"PD fraction: {pd_fraction:.4f}\n")

def pad_labels(labels, max_length):
    padded_labels = []
    for label in labels:
        padded_labels.append(np.pad(label, (0, max_length - len(label)), 'constant', constant_values=-1))
    return np.array(padded_labels)

def split_h5_datasets(test_file, train_file, pd_file, output_train_file, output_val_file, output_test_file, muon_level=1):
    # Load the input files
    with h5py.File(test_file, 'r') as f_testmc, h5py.File(train_file, 'r') as f_trainmc, h5py.File(pd_file, 'r') as f_pd:
        
        # Extract leading muon data from the input files
        pt_l1_testmc = f_testmc['pT_l1'][:]
        eta_l1_testmc = f_testmc['eta_l1'][:]
        phi_l1_testmc = f_testmc['phi_l1'][:]

        pt_l1_trainmc = f_trainmc['pT_l1'][:]
        eta_l1_trainmc = f_trainmc['eta_l1'][:]
        phi_l1_trainmc = f_trainmc['phi_l1'][:]

        pt_l1_pd = f_pd['pT_l1'][:]
        eta_l1_pd = f_pd['eta_l1'][:]
        phi_l1_pd = f_pd['phi_l1'][:]

        # Extract subleading muon data if muon_level >= 2
        if muon_level >= 2:
            pt_l2_testmc = f_testmc['pT_l2'][:]
            eta_l2_testmc = f_testmc['eta_l2'][:]
            phi_l2_testmc = f_testmc['phi_l2'][:]

            pt_l2_trainmc = f_trainmc['pT_l2'][:]
            eta_l2_trainmc = f_trainmc['eta_l2'][:]
            phi_l2_trainmc = f_trainmc['phi_l2'][:]

            pt_l2_pd = f_pd['pT_l2'][:]
            eta_l2_pd = f_pd['eta_l2'][:]
            phi_l2_pd = f_pd['phi_l2'][:]

        # Extract track-level data if muon_level == 3
        if muon_level == 3:
            pt_tracks_testmc = f_testmc['pT_tracks'][:]
            eta_tracks_testmc = f_testmc['eta_tracks'][:]
            phi_tracks_testmc = f_testmc['phi_tracks'][:]

            pt_tracks_trainmc = f_trainmc['pT_tracks'][:]
            eta_tracks_trainmc = f_trainmc['eta_tracks'][:]
            phi_tracks_trainmc = f_trainmc['phi_tracks'][:]

            pt_tracks_pd = f_pd['pT_tracks'][:]
            eta_tracks_pd = f_pd['eta_tracks'][:]
            phi_tracks_pd = f_pd['phi_tracks'][:]

        # Extract other variables common to all levels
        weight_testmc = f_testmc['weight'][:]
        weight_trainmc = f_trainmc['weight'][:]
        weight_pd = f_pd['weight'][:]

        Pmu_testmc = f_testmc['Pmu'][:]
        Pmu_trainmc = f_trainmc['Pmu'][:]
        Pmu_pd = f_pd['Pmu'][:]
        
        is_signal_testmc = f_testmc['is_signal'][:]
        is_signal_trainmc = f_trainmc['is_signal'][:]
        is_signal_pd = f_pd['is_signal'][:]
        
        label_testmc = f_testmc['label'][:]
        label_trainmc = f_trainmc['label'][:]
        label_pd = f_pd['label'][:]
        
        Nobj_testmc = f_testmc['Nobj'][:]
        Nobj_trainmc = f_trainmc['Nobj'][:]
        Nobj_pd = f_pd['Nobj'][:]

        # Prepare training, validation, and test datasets by splitting
        num_train = min(len(pt_l1_trainmc), len(pt_l1_pd))
        num_val = min(len(pt_l1_pd), len(pt_l1_trainmc))
        num_test = min(len(pt_l1_testmc), len(pt_l1_testmc))

        # Select equal numbers of signal and background for training
        train_mc_indices = np.random.choice(np.where(is_signal_trainmc == 1)[0], num_train // 2, replace=False)
        train_pd_indices = np.random.choice(np.where(is_signal_pd == 0)[0], num_train // 2, replace=False)

        # Create training set
        pt_l1_train = np.concatenate([pt_l1_trainmc[train_mc_indices], pt_l1_pd[train_pd_indices]], axis=0)
        eta_l1_train = np.concatenate([eta_l1_trainmc[train_mc_indices], eta_l1_pd[train_pd_indices]], axis=0)
        phi_l1_train = np.concatenate([phi_l1_trainmc[train_mc_indices], phi_l1_pd[train_pd_indices]], axis=0)
        weight_train = np.concatenate([weight_trainmc[train_mc_indices], weight_pd[train_pd_indices]], axis=0)

        # Select equal numbers of signal and background for validation
        val_mc_indices = np.random.choice(np.where(is_signal_trainmc == 1)[0], num_val // 2, replace=False)
        val_pd_indices = np.random.choice(np.where(is_signal_pd == 0)[0], num_val // 2, replace=False)

        # Create validation set
        pt_l1_val = np.concatenate([pt_l1_trainmc[val_mc_indices], pt_l1_pd[val_pd_indices]], axis=0)
        eta_l1_val = np.concatenate([eta_l1_trainmc[val_mc_indices], eta_l1_pd[val_pd_indices]], axis=0)
        phi_l1_val = np.concatenate([phi_l1_trainmc[val_mc_indices], phi_l1_pd[val_pd_indices]], axis=0)
        weight_val = np.concatenate([weight_trainmc[val_mc_indices], weight_pd[val_pd_indices]], axis=0)

        # Create test set
        test_mc_indices = np.random.choice(np.where(is_signal_testmc == 1)[0], num_test // 2, replace=False)
        test_pd_indices = np.random.choice(np.where(is_signal_pd == 0)[0], num_test // 2, replace=False)

        pt_l1_test = np.concatenate([pt_l1_testmc[test_mc_indices], pt_l1_testmc[test_pd_indices]], axis=0)
        eta_l1_test = np.concatenate([eta_l1_testmc[test_mc_indices], eta_l1_testmc[test_pd_indices]], axis=0)
        phi_l1_test = np.concatenate([phi_l1_testmc[test_mc_indices], phi_l1_testmc[test_pd_indices]], axis=0)
        weight_test = np.concatenate([weight_testmc[test_mc_indices], weight_testmc[test_pd_indices]], axis=0)

        # Split Pmu, is_signal, label, Nobj for train, validation, and test sets
        Pmu_train = np.concatenate([Pmu_trainmc[train_mc_indices], Pmu_pd[train_pd_indices]], axis=0)
        is_signal_train = np.concatenate([is_signal_trainmc[train_mc_indices], is_signal_pd[train_pd_indices]], axis=0)
        label_train = np.concatenate([label_trainmc[train_mc_indices], label_pd[train_pd_indices]], axis=0)
        Nobj_train = np.concatenate([Nobj_trainmc[train_mc_indices], Nobj_pd[train_pd_indices]], axis=0)

        Pmu_val = np.concatenate([Pmu_trainmc[val_mc_indices], Pmu_pd[val_pd_indices]], axis=0)
        is_signal_val = np.concatenate([is_signal_trainmc[val_mc_indices], is_signal_pd[val_pd_indices]], axis=0)
        label_val = np.concatenate([label_trainmc[val_mc_indices], label_pd[val_pd_indices]], axis=0)
        Nobj_val = np.concatenate([Nobj_trainmc[val_mc_indices], Nobj_pd[val_pd_indices]], axis=0)

        Pmu_test = np.concatenate([Pmu_testmc[test_mc_indices], Pmu_pd[test_pd_indices]], axis=0)
        is_signal_test = np.concatenate([is_signal_testmc[test_mc_indices], is_signal_pd[test_pd_indices]], axis=0)
        label_test = np.concatenate([label_testmc[test_mc_indices], label_pd[test_pd_indices]], axis=0)
        Nobj_test = np.concatenate([Nobj_testmc[test_mc_indices], Nobj_pd[test_pd_indices]], axis=0)

        # Calculate max lengths for padding
        max_len_train = max(len(pt_l1_train), len(eta_l1_train), len(phi_l1_train))
        max_len_val = max(len(pt_l1_val), len(eta_l1_val), len(phi_l1_val))
        max_len_test = max(len(pt_l1_test), len(eta_l1_test), len(phi_l1_test))

        # Pad datasets
        pt_l1_train = pad_to_max([pt_l1_train], max_len_train)
        eta_l1_train = pad_to_max([eta_l1_train], max_len_train)
        phi_l1_train = pad_to_max([phi_l1_train], max_len_train)

        pt_l1_val = pad_to_max([pt_l1_val], max_len_val)
        eta_l1_val = pad_to_max([eta_l1_val], max_len_val)
        phi_l1_val = pad_to_max([phi_l1_val], max_len_val)

        pt_l1_test = pad_to_max([pt_l1_test], max_len_test)
        eta_l1_test = pad_to_max([eta_l1_test], max_len_test)
        phi_l1_test = pad_to_max([phi_l1_test], max_len_test)

        # Save the datasets to output files
        with h5py.File(output_train_file, 'w') as f_train:
            f_train.create_dataset('pT_l1', data=pt_l1_train)
            f_train.create_dataset('eta_l1', data=eta_l1_train)
            f_train.create_dataset('phi_l1', data=phi_l1_train)
            f_train.create_dataset('Pmu', data=Pmu_train)
            f_train.create_dataset('is_signal', data=is_signal_train)
            f_train.create_dataset('label', data=label_train)
            f_train.create_dataset('Nobj', data=Nobj_train)
            f_train.create_dataset('weight', data=weight_train)

        with h5py.File(output_val_file, 'w') as f_val:
            f_val.create_dataset('pT_l1', data=pt_l1_val)
            f_val.create_dataset('eta_l1', data=eta_l1_val)
            f_val.create_dataset('phi_l1', data=phi_l1_val)
            f_val.create_dataset('Pmu', data=Pmu_val)
            f_val.create_dataset('is_signal', data=is_signal_val)
            f_val.create_dataset('label', data=label_val)
            f_val.create_dataset('Nobj', data=Nobj_val)
            f_val.create_dataset('weight', data=weight_val)

        with h5py.File(output_test_file, 'w') as f_test:
            f_test.create_dataset('pT_l1', data=pt_l1_test)
            f_test.create_dataset('eta_l1', data=eta_l1_test)
            f_test.create_dataset('phi_l1', data=phi_l1_test)
            f_test.create_dataset('Pmu', data=Pmu_test)
            f_test.create_dataset('is_signal', data=is_signal_test)
            f_test.create_dataset('label', data=label_test)
            f_test.create_dataset('Nobj', data=Nobj_test)
            f_test.create_dataset('weight', data=weight_test)

        # Calculate and print the fractions for each dataset
        calculate_fractions(is_signal_train, "Training")
        calculate_fractions(is_signal_val, "Validation")
        calculate_fractions(is_signal_test, "Test")





split_h5_datasets('h5files/testmc_muonlevel1.h5', 'h5files/trainmc_muonlevel1.h5', 'h5files/pd_muonlevel1.h5', '../datasets/muonlevel1/train.h5', '../datasets/muonlevel1/valid.h5', '../datasets/muonlevel1/test.h5', muon_level=1)
# split_h5_datasets('h5files/testmc_muonlevel2.h5', 'h5files/trainmc_muonlevel2.h5', 'h5files/pd_muonlevel2.h5', '../datasets/muonlevel2/train.h5', '../datasets/muonlevel2/valid.h5', '../datasets/muonlevel2/test.h5', muon_level=2)
# split_h5_datasets('h5files/testmc_muonlevel3.h5', 'h5files/trainmc_muonlevel3.h5', 'h5files/pd_muonlevel3.h5', '../datasets/muonlevel3/train.h5', '../datasets/muonlevel3/valid.h5', '../datasets/muonlevel3/test.h5', muon_level=3)