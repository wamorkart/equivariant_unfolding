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
        num_train = len(pt_l1_trainmc)
        num_val = len(pt_l1_pd)
        num_test = len(pt_l1_testmc)

        # Split train, validation, and test sets for leading muon
        pt_l1_train = np.concatenate([pt_l1_trainmc[:num_train // 2], pt_l1_pd[:num_train // 2]], axis=0)
        eta_l1_train = np.concatenate([eta_l1_trainmc[:num_train // 2], eta_l1_pd[:num_train // 2]], axis=0)
        phi_l1_train = np.concatenate([phi_l1_trainmc[:num_train // 2], phi_l1_pd[:num_train // 2]], axis=0)
        weight_train = np.concatenate([weight_trainmc[:num_train // 2], weight_pd[:num_train // 2]], axis=0)

        pt_l1_val = pt_l1_trainmc[num_train // 2:num_train // 2 + num_val // 2]
        eta_l1_val = eta_l1_trainmc[num_train // 2:num_train // 2 + num_val // 2]
        phi_l1_val = phi_l1_trainmc[num_train // 2:num_train // 2 + num_val // 2]
        weight_val = weight_trainmc[num_train // 2:num_train // 2 + num_val // 2]

        pt_l1_test = pt_l1_testmc[:num_test // 2]
        eta_l1_test = eta_l1_testmc[:num_test // 2]
        phi_l1_test = phi_l1_testmc[:num_test // 2]
        weight_test = weight_testmc[:num_test // 2]

         # Split Pmu, is_signal, label, Nobj for train, validation, and test sets
        Pmu_train = np.concatenate([Pmu_trainmc[:num_train // 2], Pmu_pd[:num_train // 2]], axis=0)
        is_signal_train = np.concatenate([is_signal_trainmc[:num_train // 2], is_signal_pd[:num_train // 2]], axis=0)
        label_train = np.concatenate([label_trainmc[:num_train // 2], label_pd[:num_train // 2]], axis=0)
        Nobj_train = np.concatenate([Nobj_trainmc[:num_train // 2], Nobj_pd[:num_train // 2]], axis=0)

        Pmu_val = Pmu_trainmc[num_train // 2:num_train // 2 + num_val // 2]
        is_signal_val = is_signal_trainmc[num_train // 2:num_train // 2 + num_val // 2]
        label_val = label_trainmc[num_train // 2:num_train // 2 + num_val // 2]
        Nobj_val = Nobj_trainmc[num_train // 2:num_train // 2 + num_val // 2]

        Pmu_test = Pmu_testmc[:num_test // 2]
        is_signal_test = is_signal_testmc[:num_test // 2]
        label_test = label_testmc[:num_test // 2]
        Nobj_test = Nobj_testmc[:num_test // 2]

        # Split subleading muon data if muon_level >= 2
        if muon_level >= 2:
            pt_l2_train = np.concatenate([pt_l2_trainmc[:num_train // 2], pt_l2_pd[:num_train // 2]], axis=0)
            eta_l2_train = np.concatenate([eta_l2_trainmc[:num_train // 2], eta_l2_pd[:num_train // 2]], axis=0)
            phi_l2_train = np.concatenate([phi_l2_trainmc[:num_train // 2], phi_l2_pd[:num_train // 2]], axis=0)

            pt_l2_val = pt_l2_trainmc[num_train // 2:num_train // 2 + num_val // 2]
            eta_l2_val = eta_l2_trainmc[num_train // 2:num_train // 2 + num_val // 2]
            phi_l2_val = phi_l2_trainmc[num_train // 2:num_train // 2 + num_val // 2]

            pt_l2_test = pt_l2_testmc[:num_test // 2]
            eta_l2_test = eta_l2_testmc[:num_test // 2]
            phi_l2_test = phi_l2_testmc[:num_test // 2]

        # Split track-level data if muon_level == 3
        if muon_level == 3:
            pt_tracks_train = np.concatenate([pt_tracks_trainmc[:num_train // 2], pt_tracks_pd[:num_train // 2]], axis=0)
            eta_tracks_train = np.concatenate([eta_tracks_trainmc[:num_train // 2], eta_tracks_pd[:num_train // 2]], axis=0)
            phi_tracks_train = np.concatenate([phi_tracks_trainmc[:num_train // 2], phi_tracks_pd[:num_train // 2]], axis=0)

            pt_tracks_val = pt_tracks_trainmc[num_train // 2:num_train // 2 + num_val // 2]
            eta_tracks_val = eta_tracks_trainmc[num_train // 2:num_train // 2 + num_val // 2]
            phi_tracks_val = phi_tracks_trainmc[num_train // 2:num_train // 2 + num_val // 2]

            pt_tracks_test = pt_tracks_testmc[:num_test // 2]
            eta_tracks_test = eta_tracks_testmc[:num_test // 2]
            phi_tracks_test = phi_tracks_testmc[:num_test // 2]

        # Calculate and print fractions
        calculate_fractions(is_signal_test, "Test")
        calculate_fractions(is_signal_train, "Train")
        calculate_fractions(is_signal_val, "Valid")

        # Print start and end indices
        print(f"Train indices: start=0, end={len(Pmu_train)} (used: {output_train_file})")
        print(f"Validation indices: start={len(Pmu_train)}, end={len(Pmu_train) + len(Pmu_val)} (used: {output_val_file})")
        print(f"Test indices: start={len(Pmu_train) + len(Pmu_val)}, end={len(Pmu_train) + len(Pmu_val) + len(Pmu_test)} (used: {output_test_file})")


    # Save datasets for training, validation, and testing
    with h5py.File(output_train_file, 'w') as f:
        f.create_dataset('pT_l1', data=pt_l1_train)
        f.create_dataset('eta_l1', data=eta_l1_train)
        f.create_dataset('phi_l1', data=phi_l1_train)
        f.create_dataset('weight', data=weight_train)
        f.create_dataset('Pmu', data=Pmu_train)
        f.create_dataset('is_signal', data=is_signal_train)
        f.create_dataset('label', data=label_train)
        f.create_dataset('Nobj', data=Nobj_train)

        if muon_level >= 2:
            f.create_dataset('pT_l2', data=pt_l2_train)
            f.create_dataset('eta_l2', data=eta_l2_train)
            f.create_dataset('phi_l2', data=phi_l2_train)
        if muon_level == 3:
            f.create_dataset('pT_tracks', data=pt_tracks_train)
            f.create_dataset('eta_tracks', data=eta_tracks_train)
            f.create_dataset('phi_tracks', data=phi_tracks_train)

    with h5py.File(output_val_file, 'w') as f:
        f.create_dataset('pT_l1', data=pt_l1_val)
        f.create_dataset('eta_l1', data=eta_l1_val)
        f.create_dataset('phi_l1', data=phi_l1_val)
        f.create_dataset('weight', data=weight_val)
        f.create_dataset('Pmu', data=Pmu_val)
        f.create_dataset('is_signal', data=is_signal_val)
        f.create_dataset('label', data=label_val)
        f.create_dataset('Nobj', data=Nobj_val)
        if muon_level >= 2:
            f.create_dataset('pT_l2', data=pt_l2_val)
            f.create_dataset('eta_l2', data=eta_l2_val)
            f.create_dataset('phi_l2', data=phi_l2_val)
        if muon_level == 3:
            f.create_dataset('pT_tracks', data=pt_tracks_val)
            f.create_dataset('eta_tracks', data=eta_tracks_val)
            f.create_dataset('phi_tracks', data=phi_tracks_val)

    with h5py.File(output_test_file, 'w') as f:
        f.create_dataset('pT_l1', data=pt_l1_test)
        f.create_dataset('eta_l1', data=eta_l1_test)
        f.create_dataset('phi_l1', data=phi_l1_test)
        f.create_dataset('weight', data=weight_test)
        f.create_dataset('Pmu', data=Pmu_test)
        f.create_dataset('is_signal', data=is_signal_test)
        f.create_dataset('label', data=label_test)
        f.create_dataset('Nobj', data=Nobj_test)
        if muon_level >= 2:
            f.create_dataset('pT_l2', data=pt_l2_test)
            f.create_dataset('eta_l2', data=eta_l2_test)
            f.create_dataset('phi_l2', data=phi_l2_test)
        if muon_level == 3:
            f.create_dataset('pT_tracks', data=pt_tracks_test)
            f.create_dataset('eta_tracks', data=eta_tracks_test)
            f.create_dataset('phi_tracks', data=phi_tracks_test)
    
    print(f"Training, validation, and test datasets have been successfully saved to {output_train_file}, {output_val_file}, and {output_test_file}.")





split_h5_datasets('h5files/testmc_muonlevel1.h5', 'h5files/trainmc_muonlevel1.h5', 'h5files/pd_muonlevel1.h5', '../datasets/muonlevel1/train.h5', '../datasets/muonlevel1/valid.h5', '../datasets/muonlevel1/test.h5', muon_level=1)
split_h5_datasets('h5files/testmc_muonlevel2.h5', 'h5files/trainmc_muonlevel2.h5', 'h5files/pd_muonlevel2.h5', '../datasets/muonlevel2/train.h5', '../datasets/muonlevel2/valid.h5', '../datasets/muonlevel2/test.h5', muon_level=2)
split_h5_datasets('h5files/testmc_muonlevel3.h5', 'h5files/trainmc_muonlevel3.h5', 'h5files/pd_muonlevel3.h5', '../datasets/muonlevel3/train.h5', '../datasets/muonlevel3/valid.h5', '../datasets/muonlevel3/test.h5', muon_level=3)
