import uproot
import numpy as np
import h5py

def read_root_file(file_name, output_file, f_type, mass=0, add_beams=False):
    """
    Read a ROOT file, process the data, and output to an HDF5 file with the keys:
    - 'Pmu': Tensor of 4-momentum components (E, px, py, pz)
    - 'Nobj': Number of particles per event
    - 'is_signal': 1 if the input is a signal file, 0 if it is a background file
    - 'label': label for the particles indicating their type

    Parameters:
        file_name (str): Path to the input ROOT file.
        output_file (str): Path to the output HDF5 file.
        f_type (str): "signal" or "background" to specify the file type.
        mass (float): Mass of the particles (default is 0).
        add_beams (bool): Whether to add beam particles (default is False).
    """
    
    def ensure_array(x):
        if np.ndim(x) == 0:
            return np.array([x])
        return x

    # Open the ROOT file
    file = uproot.open(file_name)
    tree = file["OmniTree"]
    
    # Extract pt, eta, phi branches for muons and tracks
    pt_muons = tree["pT_l1"].array(library="np")
    eta_muons = tree["eta_l1"].array(library="np")
    phi_muons = tree["phi_l1"].array(library="np")
    
    pt_tracks = tree["pT_tracks"].array(library="np")
    eta_tracks = tree["eta_tracks"].array(library="np")
    phi_tracks = tree["phi_tracks"].array(library="np")
    
    pass190 = tree["pass190"].array(library="np")
    
    # Combine muons and tracks for each event
    pt_combined = [np.concatenate([ensure_array(pt_mu), ensure_array(pt_tr)]) 
                   for pt_mu, pt_tr in zip(pt_muons, pt_tracks)]
    eta_combined = [np.concatenate([ensure_array(eta_mu), ensure_array(eta_tr)]) 
                    for eta_mu, eta_tr in zip(eta_muons, eta_tracks)]
    phi_combined = [np.concatenate([ensure_array(phi_mu), ensure_array(phi_tr)]) 
                    for phi_mu, phi_tr in zip(phi_muons, phi_tracks)]
    
    # Filter events where pass190 == 1
    pass_filter = pass190 == 1
    pt_combined = [pt for pt, passed in zip(pt_combined, pass_filter) if passed]
    eta_combined = [eta for eta, passed in zip(eta_combined, pass_filter) if passed]
    phi_combined = [phi for phi, passed in zip(phi_combined, pass_filter) if passed]

    valid_events = [i for i, pt in enumerate(pt_combined) if len(pt) > 0]
    pt_combined = [pt_combined[i] for i in valid_events]
    eta_combined = [eta_combined[i] for i in valid_events]
    phi_combined = [phi_combined[i] for i in valid_events]
    
    # Calculate 4-momentum components (E, px, py, pz) for each event
    px_combined = [pt * np.cos(phi) for pt, phi in zip(pt_combined, phi_combined)]
    py_combined = [pt * np.sin(phi) for pt, phi in zip(pt_combined, phi_combined)]
    pz_combined = [pt * np.sinh(eta) for pt, eta in zip(pt_combined, eta_combined)]
    E_combined = [np.sqrt(pt**2 * np.cosh(eta)**2 + mass**2) for pt, eta in zip(pt_combined, eta_combined)]
    
    # Find the maximum number of particles across all events
    max_particles = max(len(pt) for pt in pt_combined)
    
    # Pad variable-length arrays
    def pad_array(array_list, max_len):
        return np.array([np.pad(arr, (0, max_len - len(arr)), 'constant') for arr in array_list])
    
    # Pad the arrays and create the final 4-momentum tensor
    Pmu = np.stack([
        pad_array(E_combined, max_particles),
        pad_array(px_combined, max_particles),
        pad_array(py_combined, max_particles),
        pad_array(pz_combined, max_particles)
    ], axis=-1)

    # Calculate the number of particles per event (Nobj)
    Nobj = np.array([len(pt) for pt in pt_combined], dtype=np.int32)

    # If adding beams, append them
    if add_beams:
        # Define beam properties
        beam_mass = 0.0  # mass for beam particles
        beam_pz = 1.0    # longitudinal momentum for beam particles
        beam_E = np.sqrt(beam_mass ** 2 + beam_pz ** 2)
        
        # Create beam particle 4-momentum vectors (2 beams)
        beam_vecs = np.array([[beam_E, 0.0, 0.0, beam_pz], 
                               [beam_E, 0.0, 0.0, -beam_pz]], dtype=np.float32)

        # Create a new list for Pmu to include beam particles
        Pmu_with_beams = []

        # Update Pmu to include the beam particles
        for event_pmu in Pmu:
            event_with_beams = np.vstack((event_pmu, beam_vecs))
            Pmu_with_beams.append(event_with_beams)

        # Convert to an array
        Pmu = np.array(Pmu_with_beams)

        # Update Nobj
        Nobj += 2  # Adding two beam particles

    # Set the 'is_signal' flag
    is_signal = 1 if f_type == "signal" else 0
    is_signal_array = np.full((Pmu.shape[0],), is_signal, dtype=np.int32)

    # Initialize the label array and pad to the same length as Pmu
    label = np.zeros((Pmu.shape[0], Pmu.shape[1]), dtype=np.int32)

    # Assign label for reco particles
    for i, pt in enumerate(pt_combined):
        label[i, :len(pt)] = 1  # Assign 1 to reco particles
    
    # Add beam particles if required
    if add_beams:
        label[:, -2:] = -1  # Assign -1 for the last two beam particles

    # Write to HDF5 file
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('Pmu', data=Pmu)
        f.create_dataset('Nobj', data=Nobj)
        f.create_dataset('is_signal', data=is_signal_array)
        f.create_dataset('label', data=label)

    print(f"Data successfully written to {output_file}")

# Example usage:
read_root_file("/global/cfs/cdirs/m3246/ZjetOmnifold/data/slimmed_files/WithTracks_ZjetOmnifold_Aug5_PseudoDataSRew_Apr8_1_All.root", "test/pd_beams_label.h5", "background", add_beams=True)
read_root_file("/global/cfs/cdirs/m3246/ZjetOmnifold/data/slimmed_files/WithTracks_ZjetOmnifold_May19_MGPy8FxFxRew_syst_test_Mar0723.root", "test/testmc_beams_label.h5", "signal", add_beams=True)
read_root_file("/global/cfs/cdirs/m3246/ZjetOmnifold/data/slimmed_files/WithTracks_ZjetOmnifold_May19_MGPy8FxFxRew_syst_train_Mar1023.root", "test/trainmc_beams_label.h5", "signal", add_beams=True)
