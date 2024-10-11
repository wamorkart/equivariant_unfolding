# import uproot
# import numpy as np
# import h5py

# def read_root_file(file_name, output_file, f_type, mass=0, add_beams=False, muon_only=False, muon_level=1):
#     """
#     Read a ROOT file, process the data, and output to an HDF5 file with the keys:
#     - 'Pmu': Tensor of 4-momentum components (E, px, py, pz)
#     - 'Nobj': Number of particles per event
#     - 'is_signal': 1 if the input is a signal file, 0 if it is a background file
#     - 'label': label for the particles indicating their type

#     Parameters:
#         file_name (str): Path to the input ROOT file.
#         output_file (str): Path to the output HDF5 file.
#         f_type (str): "signal" or "background" to specify the file type.
#         mass (float): Mass of the particles (default is 0).
#         add_beams (bool): Whether to add beam particles (default is False).
#         muon_only (bool): Whether to only process muon data (default is False).
#         muon_level (int): Level of information to save:
#                           1 = leading muon only,
#                           2 = leading + subleading muon,
#                           3 = leading + subleading muon + first two tracks.
#     """
    
#     def ensure_array(x):
#         if np.ndim(x) == 0:
#             return np.array([x])
#         return x
    
#     print(f"muon_level: {muon_level}")

#     # Open the ROOT file
#     file = uproot.open(file_name)
#     tree = file["OmniTree"]
    
#     # Extract pt, eta, phi branches for leading and subleading muons
#     pt_muons1 = tree["pT_l1"].array(library="np")
#     eta_muons1 = tree["eta_l1"].array(library="np")
#     phi_muons1 = tree["phi_l1"].array(library="np")
    
#     pt_muons2 = tree["pT_l2"].array(library="np")
#     eta_muons2 = tree["eta_l2"].array(library="np")
#     phi_muons2 = tree["phi_l2"].array(library="np")
    
#     # If not muon_only, also extract tracks
#     if not muon_only:
#         pt_tracks = tree["pT_tracks"].array(library="np")
#         eta_tracks = tree["eta_tracks"].array(library="np")
#         phi_tracks = tree["phi_tracks"].array(library="np")
    
#     pass190 = tree["pass190"].array(library="np")
    
#     # Build combined arrays based on muon_level
#     if muon_level == 1:
#         # Only leading muon information
#         pt_combined = [ensure_array(pt1) for pt1 in pt_muons1]
#         eta_combined = [ensure_array(eta1) for eta1 in eta_muons1]
#         phi_combined = [ensure_array(phi1) for phi1 in phi_muons1]
        
#     elif muon_level == 2:
#     # Extract subleading muon information
#         pt_muons2 = tree["pT_l2"].array(library="np")
#         eta_muons2 = tree["eta_l2"].array(library="np")
#         phi_muons2 = tree["phi_l2"].array(library="np")
        
#         # Ensure both leading and subleading muon arrays are properly combined
#         pt_combined = [np.concatenate([ensure_array(pt_mu1), ensure_array(pt_mu2)]) 
#                     for pt_mu1, pt_mu2 in zip(pt_muons1, pt_muons2)]
#         print(pt_muons1)
#         print(pt_muons2)
#         # print(pt_combined)
#         eta_combined = [np.concatenate([ensure_array(eta_mu1), ensure_array(eta_mu2)]) 
#                         for eta_mu1, eta_mu2 in zip(eta_muons1, eta_muons2)]
#         phi_combined = [np.concatenate([ensure_array(phi_mu1), ensure_array(phi_mu2)]) 
#                         for phi_mu1, phi_mu2 in zip(phi_muons1, phi_muons2)]


#     elif muon_level == 3:
#     # Save leading, subleading muons, and first two tracks
#         pt_muons2 = tree["pT_l2"].array(library="np")
#         eta_muons2 = tree["eta_l2"].array(library="np")
#         phi_muons2 = tree["phi_l2"].array(library="np")
        
#         pt_tracks = tree["pT_tracks"].array(library="np")
#         eta_tracks = tree["eta_tracks"].array(library="np")
#         phi_tracks = tree["phi_tracks"].array(library="np")
        
#         # Take only the first two tracks for each event
#         pt_tracks_2 = [ensure_array(pt[:2]) for pt in pt_tracks]
#         eta_tracks_2 = [ensure_array(eta[:2]) for eta in eta_tracks]
#         phi_tracks_2 = [ensure_array(phi[:2]) for phi in phi_tracks]
        
#         # Combine leading and subleading muons with the first two tracks
#         pt_combined = [np.concatenate([ensure_array(pt_mu1), ensure_array(pt_mu2), pt_tr])
#                     for pt_mu1, pt_mu2, pt_tr in zip(pt_muons1, pt_muons2, pt_tracks_2)]
#         eta_combined = [np.concatenate([ensure_array(eta_mu1), ensure_array(eta_mu2), eta_tr])
#                         for eta_mu1, eta_mu2, eta_tr in zip(eta_muons1, eta_muons2, eta_tracks_2)]
#         phi_combined = [np.concatenate([ensure_array(phi_mu1), ensure_array(phi_mu2), phi_tr])
#                         for phi_mu1, phi_mu2, phi_tr in zip(phi_muons1, phi_muons2, phi_tracks_2)]

    
#     # Filter events where pass190 == 1
#     pass_filter = pass190 == 1
#     pt_combined = [pt for pt, passed in zip(pt_combined, pass_filter) if passed]
#     eta_combined = [eta for eta, passed in zip(eta_combined, pass_filter) if passed]
#     phi_combined = [phi for phi, passed in zip(phi_combined, pass_filter) if passed]

#     valid_events = [i for i, pt in enumerate(pt_combined) if len(pt) > 0]
#     pt_combined = [pt_combined[i] for i in valid_events]
#     eta_combined = [eta_combined[i] for i in valid_events]
#     phi_combined = [phi_combined[i] for i in valid_events]

    
#     # Calculate 4-momentum components (E, px, py, pz) for each event
#     px_combined = [pt * np.cos(phi) for pt, phi in zip(pt_combined, phi_combined)]
#     py_combined = [pt * np.sin(phi) for pt, phi in zip(pt_combined, phi_combined)]
#     pz_combined = [pt * np.sinh(eta) for pt, eta in zip(pt_combined, eta_combined)]
#     E_combined = [np.sqrt(pt**2 * np.cosh(eta)**2 + mass**2) for pt, eta in zip(pt_combined, eta_combined)]
    
#     # Find the maximum number of particles across all events
#     max_particles = max(len(pt) for pt in pt_combined)
    
#     print(np.array(px_combined).shape)

#     # Pad variable-length arrays
#     def pad_array(array_list, max_len):
#         return np.array([np.pad(arr, (0, max_len - len(arr)), 'constant') for arr in array_list])
    
#     # Pad the arrays and create the final 4-momentum tensor
#     Pmu = np.stack([
#         pad_array(E_combined, max_particles),
#         pad_array(px_combined, max_particles),
#         pad_array(py_combined, max_particles),
#         pad_array(pz_combined, max_particles)
#     ], axis=-1)
#     print(np.array(Pmu).shape)

#     # Calculate the number of particles per event (Nobj)
#     Nobj = np.array([len(pt) for pt in pt_combined], dtype=np.int32)

#     # If adding beams, append them
#     if add_beams:
#         # Define beam properties
#         beam_mass = 0.0  # mass for beam particles
#         beam_pz = 1.0    # longitudinal momentum for beam particles
#         beam_E = np.sqrt(beam_mass ** 2 + beam_pz ** 2)
        
#         # Create beam particle 4-momentum vectors (2 beams)
#         beam_vecs = np.array([[beam_E, 0.0, 0.0, beam_pz], 
#                                [beam_E, 0.0, 0.0, -beam_pz]], dtype=np.float32)

#         # Create a new list for Pmu to include beam particles
#         Pmu_with_beams = []

#         # Update Pmu to include the beam particles
#         for event_pmu in Pmu:
#             event_with_beams = np.vstack((event_pmu, beam_vecs))
#             Pmu_with_beams.append(event_with_beams)

#         # Convert to an array
#         Pmu = np.array(Pmu_with_beams)

#         # Update Nobj
#         Nobj += 2  # Adding two beam particles

#     # Set the 'is_signal' flag
#     is_signal = 1 if f_type == "signal" else 0
#     is_signal_array = np.full((Pmu.shape[0],), is_signal, dtype=np.int32)

#     # Initialize the label array and pad to the same length as Pmu
#     label = np.zeros((Pmu.shape[0], Pmu.shape[1]), dtype=np.int32)

#     # Assign label for reco particles
#     for i, pt in enumerate(pt_combined):
#         label[i, :len(pt)] = 1  # Assign 1 to reco particles
    
#     # Add beam particles if required
#     if add_beams:
#         label[:, -2:] = -1  # Assign -1 for the last two beam particles

#     # Write to HDF5 file
#     with h5py.File(output_file, 'w') as f:
#         f.create_dataset('Pmu', data=Pmu)
#         f.create_dataset('Nobj', data=Nobj)
#         f.create_dataset('is_signal', data=is_signal_array)
#         f.create_dataset('label', data=label)

#     print(f"Data successfully written to {output_file}")



# read_root_file("/global/cfs/cdirs/m3246/ZjetOmnifold/data/slimmed_files/WithTracks_ZjetOmnifold_Aug5_PseudoDataSRew_Apr8_1_All.root", "test/pd_TEST_muonlevel2.h5", f_type="background", add_beams=False, muon_level=2)


# import uproot
# import numpy as np
# import h5py

# def read_root_file(file_name, output_file, f_type, mass=0, add_beams=False, muon_only=False, muon_level=1):
#     """
#     Read a ROOT file, process the data, and output to an HDF5 file with the keys:
#     - 'Pmu': Tensor of 4-momentum components (E, px, py, pz)
#     - 'Nobj': Number of particles per event
#     - 'is_signal': 1 if the input is a signal file, 0 if it is a background file
#     - 'label': label for the particles indicating their type
#     - 'pt', 'eta', 'phi': Additional information about the particles.

#     Parameters:
#         file_name (str): Path to the input ROOT file.
#         output_file (str): Path to the output HDF5 file.
#         f_type (str): "signal" or "background" to specify the file type.
#         mass (float): Mass of the particles (default is 0).
#         add_beams (bool): Whether to add beam particles (default is False).
#         muon_only (bool): Whether to only process muon data (default is False).
#         muon_level (int): Level of information to save:
#                           1 = leading muon only,
#                           2 = leading + subleading muon,
#                           3 = leading + subleading muon + first two tracks.
#     """
    
#     def ensure_array(x):
#         if np.ndim(x) == 0:
#             return np.array([x])
#         return x
    
#     print(f"muon_level: {muon_level}")

#     # Open the ROOT file
#     file = uproot.open(file_name)
#     tree = file["OmniTree"]
    
#     # Extract pt, eta, phi branches for leading and subleading muons
#     pt_muons1 = tree["pT_l1"].array(library="np")
#     eta_muons1 = tree["eta_l1"].array(library="np")
#     phi_muons1 = tree["phi_l1"].array(library="np")
    
#     pt_muons2 = tree["pT_l2"].array(library="np")
#     eta_muons2 = tree["eta_l2"].array(library="np")
#     phi_muons2 = tree["phi_l2"].array(library="np")
    
#     # If not muon_only, also extract tracks
#     if not muon_only:
#         pt_tracks = tree["pT_tracks"].array(library="np")
#         eta_tracks = tree["eta_tracks"].array(library="np")
#         phi_tracks = tree["phi_tracks"].array(library="np")
    
#     pass190 = tree["pass190"].array(library="np")
    
#     # Build combined arrays based on muon_level
#     if muon_level == 1:
#         # Only leading muon information
#         pt_combined = [ensure_array(pt1) for pt1 in pt_muons1]
#         eta_combined = [ensure_array(eta1) for eta1 in eta_muons1]
#         phi_combined = [ensure_array(phi1) for phi1 in phi_muons1]
        
#     elif muon_level == 2:
#         # Ensure both leading and subleading muon arrays are properly combined
#         pt_combined = [np.concatenate([ensure_array(pt_mu1), ensure_array(pt_mu2)]) 
#                        for pt_mu1, pt_mu2 in zip(pt_muons1, pt_muons2)]
#         eta_combined = [np.concatenate([ensure_array(eta_mu1), ensure_array(eta_mu2)]) 
#                         for eta_mu1, eta_mu2 in zip(eta_muons1, eta_muons2)]
#         phi_combined = [np.concatenate([ensure_array(phi_mu1), ensure_array(phi_mu2)]) 
#                         for phi_mu1, phi_mu2 in zip(phi_muons1, phi_muons2)]

#     elif muon_level == 3:
#         # Save leading, subleading muons, and first two tracks
#         pt_tracks_2 = [ensure_array(pt[:2]) for pt in pt_tracks]
#         eta_tracks_2 = [ensure_array(eta[:2]) for eta in eta_tracks]
#         phi_tracks_2 = [ensure_array(phi[:2]) for phi in phi_tracks]
        
#         # Combine leading and subleading muons with the first two tracks
#         pt_combined = [np.concatenate([ensure_array(pt_mu1), ensure_array(pt_mu2), pt_tr])
#                        for pt_mu1, pt_mu2, pt_tr in zip(pt_muons1, pt_muons2, pt_tracks_2)]
#         eta_combined = [np.concatenate([ensure_array(eta_mu1), ensure_array(eta_mu2), eta_tr])
#                         for eta_mu1, eta_mu2, eta_tr in zip(eta_muons1, eta_muons2, eta_tracks_2)]
#         phi_combined = [np.concatenate([ensure_array(phi_mu1), ensure_array(phi_mu2), phi_tr])
#                         for phi_mu1, phi_mu2, phi_tr in zip(phi_muons1, phi_muons2, phi_tracks_2)]

#     # Filter events where pass190 == 1
#     pass_filter = pass190 == 1
#     pt_combined = [pt for pt, passed in zip(pt_combined, pass_filter) if passed]
#     eta_combined = [eta for eta, passed in zip(eta_combined, pass_filter) if passed]
#     phi_combined = [phi for phi, passed in zip(phi_combined, pass_filter) if passed]

#     # Only keep events with at least one particle
#     valid_events = [i for i, pt in enumerate(pt_combined) if len(pt) > 0]
#     pt_combined = [pt_combined[i] for i in valid_events]
#     eta_combined = [eta_combined[i] for i in valid_events]
#     phi_combined = [phi_combined[i] for i in valid_events]

#     # Calculate 4-momentum components (E, px, py, pz) for each event
#     px_combined = [pt * np.cos(phi) for pt, phi in zip(pt_combined, phi_combined)]
#     py_combined = [pt * np.sin(phi) for pt, phi in zip(pt_combined, phi_combined)]
#     pz_combined = [pt * np.sinh(eta) for pt, eta in zip(pt_combined, eta_combined)]
#     E_combined = [np.sqrt(pt**2 * np.cosh(eta)**2 + mass**2) for pt, eta in zip(pt_combined, eta_combined)]
    
#     # Find the maximum number of particles across all events
#     max_particles = max(len(pt) for pt in pt_combined)

#     # Pad variable-length arrays
#     def pad_array(array_list, max_len):
#         return np.array([np.pad(arr, (0, max_len - len(arr)), 'constant') for arr in array_list])
    
#     # Pad the arrays and create the final 4-momentum tensor
#     Pmu = np.stack([
#         pad_array(E_combined, max_particles),
#         pad_array(px_combined, max_particles),
#         pad_array(py_combined, max_particles),
#         pad_array(pz_combined, max_particles)
#     ], axis=-1)

#     # Calculate the number of particles per event (Nobj)
#     Nobj = np.array([len(pt) for pt in pt_combined], dtype=np.int32)

#     # If adding beams, append them
#     if add_beams:
#         # Define beam properties
#         beam_mass = 0.0  # mass for beam particles
#         beam_pz = 1.0    # longitudinal momentum for beam particles
#         beam_E = np.sqrt(beam_mass ** 2 + beam_pz ** 2)
        
#         # Create beam particle 4-momentum vectors (2 beams)
#         beam_vecs = np.array([[beam_E, 0.0, 0.0, beam_pz], 
#                               [beam_E, 0.0, 0.0, -beam_pz]], dtype=np.float32)

#         # Create a new list for Pmu to include beam particles
#         Pmu_with_beams = []

#         # Update Pmu to include the beam particles
#         for event_pmu in Pmu:
#             event_with_beams = np.vstack((event_pmu, beam_vecs))
#             Pmu_with_beams.append(event_with_beams)

#         # Convert to an array
#         Pmu = np.array(Pmu_with_beams)

#         # Update Nobj
#         Nobj += 2  # Adding two beam particles

#     # Set the 'is_signal' flag
#     is_signal = 1 if f_type == "signal" else 0
#     is_signal_array = np.full((Pmu.shape[0],), is_signal, dtype=np.int32)

#     # Initialize the label array and pad to the same length as Pmu
#     label = np.zeros((Pmu.shape[0], Pmu.shape[1]), dtype=np.int32)

#     # Assign label for reco particles
#     for i, pt in enumerate(pt_combined):
#         label[i, :len(pt)] = 1  # Assign 1 to all valid particles

#     # Save everything to the HDF5 file
#     with h5py.File(output_file, "w") as f:
#         f.create_dataset("Pmu", data=Pmu, compression="gzip")
#         f.create_dataset("Nobj", data=Nobj, compression="gzip")
#         f.create_dataset("is_signal", data=is_signal_array, compression="gzip")
#         f.create_dataset("label", data=label, compression="gzip")
        
#         # Save pt, eta, phi as well
#         f.create_dataset("pt", data=pad_array(pt_combined, max_particles), compression="gzip")
#         f.create_dataset("eta", data=pad_array(eta_combined, max_particles), compression="gzip")
#         f.create_dataset("phi", data=pad_array(phi_combined, max_particles), compression="gzip")

#     print(f"Saved to {output_file}")


# read_root_file("/global/cfs/cdirs/m3246/ZjetOmnifold/data/slimmed_files/WithTracks_ZjetOmnifold_Aug5_PseudoDataSRew_Apr8_1_All.root", "test/pd_TEST_muonlevel3.h5", f_type="background", add_beams=False, muon_level=3)


# import uproot
# import numpy as np
# import h5py

# def read_root_file(file_name, output_file, f_type, mass=0, add_beams=False, muon_only=False, muon_level=1):
#     """
#     Read a ROOT file, process the data, and output to an HDF5 file with the keys:
#     - 'Pmu': Tensor of 4-momentum components (E, px, py, pz)
#     - 'Nobj': Number of particles per event
#     - 'is_signal': 1 if the input is a signal file, 0 if it is a background file
#     - 'label': label for the particles indicating their type
#     - 'weight': event weight

#     Parameters:
#         file_name (str): Path to the input ROOT file.
#         output_file (str): Path to the output HDF5 file.
#         f_type (str): "signal" or "background" to specify the file type.
#         mass (float): Mass of the particles (default is 0).
#         add_beams (bool): Whether to add beam particles (default is False).
#         muon_only (bool): Whether to only process muon data (default is False).
#         muon_level (int): Level of information to save:
#                           1 = leading muon only,
#                           2 = leading + subleading muon,
#                           3 = leading + subleading muon + first two tracks.
#     """
    
#     def ensure_array(x):
#         if np.ndim(x) == 0:
#             return np.array([x])
#         return x

#     print(f"muon_level: {muon_level}")

#     # Open the ROOT file
#     file = uproot.open(file_name)
#     tree = file["OmniTree"]

#     # Extract pt, eta, phi branches for leading and subleading muons
#     pt_muons1 = tree["pT_l1"].array(library="np")
#     eta_muons1 = tree["eta_l1"].array(library="np")
#     phi_muons1 = tree["phi_l1"].array(library="np")
    
#     pt_muons2 = tree["pT_l2"].array(library="np")
#     eta_muons2 = tree["eta_l2"].array(library="np")
#     phi_muons2 = tree["phi_l2"].array(library="np")
    
#     # If not muon_only, also extract tracks
#     if not muon_only:
#         pt_tracks = tree["pT_tracks"].array(library="np")
#         eta_tracks = tree["eta_tracks"].array(library="np")
#         phi_tracks = tree["phi_tracks"].array(library="np")

#     pass190 = tree["pass190"].array(library="np")
#     weights = tree["weight"].array(library="np")  # Extract weights

#     # Build combined arrays based on muon_level
#     if muon_level == 1:
#         # Only leading muon information
#         pt_combined = [ensure_array(pt1) for pt1 in pt_muons1]
#         eta_combined = [ensure_array(eta1) for eta1 in eta_muons1]
#         phi_combined = [ensure_array(phi1) for phi1 in phi_muons1]
        
#     elif muon_level == 2:
#         # Ensure both leading and subleading muon arrays are properly combined
#         pt_combined = [np.concatenate([ensure_array(pt_mu1), ensure_array(pt_mu2)]) 
#                        for pt_mu1, pt_mu2 in zip(pt_muons1, pt_muons2)]
#         eta_combined = [np.concatenate([ensure_array(eta_mu1), ensure_array(eta_mu2)]) 
#                         for eta_mu1, eta_mu2 in zip(eta_muons1, eta_muons2)]
#         phi_combined = [np.concatenate([ensure_array(phi_mu1), ensure_array(phi_mu2)]) 
#                         for phi_mu1, phi_mu2 in zip(phi_muons1, phi_muons2)]

#     elif muon_level == 3:
#         # Save leading, subleading muons, and first two tracks
#         pt_tracks_2 = [ensure_array(pt[:2]) for pt in pt_tracks]
#         eta_tracks_2 = [ensure_array(eta[:2]) for eta in eta_tracks]
#         phi_tracks_2 = [ensure_array(phi[:2]) for phi in phi_tracks]
        
#         # Combine leading and subleading muons with the first two tracks
#         pt_combined = [np.concatenate([ensure_array(pt_mu1), ensure_array(pt_mu2), pt_tr])
#                        for pt_mu1, pt_mu2, pt_tr in zip(pt_muons1, pt_muons2, pt_tracks_2)]
#         eta_combined = [np.concatenate([ensure_array(eta_mu1), ensure_array(eta_mu2), eta_tr])
#                         for eta_mu1, eta_mu2, eta_tr in zip(eta_muons1, eta_muons2, eta_tracks_2)]
#         phi_combined = [np.concatenate([ensure_array(phi_mu1), ensure_array(phi_mu2), phi_tr])
#                         for phi_mu1, phi_mu2, phi_tr in zip(phi_muons1, phi_muons2, phi_tracks_2)]

#     # Filter events where pass190 == 1
#     pass_filter = pass190 == 1
#     pt_combined = [pt for pt, passed in zip(pt_combined, pass_filter) if passed]
#     eta_combined = [eta for eta, passed in zip(eta_combined, pass_filter) if passed]
#     phi_combined = [phi for phi, passed in zip(phi_combined, pass_filter) if passed]

#     valid_events = [i for i, pt in enumerate(pt_combined) if len(pt) > 0]
#     pt_combined = [pt_combined[i] for i in valid_events]
#     eta_combined = [eta_combined[i] for i in valid_events]
#     phi_combined = [phi_combined[i] for i in valid_events]

#     # Calculate 4-momentum components (E, px, py, pz) for each event
#     px_combined = [pt * np.cos(phi) for pt, phi in zip(pt_combined, phi_combined)]
#     py_combined = [pt * np.sin(phi) for pt, phi in zip(pt_combined, phi_combined)]
#     pz_combined = [pt * np.sinh(eta) for pt, eta in zip(pt_combined, eta_combined)]
#     E_combined = [np.sqrt(pt**2 * np.cosh(eta)**2 + mass**2) for pt, eta in zip(pt_combined, eta_combined)]

#     # Find the maximum number of particles across all events
#     max_particles = max(len(pt) for pt in pt_combined)

#     #print(np.array(px_combined).shape)

#     # Pad variable-length arrays
#     def pad_array(array_list, max_len):
#         return np.array([np.pad(arr, (0, max_len - len(arr)), 'constant') for arr in array_list])

#     # Pad the arrays and create the final 4-momentum tensor
#     Pmu = np.stack([
#         pad_array(E_combined, max_particles),
#         pad_array(px_combined, max_particles),
#         pad_array(py_combined, max_particles),
#         pad_array(pz_combined, max_particles)
#     ], axis=-1)

#     print(np.array(Pmu).shape)

#     # Calculate the number of particles per event (Nobj)
#     Nobj = np.array([len(pt) for pt in pt_combined], dtype=np.int32)

#     # If adding beams, append them
#     if add_beams:
#         # Define beam properties
#         beam_mass = 0.0  # mass for beam particles
#         beam_pz = 1.0    # longitudinal momentum for beam particles
#         beam_E = np.sqrt(beam_mass ** 2 + beam_pz ** 2)
        
#         # Create beam particle 4-momentum vectors (2 beams)
#         beam_vecs = np.array([[beam_E, 0.0, 0.0, beam_pz], 
#                                [beam_E, 0.0, 0.0, -beam_pz]], dtype=np.float32)

#         # Create a new list for Pmu to include beam particles
#         Pmu_with_beams = []

#         # Update Pmu to include the beam particles
#         for event_pmu in Pmu:
#             event_with_beams = np.vstack((event_pmu, beam_vecs))
#             Pmu_with_beams.append(event_with_beams)

#         # Convert to an array
#         Pmu = np.array(Pmu_with_beams)

#         # Update Nobj
#         Nobj += 2  # Adding two beam particles

#     # Set the 'is_signal' flag
#     is_signal = 1 if f_type == "signal" else 0
#     is_signal_array = np.full((Pmu.shape[0],), is_signal, dtype=np.int32)

#     # Initialize the label array and pad to the same length as Pmu
#     label = np.zeros((Pmu.shape[0], Pmu.shape[1]), dtype=np.int32)

#     # Assign label for reco particles
#     for i, pt in enumerate(pt_combined):
#         label[i, :len(pt)] = 1  # Assign 1 to reco particles
    
#     # Add beam particles if required
#     if add_beams:
#         label[:, -2:] = -1  # Assign -1 to beam particles

#     # Save the data into an HDF5 file
#     with h5py.File(output_file, "w") as h5file:
#         h5file.create_dataset("Pmu", data=Pmu)
#         h5file.create_dataset("Nobj", data=Nobj)
#         h5file.create_dataset("is_signal", data=is_signal_array)
#         h5file.create_dataset("label", data=label)
#         h5file.create_dataset("weight", data=np.array(weights)[pass_filter])  # Save weights as a dataset
        
#         # Optionally save muon information
#         if muon_level >= 1:
#             h5file.create_dataset("pT_l1", data=np.array(pt_muons1)[pass_filter])
#             h5file.create_dataset("eta_l1", data=np.array(eta_muons1)[pass_filter])
#             h5file.create_dataset("phi_l1", data=np.array(phi_muons1)[pass_filter])
        
#         if muon_level >= 2:
#             h5file.create_dataset("pT_l2", data=np.array(pt_muons2)[pass_filter])
#             h5file.create_dataset("eta_l2", data=np.array(eta_muons2)[pass_filter])
#             h5file.create_dataset("phi_l2", data=np.array(phi_muons2)[pass_filter])
        
#         if muon_level == 3:
#             # Save track information if muon_level is 3
#             h5file.create_dataset("pT_tracks", data=np.array(pt_tracks)[pass_filter])
#             h5file.create_dataset("eta_tracks", data=np.array(eta_tracks)[pass_filter])
#             h5file.create_dataset("phi_tracks", data=np.array(phi_tracks)[pass_filter])

#     print(f"Data saved to {output_file} successfully.")


import uproot
import numpy as np
import h5py

def read_root_file(file_name, output_file, f_type, mass=0, add_beams=False, muon_only=False, muon_level=1):
    """
    Read a ROOT file, process the data, and output to an HDF5 file with the keys:
    - 'Pmu': Tensor of 4-momentum components (E, px, py, pz)
    - 'Nobj': Number of particles per event
    - 'is_signal': 1 if the input is a signal file, 0 if it is a background file
    - 'label': label for the particles indicating their type
    - 'weight': event weight

    Parameters:
        file_name (str): Path to the input ROOT file.
        output_file (str): Path to the output HDF5 file.
        f_type (str): "signal" or "background" to specify the file type.
        mass (float): Mass of the particles (default is 0).
        add_beams (bool): Whether to add beam particles (default is False).
        muon_only (bool): Whether to only process muon data (default is False).
        muon_level (int): Level of information to save:
                          1 = leading muon only,
                          2 = leading + subleading muon,
                          3 = leading + subleading muon + first two tracks.
    """

    def ensure_array(x):
        if np.ndim(x) == 0:
            return np.array([x])
        return x

    print(f"muon_level: {muon_level}")

    # Open the ROOT file
    file = uproot.open(file_name)
    tree = file["OmniTree"]

    # Extract pt, eta, phi branches for leading and subleading muons
    pt_muons1 = tree["pT_l1"].array(library="np")
    eta_muons1 = tree["eta_l1"].array(library="np")
    phi_muons1 = tree["phi_l1"].array(library="np")
    
    pt_muons2 = tree["pT_l2"].array(library="np")
    eta_muons2 = tree["eta_l2"].array(library="np")
    phi_muons2 = tree["phi_l2"].array(library="np")
    
    # If not muon_only, also extract tracks
    if not muon_only:
        pt_tracks = tree["pT_tracks"].array(library="np")
        eta_tracks = tree["eta_tracks"].array(library="np")
        phi_tracks = tree["phi_tracks"].array(library="np")

    pass190 = tree["pass190"].array(library="np")
    weights = tree["weight"].array(library="np")  # Extract weights

    # Build combined arrays based on muon_level
    if muon_level == 1:
        # Only leading muon information
        pt_combined = [ensure_array(pt1) for pt1 in pt_muons1]
        eta_combined = [ensure_array(eta1) for eta1 in eta_muons1]
        phi_combined = [ensure_array(phi1) for phi1 in phi_muons1]
        
    elif muon_level == 2:
        # Ensure both leading and subleading muon arrays are properly combined
        pt_combined = [np.concatenate([ensure_array(pt_mu1), ensure_array(pt_mu2)]) 
                       for pt_mu1, pt_mu2 in zip(pt_muons1, pt_muons2)]
        eta_combined = [np.concatenate([ensure_array(eta_mu1), ensure_array(eta_mu2)]) 
                        for eta_mu1, eta_mu2 in zip(eta_muons1, eta_muons2)]
        phi_combined = [np.concatenate([ensure_array(phi_mu1), ensure_array(phi_mu2)]) 
                        for phi_mu1, phi_mu2 in zip(phi_muons1, phi_muons2)]

    elif muon_level == 3:
        # Save leading, subleading muons, and first two tracks
        pt_tracks_2 = [ensure_array(pt[:2]) for pt in pt_tracks]
        eta_tracks_2 = [ensure_array(eta[:2]) for eta in eta_tracks]
        phi_tracks_2 = [ensure_array(phi[:2]) for phi in phi_tracks]
        
        # Combine leading and subleading muons with the first two tracks
        pt_combined = [np.concatenate([ensure_array(pt_mu1), ensure_array(pt_mu2), pt_tr])
                       for pt_mu1, pt_mu2, pt_tr in zip(pt_muons1, pt_muons2, pt_tracks_2)]
        eta_combined = [np.concatenate([ensure_array(eta_mu1), ensure_array(eta_mu2), eta_tr])
                        for eta_mu1, eta_mu2, eta_tr in zip(eta_muons1, eta_muons2, eta_tracks_2)]
        phi_combined = [np.concatenate([ensure_array(phi_mu1), ensure_array(phi_mu2), phi_tr])
                        for phi_mu1, phi_mu2, phi_tr in zip(phi_muons1, phi_muons2, phi_tracks_2)]

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
        return np.array([np.pad(arr, (0, max_len - len(arr)), 'constant', constant_values=-999) for arr in array_list])

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

    # Set the 'is_signal' array based on the input type
    is_signal_array = np.ones(len(Pmu)) if f_type == "signal" else np.zeros(len(Pmu))

    # Create labels for all particles (muons and tracks)
    # 0 for beam particles, 1 for leading muon, 2 for subleading muon, and -1 for tracks
    label = np.ones(Pmu.shape[:-1], dtype=np.int32) * -1  # Default: tracks
    label[:, 0] = 1  # Leading muon
    if muon_level >= 2:
        label[:, 1] = 2  # Subleading muon
    if add_beams:
        label[:, -2:] = 0  # Assign 0 to beam particles

    # Save the data into an HDF5 file
    with h5py.File(output_file, "w") as h5file:
        h5file.create_dataset("Pmu", data=Pmu)
        h5file.create_dataset("Nobj", data=Nobj)
        h5file.create_dataset("is_signal", data=is_signal_array)
        h5file.create_dataset("label", data=label)
        h5file.create_dataset("weight", data=np.array(weights)[pass_filter])  # Save weights as a dataset
        
        # Optionally save muon information
        if muon_level >= 1:
            h5file.create_dataset("pT_l1", data=np.array(pt_muons1)[pass_filter])
            h5file.create_dataset("eta_l1", data=np.array(eta_muons1)[pass_filter])
            h5file.create_dataset("phi_l1", data=np.array(phi_muons1)[pass_filter])
        
        if muon_level >= 2:
            h5file.create_dataset("pT_l2", data=np.array(pt_muons2)[pass_filter])
            h5file.create_dataset("eta_l2", data=np.array(eta_muons2)[pass_filter])
            h5file.create_dataset("phi_l2", data=np.array(phi_muons2)[pass_filter])
        
        if muon_level == 3:
            # Save track information if muon_level is 3
            h5file.create_dataset("pT_tracks", data=pad_array(pt_tracks_2, 2))  # Save the first two tracks, padded
            h5file.create_dataset("eta_tracks", data=pad_array(eta_tracks_2, 2))
            h5file.create_dataset("phi_tracks", data=pad_array(phi_tracks_2, 2))

    print(f"Data saved to {output_file} successfully.")



# Example usage
# read_root_file("/global/cfs/cdirs/m3246/ZjetOmnifold/data/slimmed_files/WithTracks_ZjetOmnifold_Aug5_PseudoDataSRew_Apr8_1_All.root", "test/pd_TEST_muonlevel1.h5", f_type="background", add_beams=False, muon_level=1)
read_root_file("/global/cfs/cdirs/m3246/ZjetOmnifold/data/slimmed_files/WithTracks_ZjetOmnifold_May19_MGPy8FxFxRew_syst_test_Mar0723.root", "h5files/testmc_muonlevel1.h5", f_type="signal", add_beams=False, muon_level=1)
read_root_file("/global/cfs/cdirs/m3246/ZjetOmnifold/data/slimmed_files/WithTracks_ZjetOmnifold_May19_MGPy8FxFxRew_syst_test_Mar0723.root", "h5files/testmc_muonlevel2.h5", f_type="signal", add_beams=False, muon_level=2)
read_root_file("/global/cfs/cdirs/m3246/ZjetOmnifold/data/slimmed_files/WithTracks_ZjetOmnifold_May19_MGPy8FxFxRew_syst_test_Mar0723.root", "h5files/testmc_muonlevel3.h5", f_type="signal", add_beams=False, muon_level=3)


read_root_file("/global/cfs/cdirs/m3246/ZjetOmnifold/data/slimmed_files/WithTracks_ZjetOmnifold_Aug5_PseudoDataSRew_Apr8_1_All.root", "h5files/pd_muonlevel1.h5", f_type="background", add_beams=False, muon_level=1)
read_root_file("/global/cfs/cdirs/m3246/ZjetOmnifold/data/slimmed_files/WithTracks_ZjetOmnifold_Aug5_PseudoDataSRew_Apr8_1_All.root", "h5files/pd_muonlevel2.h5", f_type="background", add_beams=False, muon_level=2)
read_root_file("/global/cfs/cdirs/m3246/ZjetOmnifold/data/slimmed_files/WithTracks_ZjetOmnifold_Aug5_PseudoDataSRew_Apr8_1_All.root", "h5files/pd_muonlevel3.h5", f_type="background", add_beams=False, muon_level=3)

read_root_file("/global/cfs/cdirs/m3246/ZjetOmnifold/data/slimmed_files/WithTracks_ZjetOmnifold_May19_MGPy8FxFxRew_syst_train_Mar1023.root", "h5files/trainmc_muonlevel1.h5", f_type="signal", add_beams=False, muon_level=1)
read_root_file("/global/cfs/cdirs/m3246/ZjetOmnifold/data/slimmed_files/WithTracks_ZjetOmnifold_May19_MGPy8FxFxRew_syst_train_Mar1023.root", "h5files/trainmc_muonlevel2.h5", f_type="signal", add_beams=False, muon_level=2)
read_root_file("/global/cfs/cdirs/m3246/ZjetOmnifold/data/slimmed_files/WithTracks_ZjetOmnifold_May19_MGPy8FxFxRew_syst_train_Mar1023.root", "h5files/trainmc_muonlevel3.h5", f_type="signal", add_beams=False, muon_level=3)


