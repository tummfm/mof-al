"""Pre-processes the MOF data. This will take a while."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = str()  # CPU here often faster

from jax import config
config.update('jax_disable_jit', True)

from multiprocessing.connection import Client
import pickle

import numpy as onp

from chemtrain import sparse_graph


train_cutoff = 100
r_cut = 5.  # 5 A for dimenet++
save_path = 'data/train_data.pkl'
save_path_big = 'data/test_data.pkl'
data_path = 'data/raw_mof_data.pkl'

with open(data_path, 'rb') as pickle_file:
    retrieved_dataset = pickle.load(pickle_file)
json_data = retrieved_dataset['properties']
n_mofs = len(json_data['struc_cell'])

init_train_mof_atoms = []
for i in range(n_mofs):
    atoms = onp.array(json_data['struc_numbers'][i])
    if atoms.size <= train_cutoff:
        init_train_mof_atoms.append(atoms)

train_atom_types = onp.concatenate(init_train_mof_atoms)
train_species_frequencies = onp.bincount(train_atom_types)
train_atom_types = onp.arange(train_species_frequencies.size)
valid_species = train_atom_types[(train_species_frequencies >= 10)]

atom_types, positions, boxes, ids, charges = [], [], [], [], []
atom_types_big, positions_big, boxes_big, ids_big, partial_charges_big = [], [], [], [], []
atom_types_big_dropped, positions_big_dropped, boxes_big_dropped, ids_big_dropped, partial_charges_big_dropped = [], [], [], [], []
dropped_mofs_train, dropped_mofs_big = 0, 0
for i in range(n_mofs):
    atoms = onp.array(json_data['struc_numbers'][i])
    charge_labels = json_data['partial_charges'][i]
    unique_atom_types = onp.unique(atoms)
    valid_types = onp.in1d(unique_atom_types, valid_species)

    if atoms.size <= train_cutoff:
        if not onp.all(valid_types):
            dropped_mofs_train += 1
            continue
        atom_types.append(atoms)
        positions.append(onp.array(json_data['struc_positions'][i]))
        boxes.append(onp.array(json_data['struc_cell'][i]))
        ids.append(json_data['id'][i])
        charges.append(onp.array(charge_labels))
    else:
        if not onp.all(valid_types):
            dropped_mofs_big += 1
            atom_types_big_dropped.append(atoms)
            positions_big_dropped.append(onp.array(json_data['struc_positions'][i]))
            boxes_big_dropped.append(onp.array(json_data['struc_cell'][i]))
            ids_big_dropped.append(json_data['id'][i])
            partial_charges_big_dropped.append(onp.array(charge_labels))
            continue

        # convert to list of arrays for labeled data
        atom_types_big.append(atoms)
        positions_big.append(onp.array(json_data['struc_positions'][i]))
        boxes_big.append(onp.array(json_data['struc_cell'][i]))
        ids_big.append(json_data['id'][i])
        partial_charges_big.append(onp.array(charge_labels))

boxes = onp.array(boxes)
boxes_big = onp.array(boxes_big)

charges, _ = sparse_graph.pad_per_atom_quantities(charges)

data_graph = sparse_graph.convert_dataset_to_graphs(
    r_cut, positions, boxes, atom_types)

with open(save_path, 'wb') as f:
    pickle.dump([data_graph, charges, ids], f)

with open(save_path_big, 'wb') as f:
    pickle.dump([atom_types_big, positions_big, boxes_big, ids_big,
                 partial_charges_big], f)
