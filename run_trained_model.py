"""Example prediction of partial charges using model
trained via active learning.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import pickle
from pathlib import Path

from jax import numpy as jnp, random, vmap
import numpy as onp

from chemtrain import util, probabilistic, property_prediction, sparse_graph


n_dropout_samples = 8
data_idx = 0

data_path = 'data/validation_MOF_data.pkl'
param_path = 'data/params.pkl'
dropout_mode = {'output': 0.1, 'interaction': 0.1, 'embedding': 0.1}
r_cut = 5.  # A
alpha = 1.504

with open(data_path, 'rb') as pickle_file:
    atom_types, positions, boxes, partial_charges, ids, dbs = pickle.load(pickle_file)
with open(param_path, 'rb') as pickle_file:
    trained_params = pickle.load(pickle_file)

key = random.PRNGKey(0)

_, predictor = property_prediction.partial_charge_prediction(
        r_cut, dropout_mode=dropout_mode)
model = property_prediction.init_model(predictor)
batched_model = vmap(model, in_axes=(None, 0))
uq_fn = probabilistic.init_dropout_uq_fwd(batched_model, trained_params, n_dropout_samples)

graph = sparse_graph.sparse_graph_from_positions(
    r_cut, jnp.array(positions[data_idx]), jnp.array(boxes[data_idx]),
    jnp.array(atom_types[data_idx])
)
graph = util.tree_expand_dim(graph)
uq_prediction = onp.array(uq_fn(key, graph)[0])
print(uq_prediction.shape)

mean_prediction = onp.mean(uq_prediction, axis=0)
uq_metric = jnp.std(uq_prediction, axis=0, ddof=1)
calibrated_uq_metric = uq_metric * alpha

