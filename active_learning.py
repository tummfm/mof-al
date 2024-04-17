"""Active learning of a GNN model for partial charge prediction.

The code was run on 4 RTX3090 GPUs.
"""
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

from functools import partial
from pathlib import Path
import pickle

from jax import random, numpy as jnp, lax, device_count
from jax_sgmc.data import numpy_loader
import numpy as onp
import optax

from chemtrain import (property_prediction, util, max_likelihood, trainers,
                       dropout, data_processing, probabilistic)

dataset_str = 'data/train_data.pkl'
save_path = 'initial_model.pkl'
checkpoint_folder = 'checkpoints/'
state_folder = 'checkpoints/last_state/'
init_recompute = True
load_last_checkpoint = False

checkpoint_freq = 50

# due to small dataset size, many epochs necessary to have sufficient number of
# updates to yield best model possible given the dataset size
init_epochs = 1000

# new training needs to be sufficient also to learn very different new MOFs
new_data_epochs = 200
al_retrain_epochs = 3

heldout_data_ratio = 0.2  # validation and test
init_train_ratio = 0.01
train_ratio = 1.
val_ratio = 0.
al_batch_size = 16
new_samples_per_al_batch = 4
al_case = 'true_error'
al_case = 'random'
al_case = 'UQ'


dropout_mode = {'output': 0.1, 'interaction': 0.1, 'embedding': 0.1}
n_dropout_samples = 8

r_cut = 5.  # A
n_species = 100
model_key = random.PRNGKey(0)
batch_cache = 2
batch_per_device = 2
forward_batch_size = 8
batch_size = batch_per_device * device_count()

initial_lr = 0.001
decay_factor = 0.1

checkpoint_folder += al_case
state_folder += al_case
numpy_seed = 0
dropout_seed = 42

Path(checkpoint_folder).mkdir(parents=True, exist_ok=True)
Path(state_folder).mkdir(parents=True, exist_ok=True)

with open(dataset_str, 'rb') as f:
    data_graph, padded_charges, ids = pickle.load(f)

targets = {'charges': padded_charges}
combined_dataset = {'targets': targets, 'graph_dataset': data_graph}

full_train_data, _, test_data = data_processing.train_val_test_split(
    combined_dataset, train_ratio=1. - heldout_data_ratio, val_ratio=0.)
train_idxs, _, test_idxs = data_processing.train_val_test_split(
    onp.arange(len(ids)), train_ratio=1. - heldout_data_ratio, val_ratio=0.)


def data_from_idxs(idxs):
    return util.tree_take(full_train_data, idxs, axis=0)


def save_data_idxs(path, label_idxs, pool_idxs):
    with open(path, 'wb') as f:
        pickle.dump((label_idxs, pool_idxs), f)


def al_splits(al_batch_size, samples_per_batch):
    assert samples_per_batch <= batch_size, ('Cannot put more samples in batch'
                                             ' than batch_size.')
    assert al_batch_size % samples_per_batch == 0, ('al_batch is not divisible'
                                                    ' by samples_per_batch.')
    n_splits = al_batch_size // samples_per_batch
    return n_splits


n_al_splits = al_splits(al_batch_size, new_samples_per_al_batch)
numpy_rng = onp.random.default_rng(numpy_seed)
total_train_data_size = util.tree_multiplicity(full_train_data)
n_init_train_samples = int(init_train_ratio * total_train_data_size)
full_train_indicies = onp.arange(total_train_data_size)
numpy_rng_data_shuffle = onp.random.default_rng(0)
labeled_idxs = numpy_rng_data_shuffle.choice(full_train_indicies,
                                             size=n_init_train_samples,
                                             replace=False)
pool_idxs = onp.delete(full_train_indicies, labeled_idxs)
pool_data = data_from_idxs(pool_idxs)
train_data = data_from_idxs(labeled_idxs)
mean_std_history, uq_pred_history, per_species_history = [], [], []

al_iterations = pool_idxs.size // al_batch_size
updates_for_new_samples = n_al_splits * new_data_epochs * al_iterations
al_equilibration_updates = (al_retrain_epochs * al_iterations
                            * int(0.5 * pool_idxs.size / batch_size))
initial_updates = int(init_epochs * labeled_idxs.size / batch_size)

total_updates = (updates_for_new_samples + initial_updates
                 + al_equilibration_updates)
lr_schedule = optax.exponential_decay(-initial_lr, total_updates, decay_factor)
optimizer = optax.chain(
    optax.scale_by_adam(),
    optax.scale_by_schedule(lr_schedule)
)

if init_recompute and not load_last_checkpoint:
    sample_graph = util.tree_get_single(data_graph)
    init_fn, predictor = property_prediction.partial_charge_prediction(
        r_cut, dropout_mode=dropout_mode)

    if dropout_seed is None:
        init_params = init_fn(model_key, sample_graph)
    else:
        dropout_init_key = random.PRNGKey(dropout_seed)
        init_params = init_fn(model_key, sample_graph,
                              dropout_key=dropout_init_key)
        init_params = dropout.build_dropout_params(init_params,
                                                   dropout_init_key)


    def error_fn(predictions, batch, mask, test_data=False):
        if test_data:
            return max_likelihood.mae_loss(predictions, batch['charges'], mask)
        else:
            return jnp.sqrt(max_likelihood.mse_loss(predictions,
                                                    batch['charges'], mask))

    test_error_fn = partial(error_fn, test_data=True)

    trainer = trainers.PropertyPrediction(
        error_fn, predictor, init_params, optimizer,
        train_data['graph_dataset'], train_data['targets'], batch_per_device,
        batch_cache, test_error_fn=test_error_fn, train_ratio=train_ratio,
        val_ratio=val_ratio
    )
    trainer.set_testloader(test_data)

    # initial training
    trainer.train(init_epochs)
    trainer.save_trainer(save_path)
elif load_last_checkpoint:
    with open(state_folder + '_idxs.pkl', 'rb') as f:
        labeled_idxs, pool_idxs = pickle.load(f)
    trainer = util.load_trainer(state_folder + '_trainer.pkl')
    with open(state_folder + '_errors.pkl', 'rb') as f:
        (mean_std_history, uq_pred_history, per_species_history) = pickle.load(f)
else:
    trainer = util.load_trainer(save_path)

pool_data = data_from_idxs(pool_idxs)
train_data = data_from_idxs(labeled_idxs)
max_al_iters = int(pool_idxs.size / al_batch_size)
rng_key = random.PRNGKey(0)

# for per-species error evaluation
test_mask = test_data['graph_dataset']['species_mask']
real_test_species = test_data['graph_dataset']['species'][test_mask]
unique_test_species = jnp.unique(real_test_species)


def uq_predictions(trainer, dataset, rng_key):
    use_key, new_key = random.split(rng_key, 2)
    pool_loader = numpy_loader.NumpyDataLoader(
        **dataset['graph_dataset'].to_dict(), copy=False)
    test_uncertainties = probabilistic.dropout_uq_predictions(
        trainer.batched_model, trainer.params, pool_loader,
        n_dropout_samples=n_dropout_samples, batch_size=forward_batch_size,
        batch_cache=batch_cache, include_without_dropout=False,
        init_rng_key=use_key)
    return test_uncertainties, new_key


for i_al in range(len(mean_std_history), len(mean_std_history) + max_al_iters):
    pool_uncertainties, rng_key = uq_predictions(trainer, pool_data, rng_key)

    mask = pool_data['graph_dataset'].species_mask
    particles_per_box = jnp.sum(mask, axis=1)
    uq_pool_mean = jnp.mean(pool_uncertainties, axis=1)
    uq_metric = jnp.std(pool_uncertainties, axis=1, ddof=1)  # Bessel correction
    box_uq_metric = jnp.sum(uq_metric, axis=1) / particles_per_box

    mean_per_box_std = jnp.mean(box_uq_metric)
    mean_std_history.append(mean_per_box_std)
    print(f'Iteration {i_al}: Mean box std = {mean_per_box_std}')

    uq_test_samples, rng_key = uq_predictions(trainer, test_data, rng_key)
    uq_mean_test_preictions = jnp.mean(uq_test_samples, axis=1)
    test_abs_errors = jnp.abs(uq_mean_test_preictions
                              - test_data['targets']['charges'])
    test_abs_errors_masked = test_abs_errors[test_mask]
    mae_uq_testerror = jnp.mean(test_abs_errors_masked)
    print(f'Iteration {i_al}: UQ prediction MAE = {mae_uq_testerror}')
    uq_pred_history.append(mae_uq_testerror)

    per_species_errors = property_prediction.per_species_results(
        real_test_species, test_abs_errors_masked, unique_test_species)
    mean_per_species_errors = jnp.mean(per_species_errors)
    print(f'Iteration {i_al}: Mean per-species error:{mean_per_species_errors}')
    per_species_history.append(mean_per_species_errors)


    # move samples with highest uncertainty from unlabeled pool into train_data
    if al_case == 'random':
        highest_indices = numpy_rng.choice(onp.arange(pool_idxs.size),
                                           size=al_batch_size, replace=False)
    elif al_case == 'UQ':
        _, highest_indices = lax.top_k(box_uq_metric, al_batch_size)
        highest_indices = onp.array(highest_indices)
    elif al_case == 'true_error':
        uq_pool_predictions = jnp.mean(pool_uncertainties, axis=1)
        pool_errors = jnp.abs(uq_pool_predictions
                              - pool_data['targets']['charges'])
        box_errors = jnp.sum(pool_errors, axis=1) / particles_per_box
        _, highest_indices = lax.top_k(box_errors, al_batch_size)
        highest_indices = onp.array(highest_indices)
    else:
        raise ValueError('"al_case" not recognized.')

    selected_indices = pool_idxs[highest_indices]
    labeled_idxs = onp.append(labeled_idxs, selected_indices)
    pool_idxs = onp.delete(pool_idxs, highest_indices)

    recent_added_data = data_from_idxs(selected_indices)
    pool_data = data_from_idxs(pool_idxs)
    train_data = data_from_idxs(labeled_idxs)

    for _ in range(new_data_epochs):
        shuffled_indices = onp.arange(al_batch_size)
        numpy_rng.shuffle(shuffled_indices)
        al_batches = onp.split(shuffled_indices, n_al_splits)
        for al_batch in al_batches:
            samples = util.tree_take(recent_added_data, al_batch)
            trainer.update_with_samples(**samples)

    trainer.update_dataset(train_ratio, val_ratio, **train_data)
    trainer.set_testloader(test_data)
    trainer.train(al_retrain_epochs)

    def save_trainingstate(folder, i_al=''):
        trainer.save_trainer(folder + f'_trainer{i_al}.pkl')
        save_data_idxs(folder + f'_idxs{i_al}.pkl', labeled_idxs, pool_idxs)
        with open(folder + f'_errors{i_al}.pkl', 'wb') as f:
            pickle.dump((mean_std_history,
                         uq_pred_history, per_species_history), f)
        trainer.save_energy_params(f'{folder}_params{i_al}.pkl',
                                   save_format='.pkl')

    save_trainingstate(state_folder)
    if checkpoint_freq is not None:
        if i_al % checkpoint_freq == 0:
            save_trainingstate(checkpoint_folder, i_al)

