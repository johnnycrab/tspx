import os
import shutil
import time

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from dataset import TSPDataset, donothing_collate
from torch.utils.data import DataLoader
from network import TSPNetwork
from functools import partial
from tqdm import tqdm
import orbax.checkpoint as ocp

from jumanji.environments import TSP
from jumanji.environments.routing.tsp.reward import SparseReward
from jumanji.environments.routing.tsp.types import State
from env_utils import InstancesFromFileGenerator, batch_env_reset_and_visit_first, state_to_network_rep
from jumanji.types import TimeStep

seed = 43
TRAIN_DATASET_PATH = "./data/tsp_100_10k.pickle"
VAL_DATASET_PATH = "./data/tsp_100_10k.pickle"
learning_rate = 2e-4
num_epochs = 1000
num_batches_per_epoch = 20
batch_size = 32
validation_batch_size = 8
custom_val_instances = 40  # Set to `None` for taking all instances vom VAL_DATASET_PATH, otherwise we only take the first n instances

results_path = "./results/model_weights"  # Path where to store checkpoints
# IMPORTANT NOTE: Due to a "fork()" problem that tensorstore has (https://github.com/google/orbax/issues/1658), we
# need to have tensorstore at version < 0.1.72
load_checkpoint_from_path = "./results/model_weights/composite_checkpoint"


## Training methods

@nnx.jit
def forward(model: TSPNetwork, batch: dict):
    """
    Aux function to pass batch through the model.
    """
    return model(batch)


def loss_fn(model: TSPNetwork, batch: dict):
    """
    Cross-entropy loss on batch.
    """
    logits = model(batch)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch["next_node_idx"].squeeze(-1)
    ).mean()
    return loss


@nnx.jit
def train_step(model: TSPNetwork, optimizer: nnx.Optimizer, metrics: nnx.Metric, batch: dict):
    """
    Train for a single step, updating the loss in `metrics`.
    """
    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(model, batch)
    metrics.update(loss=loss)
    optimizer.update(grads)
    return loss


# Jumanji TSP rollout methods


@partial(nnx.jit, static_argnames="sharding")
def log_probability_fn(model: TSPNetwork, sharding: NamedSharding, states: State):
    """
    Given a batch of states and TSPNetwork, return logits
    for the next actions. Function is jitted, so we should make sure to
    not vary the batch size.
    """
    batch = jax.vmap(state_to_network_rep)(states)
    batch = jax.tree.map(lambda x: jax.lax.with_sharding_constraint(x, sharding), batch)
    logits = model(
        batch
    )
    return nnx.log_softmax(logits)


@partial(nnx.jit, static_argnames=("sharding", "env_wrapper"))
def batch_greedy_rollout(model: TSPNetwork, sharding: NamedSharding, env_wrapper: TSP, initial_states: State, initial_timesteps: TimeStep) -> tuple[State, TimeStep, TSPNetwork]:
    """
    Greedy rollout given a batch of states and a model.

    Args:
        model: TSPNetwork instance
        env_wrapper: TSP environment from which we can call the step_fn
        sharding: NamedSharding on which to constrain the batches of states piped into network.
        initial_states: Batch of TSP states given as PyTree, initialized where first action has already been taken
        initial_timesteps: Batch of TSP timesteps given as PyTree, initialized where first action has already been taken

    Returns:
        Batch of final states and timesteps
    """
    step_fn = jax.vmap(env_wrapper.step)

    def body_fun(input):
        states, timesteps, network = input
        logits = log_probability_fn(network, sharding, states)  # batched logits
        # get argmax of logits, corresponding to greedy action
        actions = jnp.argmax(logits, axis=-1)
        new_states, new_timesteps = step_fn(states, actions)
        return new_states, new_timesteps, network

    return nnx.while_loop(
        cond_fun=lambda x: x[1].reward[0] == 0,  # we continue until we have obtained a nonzero reward in a timestep.
        body_fun=body_fun,
        init_val=(initial_states, initial_timesteps, model)  # don't forget to also pass in the network as initial value
    )


def save_checkpoint(val_metric: float, epoch_num: int, model: TSPNetwork, optimizer: nnx.Optimizer, ckpt_name: str, absolute_destination_path: str):
    """
    Save model and optimizer state together with metadata.

    Params:
        val_metric: Validation metric that is saved with the checkpoint, usually optimality gap.
        epoch_num: Number of epochs trained that is saved with the checkpoint.
        model: Neural network to save
        optimizer: Optimizer to save
        ckpt_name: Name of the subfolder in which the data will be stored.
        absolute_destination_path: Absolute path to folder where subfolder with checkpoint should be stored.

    """
    path = os.path.join(absolute_destination_path, ckpt_name)
    if os.path.exists(path):
        shutil.rmtree(path)

    model_state = nnx.state(model)
    opt_state = nnx.state(optimizer.opt_state)
    with ocp.Checkpointer(ocp.CompositeCheckpointHandler()) as checkpointer:
        checkpointer.save(
            path,
            args=ocp.args.Composite(
                model_state=ocp.args.PyTreeSave(model_state),
                opt_state=ocp.args.PyTreeSave(opt_state),
                metadata=ocp.args.JsonSave(dict(
                    val_metric=val_metric,
                    epoch=epoch_num
                )),
            )
        )


def load_checkpoint(path_to_ckpt, model: TSPNetwork, optimizer: nnx.Optimizer):
    graph_def, model_state = nnx.split(model)
    opt_def, opt_state = nnx.split(optimizer.opt_state)
    with ocp.Checkpointer(ocp.CompositeCheckpointHandler()) as checkpointer:
        restored = checkpointer.restore(
            directory=os.path.abspath(path_to_ckpt),
            args=ocp.args.Composite(
                model_state=ocp.args.PyTreeRestore(item=model_state),
                opt_state=ocp.args.PyTreeRestore(item=opt_state),
                metadata=ocp.args.JsonRestore()
            )
        )
    model = nnx.merge(graph_def, restored["model_state"])

    # Recreate checkpoint with the new model
    optimizer = nnx.Optimizer(
        model,
        optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate)
        )
    )
    opt_def, opt_state = nnx.split(optimizer.opt_state)
    optimizer.opt_state = nnx.merge(opt_def, restored["opt_state"])
    return model, optimizer, restored["metadata"]


if __name__ == '__main__':
    os.makedirs(results_path, exist_ok=True)

    # Create a mesh of two dimensions and annotate each axis with a name.
    mesh = Mesh(devices=np.array(jax.devices()).reshape(len(jax.devices()),),
                axis_names=('data'))
    # In data parallelism, the first dimension (batch) will be sharded on the `data` axis.
    data_sharding = NamedSharding(mesh, PartitionSpec('data', None))

    # Prepare dataset and dataloader
    train_dataset = TSPDataset(path_to_pickle=TRAIN_DATASET_PATH,
                               batch_size=batch_size,
                               num_batches=num_batches_per_epoch,
                               data_augmentation=True,
                               data_augmentation_linear_scale=True,
                               augment_direction=True,
                               custom_num_instances=None
                               )
    loader = DataLoader(train_dataset,
                        batch_size=1, shuffle=False, num_workers=2, drop_last=False, collate_fn=donothing_collate,
                        persistent_workers=True)

    # Init network
    network = TSPNetwork(latent_dim=128, num_trf_blocks=3, num_attn_heads=8, feedforward_dim=512, dropout=0.0, rngs=nnx.Rngs(seed))

    # Loss metrics and optimizer
    avg_loss_metrics = nnx.metrics.Average("loss")
    optimizer = nnx.Optimizer(
        network,
        optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate)
        )
    )

    # Load checkpoint if needed
    current_validation_metric = float("inf")
    start_epoch = 0
    if load_checkpoint_from_path:
        network, optimizer, metrics = load_checkpoint(load_checkpoint_from_path, network, optimizer)
        current_validation_metric = metrics["val_metric"]
        start_epoch = metrics["epoch"]

    # Jumanji TSP environment generator used for validation (where we perform greedy rollouts)
    env = TSP(
        generator=InstancesFromFileGenerator(num_cities=100, path_to_pickle=VAL_DATASET_PATH, custom_num_instances=custom_val_instances),
        reward_fn=SparseReward()
    )

    # Training loop
    for epoch in range(start_epoch, num_epochs + 1):
        data_iter = iter(loader)

        with mesh:
            network.train()
            num_batches = min(num_batches_per_epoch, len(loader))

            for i in (progress_bar := tqdm(range(num_batches))):
                # `next(data_iter)` returns the batch as a dict of PyTorch tensors. Convert them to jnp.Array.
                jnp_batch = jax.tree.map(lambda x: jax.device_put(jnp.asarray(x), data_sharding), next(data_iter))
                loss = train_step(network, optimizer, avg_loss_metrics, jnp_batch)
                progress_bar.set_postfix({"loss": f"{loss:.3f}"})

            avg_loss_in_epoch = avg_loss_metrics.compute()
            avg_loss_metrics.reset()
            print(f">> Epoch {epoch}. Average loss: {avg_loss_in_epoch:.3f}")

            # Epoch done.
            # Perform validation.
            network.eval()
            idcs = list(range(env.generator.length))
            idx_batches = [idcs[i*validation_batch_size: (i+1)*validation_batch_size] for i in range(len(idcs) // validation_batch_size)]

            tour_lengths = []
            for i in (progress_bar := tqdm(range(len(idx_batches)))):
                state_batch, timestep_batch = batch_env_reset_and_visit_first(env, idx_batches[i])
                term_states, term_timesteps, _ = batch_greedy_rollout(network, data_sharding, env, state_batch, timestep_batch)
                tour_lengths.append(- np.asarray(term_timesteps.reward))
            tour_lengths = np.concatenate(tour_lengths)
            mean_tour_length = tour_lengths.mean()
            mean_optimality_gap = np.mean((tour_lengths - env.generator.optimal_lengths[:len(tour_lengths)]) / env.generator.optimal_lengths[:len(tour_lengths)])
            print(f">> Validation epoch {epoch}. Mean tour length: {mean_tour_length:.3f}, mean optimality gap: {mean_optimality_gap:.5f}")

            # Save if the mean optimality gap has improved
            if mean_optimality_gap < current_validation_metric:
                print("Got new best model.")
                current_validation_metric = mean_optimality_gap
                destination_path = os.path.abspath(results_path)
                save_checkpoint(float(mean_optimality_gap), epoch, network, optimizer, "composite_checkpoint", destination_path)
