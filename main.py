import time

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from dataset import TSPDataset
from torch.utils.data import DataLoader
from network import TSPNetwork
from functools import partial
from tqdm import tqdm

from jumanji.environments import TSP
from jumanji.environments.routing.tsp.reward import SparseReward
from jumanji.environments.routing.tsp.types import State
from env_utils import InstancesFromFileGenerator, batch_env_reset_and_visit_first, state_to_network_rep
from jumanji.types import TimeStep

seed = 43
TRAIN_DATASET_PATH = "./data/tsp_100_10k.pickle"
VAL_DATASET_PATH = "./data/tsp_100_10k.pickle"
learning_rate = 2e-4
num_epochs = 100
num_batches_per_epoch = 50
batch_size = 32
validation_batch_size = 25

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


@nnx.jit
def log_probability_fn(model: TSPNetwork, states: State):
    """
    Given a batch of states and TSPNetwork, return logits
    for the next actions. Function is jitted, so we should make sure to
    not vary the batch size.
    """
    logits = model(
        jax.vmap(state_to_network_rep)(states)
    )
    return nnx.log_softmax(logits)


@partial(nnx.jit, static_argnames=("env_wrapper"))
def batch_greedy_rollout(model: TSPNetwork, env_wrapper: TSP, initial_states: State, initial_timesteps: TimeStep) -> tuple[State, TimeStep, TSPNetwork]:
    """
    Greedy rollout given a batch of states and a model.

    Args:
        model: TSPNetwork instance
        env_wrapper: TSP environment from which we can call the step_fn
        initial_states: Batch of TSP states given as PyTree, initialized where first action has already been taken
        initial_timesteps: Batch of TSP timesteps given as PyTree, initialized where first action has already been taken

    Returns:
        Batch of final states and timesteps
    """
    step_fn = jax.vmap(env_wrapper.step)

    def body_fun(input):
        states, timesteps, network = input
        logits = log_probability_fn(network, states)  # batched logits
        # get argmax of logits, corresponding to greedy action
        actions = jnp.argmax(logits, axis=-1)
        new_states, new_timesteps = step_fn(states, actions)
        return new_states, new_timesteps, network

    return nnx.while_loop(
        cond_fun=lambda x: x[1].reward[0] == 0,  # we continue until we have obtained a nonzero reward in a timestep.
        body_fun=body_fun,
        init_val=(initial_states, initial_timesteps, model)  # don't forget to also pass in the network as initial value
    )


if __name__ == '__main__':
    # Create a mesh of two dimensions and annotate each axis with a name.
    mesh = Mesh(devices=np.array(jax.devices()).reshape(len(jax.devices()),),
                axis_names=('data'))
    # In data parallelism, the first dimension (batch) will be sharded on the `data` axis.
    data_sharding = NamedSharding(mesh, PartitionSpec('data', None))

    # Prepare dataset and dataloader
    train_dataset = TSPDataset(path_to_pickle=TRAIN_DATASET_PATH,
                               data_augmentation=True,
                               data_augmentation_linear_scale=True,
                               augment_direction=True,
                               custom_num_instances=None
                               )
    loader = DataLoader(train_dataset,
                        batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True,
                        prefetch_factor=2, pin_memory=True, persistent_workers=True)

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

    # Jumanji TSP environment generator used for validation (where we perform greedy rollouts)
    env = TSP(
        generator=InstancesFromFileGenerator(num_cities=100, path_to_pickle=VAL_DATASET_PATH),
        reward_fn=SparseReward()
    )

    # Training loop
    for epoch in range(1, num_epochs + 1):
        with mesh:
            network.train()
            data_iter = iter(loader)
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
                jax.debug.visualize_array_sharding(state_batch.coordinates[:, :, 0])
                term_states, term_timesteps, _ = batch_greedy_rollout(network, env, state_batch, timestep_batch)
                tour_lengths.append(- np.asarray(term_timesteps.reward))
            tour_lengths = np.concatenate(tour_lengths)
            mean_tour_length = tour_lengths.mean()
            mean_optimality_gap = np.mean((tour_lengths - env.generator.optimal_lengths) / env.generator.optimal_lengths)
            print(f">> Validation epoch {epoch}. Mean tour length: {mean_tour_length:.3f}, mean optimality gap: {mean_optimality_gap:.5f}")