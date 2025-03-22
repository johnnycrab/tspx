# Utilities for the jumanji TSP environment
# See https://github.com/instadeepai/jumanji/tree/main

import numpy as np
import jax
import jax.numpy as jnp
import pickle

from jumanji.environments import TSP
from jumanji.environments.routing.tsp.types import State
from jumanji.environments.routing.tsp.generator import Generator


class InstancesFromFileGenerator(Generator):
    """
    TSP instance generator that loads TSP instances from a pickle file, similar to the TSPDataset in `dataset.py`.
    The generator method __call__ is called with the index of the instance to use (in contrast to passing a random
    key, as is done in the UniformGenerator used by jumanji).
    """

    def __init__(self, num_cities: int, path_to_pickle: str, custom_num_instances: int | None):
        super().__init__(num_cities)
        with open(path_to_pickle, "rb") as f:
            instances = pickle.load(f)
        if custom_num_instances is not None:
            instances = instances[:custom_num_instances]
        self.coordinates = [x["inst"] for x in instances]  # list of np.array of shape (num_cities, 2)
        self.optimal_lengths = np.array([x["sol"] for x in instances])
        self.length = len(self.coordinates)

    def __call__(self, key) -> State:
        # Get the coordinates of the cities at index `key`.
        coordinates = jnp.asarray(self.coordinates[key])

        # Initially, the position is set to -1, which means that the agent is not in any city.
        position = jnp.array(-1, jnp.int32)

        # Initially, the agent has not visited any city.
        visited_mask = jnp.zeros(self.num_cities, dtype=bool)
        trajectory = jnp.full(self.num_cities, -1, jnp.int32)

        # The number of visited cities is set to 0.
        num_visited = jnp.array(0, jnp.int32)

        state = State(
            coordinates=coordinates,
            position=position,
            visited_mask=visited_mask,
            trajectory=trajectory,
            num_visited=num_visited,
            key=key,
        )

        return state


def batch_env_reset_and_visit_first(tsp_env: TSP, index_keys: [int]):
    """
    Given a `tsp_env` that was created with an `InstancesFromFileGenerator`, creates a batch
    of TSP instances (coordinates given by a list of indices corresponding to the loaded instances), initializes
    them and then takes a first action by simply picking the first city.

    Returns the batch of resulting State and Timestep (batched pytree)
    """
    states = []
    for i in index_keys:
        state, _ = tsp_env.reset(i)
        states.append(state)

    state = jax.tree.map(lambda *v: jnp.stack(v), *states)  # Stack the pytrees, see https://gist.github.com/willwhitney/dd89cac6a5b771ccff18b06b33372c75
    state, timestep = jax.vmap(tsp_env.step, in_axes=(0, None))(state, 0)  # Pick first city
    return state, timestep


def state_to_network_rep(state: State):
    """
    Given a state, returns a representation that can be piped into the TSPNetwork. See `dataset.py` for more info
    about the structure of the returned dictionary.
    """
    nodes = state.coordinates
    mask = jnp.concatenate((jnp.ones(2, dtype=bool), ~state.visited_mask))
    start_node_idx = state.position   # last visited index
    destination_node_idx = state.trajectory[0]  # the index where we need to return to
    nodes = jnp.concatenate((
        nodes[start_node_idx][None, :],  # Start node comes first
        nodes[destination_node_idx][None, :],   # Then the destination node
        nodes   # Then simply all nodes again. Already visited ones will be masked.
    ), axis=0)
    return {"nodes": nodes, "mask": mask}