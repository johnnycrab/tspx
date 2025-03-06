import pickle
import numpy as np
import torch
from jax import tree_map
from torch.utils import data
from typing import Tuple, Optional

from tsp_geometry import TSPGeometry

def numpy_collate(batch):
    return tree_map(np.asarray, data.default_collate(batch))

class TSPDataset(data.Dataset):
    """
    Dataset for supervised training of TSP given expert solutions (i.e., optimal solutions).
    Each problem instance is a dictionary of the form { "inst": np.array: (<num_nodes>,2), "tour": list (<num_nodes),
    "sol": float }.
    Let n be the size of the TSP instances (here, n=100). We samples subtours of size k, where 4 <= k <= n+1. A subtour
    is a continuous sequence of cities of the optimum round trip. In such a subtour, the first entry corresponds to
    the start node, the last entry corresponds to the destination node. In case k=n+1, the start and destination nodes
    are the same.

    We pad the nodes to the maximum subtour length to guarantee constant batch shapes.
    The dataset represents a single, unbatched point by a dictionary of the form:
    {
        "nodes": (float tensor of shape (n+2, 2)) Given a subtour of length 4 <= k <= n+1,
            idx 0 is the start node, idx 1 is the destination node, idcs 2 to k are the nodes in between, idcs k+1 to n+1
            is padding.
        "mask": (boolean tensor of shape (n+2,), where element i = False if i is a padded index.
        "next_node_idx": Index of target node to choose in subtour. Since later in the model we exclude start and destination node,
            the next_node_idx will always be 0 (i.e., the first unvisited node in the optimal subtour)
    }
    """
    def __init__(self, path_to_pickle: str,
                 data_augmentation: bool = False, data_augmentation_linear_scale: bool = False,
                 augment_direction: bool = False, custom_num_instances: Optional[int] = None,
                 ):
        """
        Parameters:
            path_to_pickle [str]: Path to file with expert trajectories.
                We assume that the number of cities in each instance is constant.
            data_augmentation [bool]: If True, a random
                geometric augmentation (rotation, flipping) is performed on the instance
                before choosing a subtour.
            data_augmentation_linear_scale [bool]: If True, linear scaling is performed in geometric augmentation.
                Not that this changes the distribution of the points.
            augment_direction [bool]: If True, direction of the subtour is randomly swapped.
            custom_num_instances [int]: If given, only the first num instances are taken from `path_to_pickle`.
        """
        self.data_augmentation = data_augmentation
        self.data_augmentation_linear_scale = data_augmentation_linear_scale
        self.augment_direction = augment_direction

        with open(path_to_pickle, "rb") as f:
            self.instances = pickle.load(f)

        if custom_num_instances is not None:
            self.instances = self.instances[:custom_num_instances]

        self.num_nodes = self.instances[0]["inst"].shape[0]  # number of cities in each instance.

        # Calculate the length of the dataset. We have countless options of flipping / rotation / random order
        # of subtour and taking start/end nodes. Disregarding geometric augmentation and the order, we say for each
        # instance, there are n possibilities for the start node. For each start node, there are n-2 options for choosing
        # 4 <= k <= n+1.
        self.length = len(self.instances) * self.num_nodes * (self.num_nodes - 2)

        print(f"Loaded dataset from {path_to_pickle}. Num instances: {len(self.instances)}. Total length: {self.length}")

    def __len__(self):
        return self.length

    def subtour_location_from_idx(self, idx: int):
        """
        Unfolds the index of the instance, the index of the start node and the length k of the subtour from `idx`.
        """
        # Unfold the index of the instance, the start node and the subtour length from `idx`.
        instance_idx = idx // (self.num_nodes * (self.num_nodes - 2))
        idx = idx - instance_idx * (self.num_nodes * (self.num_nodes - 2))
        start_node_idx = idx // (self.num_nodes - 2)
        k = idx - start_node_idx * (self.num_nodes - 2) + 4  # subtour length 4 <= k <= n+1
        assert 4 <= k <= self.num_nodes + 1
        return instance_idx, start_node_idx, k

    def __getitem__(self, idx: int):
        """
        See class docstring for description of a single item.
        """
        instance_idx, start_node_idx, k = self.subtour_location_from_idx(idx)

        nodes = self.instances[instance_idx]["inst"]
        optimal_full_tour = list(self.instances[instance_idx]["tour"])

        subtour = optimal_full_tour[start_node_idx: start_node_idx + k]
        if len(subtour) < k:  # wrap around
            subtour = subtour + optimal_full_tour[0: k - len(subtour)]

        if self.augment_direction and np.random.randint(0, 2) == 0:
            # reverse order of tour
            subtour = subtour[::-1]

        if self.data_augmentation:
            nodes = TSPGeometry.random_state_augmentation(nodes, do_linear_scale=self.data_augmentation_linear_scale)

        # Move the destination node to index 1, and pad with zeros to length n+1
        nodes = nodes[subtour]

        num_padded_nodes = self.num_nodes + 2 - k
        nodes = np.concatenate((
            nodes[:1],
            nodes[-1:],
            nodes[1:-1],
            np.zeros((num_padded_nodes, 2))
        ), axis=0
        )

        # Padded nodes mask for attention
        mask = np.pad(
            np.ones(k, dtype=bool),
            pad_width=((0, num_padded_nodes),), mode='constant', constant_values=0
        )

        return {
            "nodes": torch.from_numpy(nodes),
            "mask": torch.from_numpy(mask).bool(),
            "next_node_idx": torch.LongTensor([0])  # The target idx to choose is - when discarding the first and last node - always 0!
        }
