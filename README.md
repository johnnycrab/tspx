# Traveling Salesman Problem – Supervised Learning in JAX/FLAX

I am gradually moving Gumbeldore (https://github.com/grimmlab/gumbeldore) and TASAR (https://github.com/grimmlab/step-and-reconsider/)
from PyTorch to JAX.

In this repository, we implement supervised learning for the TSP with 100 nodes, with
the BQ-NCO Transformer architecture.

We base the TSP environment on Jumanji (https://github.com/instadeepai/jumanji/).

## Steps to do

- First, we build a PyTorch Dataloader that loads in the batches but collates them to numpy
- We write the Transformer module in NNX
- Training loop written in NNX