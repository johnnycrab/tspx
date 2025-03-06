import jax
import jax.numpy as jnp
from flax import nnx


class RZTXTransformerLayer(nnx.Module):
    """
    RZTXTransformerLayer is a Transformer block with residual weights for faster convergence,
    based on the paper https://arxiv.org/abs/2003.04887.
    """
    def __init__(self, latent_dim: int, num_heads: int, feedforward_dim: int, dropout: float = 0.0, rngs: nnx.Rngs | None = None):
        rngs = rngs or nnx.Rngs(0)

        self.self_attn = nnx.MultiHeadAttention(num_heads=num_heads, in_features=latent_dim, qkv_features=latent_dim,
                                                decode=False, rngs=rngs)
        # Feedforward model
        self.linear1 = nnx.Linear(latent_dim, feedforward_dim, rngs=rngs)
        self.dropout = nnx.Dropout(rate=dropout, rngs=rngs)
        self.linear2 = nnx.Linear(feedforward_dim, latent_dim, rngs=rngs)
        self.dropout1 = nnx.Dropout(rate=dropout, rngs=rngs)
        self.dropout2 = nnx.Dropout(rate=dropout, rngs=rngs)
        self.resweight = nnx.Param(jnp.zeros((1,)))

    def __call__(self, x: jax.Array, mask: jax.Array | None = None):
        """
        Args:
            x: input sequence of shape [batch_sizes..., length, latent_dim]
            mask: attention mask of shape [batch_sizes..., num_heads, length, length]. Attention masks are
                masked out if their corresponding mask value is `False`.
        """
        # Self-attention
        x2 = x
        x2 = self.self_attn(inputs_q=x2, inputs_k=x2, inputs_v=x2, mask=mask)
        x2 = x2 * self.resweight
        x = x + self.dropout(x2)

        # Pointwise FF layer
        x2 = x
        x2 = self.linear2(self.dropout1(nnx.gelu(self.linear1(x2))))
        x2 = x2 * self.resweight
        x = x + self.dropout2(x2)

        return x


class TSPNetwork(nnx.Module):
    """
    Transformer Network for the TSP, based on BQ-NCO (https://arxiv.org/abs/2301.03313)
    """
    def __init__(self, latent_dim: int, num_trf_blocks: int, num_attn_heads: int, feedforward_dim: int, dropout: float, rngs: nnx.Rngs):
        self.latent_dim, self.num_attn_heads, self.feedforward_dimension, self.dropout = latent_dim, num_attn_heads, \
                                                                                         feedforward_dim, dropout

        # Embedding of 2-dimensional node coordinates into latent sapce
        self.node_embedding = nnx.Linear(in_features=2, out_features=latent_dim, rngs=rngs)

        # Additive marker to indicate the start/destination node.
        # Gets added to the embedding of the start node (node with index 0) and the destination node (index 1), resp.
        self.start_marker = nnx.Param(jnp.zeros((latent_dim,)))
        self.destination_marker = nnx.Param(jnp.zeros((latent_dim,)))

        #self.trf_blocks = [
        #    RZTXTransformerLayer(latent_dim, num_attn_heads, feedforward_dim, dropout, rngs)
        #    for _ in range(num_trf_blocks)
        #]

        @nnx.vmap(in_axes=0, out_axes=0)
        def create_trf_block(key: nnx.Rngs):
            return RZTXTransformerLayer(latent_dim, num_attn_heads, feedforward_dim, dropout, nnx.Rngs(key))

        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry))
        def scan_trf_fn(y, block):
            the_seq, the_mask = y
            transformed_seq = block(the_seq, the_mask)
            return (transformed_seq, the_mask)

        self.scan_trf_fn = scan_trf_fn

        keys = jax.random.split(jax.random.key(0), num_trf_blocks)
        self.trf_blocks = create_trf_block(keys)

        # The transformed nodes get projected with a linear layer to their logit
        self.policy_linear_out = nnx.Linear(latent_dim, 1, rngs=rngs)

    def __call__(self, x: dict):
        """
        Parameters:
             x: dictionary with entries:
                "nodes": float array of shape (n+2, 2). The first entry is the start node (current position),
                    the second entry is the destination node (where the tour began, for TSP), nodes with idcs 2,...n+1
                    are the coordinates of all nodes.
                "mask": (boolean array of shape (n+2,), where element i = False if i has already been visited (excluding start
                    and destination nodes at index 0 and 1).
        """
        node_seq: jax.Array = x["nodes"]
        num_nodes = node_seq.shape[-2]
        node_seq = self.node_embedding(node_seq)  # Embed the nodes

        # Add start and destination marker to embedding at position 0 and 1 respectively.
        node_seq = node_seq.at[..., 0, :].set(node_seq[..., 0, :] + self.start_marker.value)
        node_seq = node_seq.at[..., 1, :].set(node_seq[..., 1, :] + self.destination_marker.value)

        mask: jax.Array = x["mask"]  # [batch_sizes..., num_nodes]

        # Repeat the mask for all queries and across all heads
        _mask = mask[..., None, None, :]
        _mask = jnp.repeat(_mask, repeats=num_nodes, axis=-2, total_repeat_length=num_nodes)  # repeat for rows
        _mask = jnp.repeat(_mask, repeats=self.num_attn_heads, axis=-3, total_repeat_length=self.num_attn_heads)  # repeat for heads

        #for trf_block in self.trf_blocks:
        #    node_seq = trf_block(node_seq, _mask)

        node_seq, _ = self.scan_trf_fn((node_seq, _mask), self.trf_blocks)

        logits = self.policy_linear_out(node_seq).squeeze(-1)

        big_neg = jnp.finfo(logits.dtype).min
        logits = jnp.where(mask, logits, big_neg)
        logits = logits[..., 2:]  # remove logits for start and destination

        return logits