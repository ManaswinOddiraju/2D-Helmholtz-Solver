import flax.nnx as nnx
import jax.numpy as jnp
from joblib import load, dump
import jax


def save_model(model, filepath):
    pass


def load_model(model, filepath):
    pass


class MLP(nnx.Module):
    def __init__(self, d_in, d_hidden, d_out, num_layers, rngs: nnx.Rngs):
        self.relu = nnx.leaky_relu
        self.activation = jnp.sin
        self.linear_in = nnx.Linear(d_in, d_hidden, rngs=rngs)
        self.hidden_layers = []
        for i in range(num_layers):
            self.hidden_layers.append(nnx.Linear(d_hidden, d_hidden, rngs=rngs))
        self.linear_out = nnx.Linear(d_hidden, d_out, rngs=rngs)

    def __call__(self, x):
        x = self.linear_in(x)
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        out = self.relu(self.linear_out(x))
        return out
