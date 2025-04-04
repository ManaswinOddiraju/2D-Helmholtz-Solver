import flax.nnx as nnx
import jax.numpy as jnp
from joblib import load, dump
import jax


def save_model(model, filepath):
    _, state = nnx.split(model)
    dump(state, filepath + ".joblib")


def load_model(model, filepath):
    state = load(filepath + ".joblib")
    graph, _ = nnx.split(model)
    model = nnx.merge(graph, state)
    return model


class MLP(nnx.Module):
    def __init__(self, d_in, d_hidden, d_out, num_layers, rngs: nnx.Rngs):
        self.tanh = nnx.tanh
        self.leaky_relu = nnx.tanh
        self.coeff = nnx.Param(value=1.0)
        self.coeff2 = nnx.Param(value=10.0)
        self.activation = lambda x: jnp.sin(self.coeff * x)
        self.linear_in = nnx.Linear(d_in, d_hidden, rngs=rngs)
        self.hidden_layers = []
        for i in range(num_layers):
            self.hidden_layers.append(
                nnx.Linear(
                    d_hidden,
                    d_hidden,
                    rngs=rngs,
                    kernel_init=nnx.initializers.lecun_normal(),
                )
            )
        self.linear_out = nnx.Linear(d_hidden, d_out, rngs=rngs)

    def __call__(self, x):
        x = self.linear_in(x)
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        out = self.linear_out(x)
        return self.activation(out)  # self.coeff2 * self.activation(out)
