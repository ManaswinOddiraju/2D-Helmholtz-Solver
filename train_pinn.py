import flax.nnx as nnx
import jax.numpy as jnp
from joblib import load, dump
import optax
import jax_dataloader as jdl
import jax
import finite_diff
import pinn_utils
import numpy as np

OMEGA = 0.2


# def jac_fun(x, graph, params):
#     model = nnx.merge(graph, params)
#     return model(x)


def grad_with_pml(x, sx, sy, model):
    spatial_derivatives = jax.jacfwd(model)
    first_grad = spatial_derivatives(x)
    dx = jax.lax.complex(first_grad[0, 2], first_grad[1, 2])
    dy = jax.lax.complex(first_grad[0, 3], first_grad[1, 3])
    return jnp.array([dx * sy / sx, dy * sx / sy])


def helper_fn(x, model):
    sx, sy = finite_diff.s_func(x[2:].reshape(1, -1), d=30)
    jacobian = jax.jacfwd(grad_with_pml)(x, sx, sy, model).squeeze()
    ddx, ddy = jacobian[0, 2], jacobian[0, 3]
    return ddx + ddy, sx, sy


batch_grad_pml = jax.vmap(helper_fn, in_axes=(0, None), out_axes=(0, 0, 0))


## Train functions
@nnx.jit
def train_step(model, optimizer, x):
    def loss_fn(model):
        E = model(x)  # call methods directly
        res = jnp.mean(
            jnp.abs(
                (
                    pml_grad
                    + sx.ravel() * sy.ravel() * k * jax.lax.complex(E[:, 0], E[:, 1])
                    - source_term
                )
                ** 2
            )
        )
        return res

    k = (
        jnp.where(
            jnp.sqrt(jnp.sum(x[:, 2:] - jnp.array([400, 400]) ** 2, axis=1)) <= 25, 2, 1
        )
        * (OMEGA) ** 2
    )
    pml_grad, sx, sy = batch_grad_pml(x, model)
    source_term = finite_diff.build_vector(x[:, :2], x[:, 2:], omega=0.2)
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)  # inplace updates
    return loss


def train(model, optimizer, train_loader, epochs):

    for epoch in range(epochs):
        loss = []
        test_loss = []
        for x in train_loader:
            loss.append(train_step(model, optimizer, x[0]))
        for x, y in test_loader:
            out = model(x)
            test_loss.append(jnp.mean((out - y) ** 2))
        print(
            f"Epoch: {epoch} Loss: {np.mean(loss)} Test Loss (MSE):{np.mean(test_loss)}"
        )


## Load Data
data = load("train_data.joblib")
train_ds = jdl.ArrayDataset(data["train_inputs"])
test_ds = jdl.ArrayDataset(data["test_inputs"], data["test_outputs"])
train_loader = jdl.DataLoader(train_ds, "jax", batch_size=100000, shuffle=True)
test_loader = jdl.DataLoader(test_ds, "jax", batch_size=51005, shuffle=True)

## Training
key = jax.random.key(300)
model = pinn_utils.MLP(
    d_in=4, d_out=2, d_hidden=400, num_layers=4, rngs=nnx.Rngs(0)
)  # eager initialization
optimizer = nnx.Optimizer(model, optax.adam(1e-4))  # reference sharing
# Create an `ArrayDataset`
train(model, optimizer, train_loader, 1000)
