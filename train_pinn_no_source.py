import flax.nnx as nnx
import jax.numpy as jnp
from joblib import load, dump
import optax
import jax_dataloader as jdl
import jax

jax.config.update("jax_enable_x64", True)
import finite_diff
import pinn_utils
import numpy as np
import time

OMEGA = 0.2
SOURCE = jnp.array([200, 400]).reshape(1, 2)


def complex_grad(x, model):
    out = model(x)
    return jax.lax.complex(out[0], out[1])


def grad_with_pml(x, model):
    sx, sy = finite_diff.s_func(600 * x.reshape(1, -1), d=30)
    spatial_derivatives = jax.jacfwd(complex_grad)
    first_grad = spatial_derivatives(x, model)
    return first_grad * jnp.array([sy / sx, sx / sy]).ravel()


def helper_fn(x, model):
    jacobian = jax.jacfwd(grad_with_pml)(x, model).squeeze()
    return jnp.sum(jnp.diag(jacobian))


batch_grad_pml = jax.vmap(helper_fn, in_axes=(0, None), out_axes=(0))


## Residual Loss
# @nnx.jit
# def train_step_fd(model, x, y):
#     def loss_fn(model):
#         E = model(x)
#         return jnp.mean(
#             jnp.abs(
#                 (A @ jax.lax.complex(E[:, 0], E[:, 1]).reshape(-1, 1) - source_v) ** 2
#             )
#         )

#     loss, grads = nnx.value_and_grad(loss_fn)(model)
#     return loss, grads


@nnx.jit
def train_step_pinn(model, x, y=None):
    def loss_fn(model):
        E = model(x)  # call methods directly
        # c = model(source)
        # max_penalty = jnp.maximum(
        #     jnp.max(jnp.sum(E**2, axis=1)) - jnp.sum(c**2, axis=1), -1
        # )[0] # Penalty to ensure field magnitude at source location is highest
        sx, sy = finite_diff.s_func(600 * x.reshape(1, -1), d=30)
        pml_grad = batch_grad_pml(x, model)
        k = (
            jnp.where(
                jnp.sqrt(jnp.sum((600 * x - jnp.array([350, 200])) ** 2, axis=1)) <= 60,
                2,
                1,
            )
            * (OMEGA) ** 2
        )
        res = jnp.mean(
            lmbda
            * (pml_grad + sx * sy * k * jax.lax.complex(E[:, 0], E[:, 1]) - source_term)
            ** 2
        )
        return jnp.abs(res)  # + 10 * max_penalty

    source_term = finite_diff.build_vector(source_loc=SOURCE, coords=600 * x, omega=0.2)
    lmbda = jnp.where(source_term > 1e-2, 2, 1)
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    return loss, grads


def train(model, optimizer, train_loader, train_step, epochs):
    for epoch in range(epochs):
        t1 = time.time()
        loss = []
        test_loss = []
        test_loss_mse = []
        for x in train_loader:
            tmp_loss, grads = train_step(model, *x)
            optimizer.update(grads)
            loss.append(tmp_loss)
        for x, y in test_loader:
            out = model(x)
            tmp_loss, _ = train_step(model, x, y)
            test_loss.append(tmp_loss)
            test_loss_mse.append(jnp.mean((out - y) ** 2))
        t2 = time.time()
        print(
            f"Epoch:{epoch} Time:{t2-t1} Loss:{np.mean(loss)} Test Loss:{np.mean(test_loss)} Test Loss (MSE):{np.mean(test_loss_mse)}"
        )
        if epoch > 0 and epoch % 100 == 0:
            pinn_utils.save_model(model, f"Weights/epoch_{epoch}")


## Load Data
data = load("no_source_train_data.joblib")
train_ds = jdl.ArrayDataset(data["train_inputs"])
test_ds = jdl.ArrayDataset(
    data["test_inputs"][:4000, :], data["test_outputs"][:4000, :]
)

train_loader = jdl.DataLoader(
    train_ds, "jax", batch_size=5000, shuffle=False, drop_last=True
)
test_loader = jdl.DataLoader(
    test_ds, "jax", batch_size=100, shuffle=False, drop_last=True
)
single_loader = jdl.DataLoader(
    jdl.ArrayDataset(
        jnp.array([200.0, 400.0]).reshape(1, 2) / 600,
    ),
    "jax",
    batch_size=1,
)

key = jax.random.key(300)
model = pinn_utils.MLP(d_in=2, d_out=2, d_hidden=500, num_layers=5, rngs=nnx.Rngs(0))


optimizer = nnx.Optimizer(model, optax.adam(1e-5))  # reference sharing
## Initializing by training only at source location
train(model, optimizer, single_loader, train_step_pinn, 100)
print(model(jnp.array(jnp.array([200.0, 400.0]).reshape(1, 2) / 600)))

## Main Training
train(model, optimizer, train_loader, train_step_pinn, 10000)
pinn_utils.save_model(model, "Weights/final_weights")
