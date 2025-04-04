import flax.nnx as nnx
import jax.numpy as jnp
from joblib import load, dump
import optax
import jax_dataloader as jdl
import jax
import finite_diff
import pinn_utils
import numpy as np
import time

OMEGA = 0.2


def complex_grad(x, model):
    out = model(x)
    return jax.lax.complex(out[0], out[1])


def grad_with_pml(x, model):
    sx, sy = finite_diff.s_func(x[2:].reshape(1, -1), d=30)
    spatial_derivatives = jax.jacfwd(complex_grad)
    first_grad = spatial_derivatives(x, model)
    return jnp.array([first_grad[0] * sy / sx, first_grad[1] * sx / sy])


def helper_fn(x, model):
    jacobian = jax.jacfwd(grad_with_pml)(x, model).squeeze()
    ddx, ddy = (
        jacobian[0, 2],
        jacobian[1, 3],
    )  ## Extracting second derivatives w.r.t x and y
    return ddx + ddy


batch_grad_pml = jax.vmap(helper_fn, in_axes=(0, None), out_axes=(0))


## Train functions
@nnx.jit
def train_step_pinn(model, x, y=None):
    def loss_fn(model):
        E = model(x)  # call methods directly
        sx, sy = finite_diff.s_func(x[2:].reshape(1, -1), d=30)
        pml_grad = batch_grad_pml(x, model)
        res = jnp.mean(
            jnp.abs(
                (
                    pml_grad
                    + sx.ravel() * sy.ravel() * k * jax.lax.complex(E[:, 0], E[:, 1])
                    - source_term
                )
            )
            ** 2
        )
        return res

    k = (
        jnp.where(
            jnp.sqrt(jnp.sum(x[:, 2:] - jnp.array([400, 400]) ** 2, axis=1)) <= 25, 2, 1
        )
        * (OMEGA) ** 2
    )
    source_term = finite_diff.build_vector(x[:, :2], x[:, 2:], omega=0.2)
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
            pinn_utils.save_model(model, f"epoch_{epoch}")


## Load Data
data = load("train_data.joblib")
train_ds = jdl.ArrayDataset(data["train_inputs"])
test_ds = jdl.ArrayDataset(data["test_inputs"], data["test_outputs"])
##Avoid shuffling to keep samples from one source in one mini batch
train_loader = jdl.DataLoader(train_ds, "jax", batch_size=2000, shuffle=False)
test_loader = jdl.DataLoader(test_ds, "jax", batch_size=2000, shuffle=True)

key = jax.random.key(300)
model = pinn_utils.MLP(d_in=4, d_out=2, d_hidden=400, num_layers=5, rngs=nnx.Rngs(0))

## PINN Training
optimizer = nnx.Optimizer(model, optax.adam(1e-5))
train(model, optimizer, train_loader, train_step_pinn, 5000)
pinn_utils.save_model(model, "Weights/final_weights")
