from scipy.stats import qmc
import jax.numpy as jnp
from joblib import load, dump
import finite_diff


nx = 101
omega = 0.2
coords, conn, coeffs, diag_coeff, epsilon = finite_diff.build_mesh(
    nx, omega=omega, dielectric_loc=jnp.array([400, 400])
)

source_domain = jnp.array(
    [[30, 30], [570, 570]]
)  # reduced domain to avoid having source too close to boundary

sampler = qmc.LatinHypercube(d=2, seed=345)
test_sources = source_domain[0, :] + (
    source_domain[1, :] - source_domain[0, :]
) * sampler.random(n=5)

test_outputs = jnp.zeros((test_sources.shape[0] * 101**2, 2))
for i in range(test_sources.shape[0]):
    A = finite_diff.build_operator(nx, conn, coeffs, diag_coeff)
    b = finite_diff.build_vector(test_sources[i, :], coords, omega=omega)
    out_field = jnp.linalg.solve(A, b)
    test_outputs = test_outputs.at[i * 101**2 : (i + 1) * 101**2, 0].set(
        jnp.real(out_field)
    )
    test_outputs = test_outputs.at[i * 101**2 : (i + 1) * 101**2, 1].set(
        jnp.imag(out_field)
    )


test_samples = jnp.hstack(
    (jnp.repeat(test_sources, coords.shape[0], axis=0), jnp.tile(coords, (5, 1)))
)

source_sampler = qmc.LatinHypercube(d=2, seed=6789)
grid_sampler = qmc.LatinHypercube(d=2, seed=6789)
train_sources = sampler.random(n=100)
grid_samples = grid_sampler.random(n=5000)

train_samples = jnp.hstack(
    (
        jnp.repeat(train_sources, grid_samples.shape[0], axis=0),
        jnp.tile(grid_samples, (train_sources.shape[0], 1)),
    )
)

dump(
    {
        "train_inputs": train_samples,
        "test_inputs": test_samples,
        "test_outputs": test_outputs,
    },
    "train_data.joblib",
)
