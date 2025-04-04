from scipy.stats import qmc
import jax.numpy as jnp
import jax

jax.config.update("jax_platform_name", "cpu")
from joblib import load, dump
import finite_diff


nx = 101
omega = 0.2
coords, conn, coeffs, diag_coeff, epsilon = finite_diff.build_mesh(
    nx, omega=omega, dielectric_loc=jnp.array([400, 400])
)
A = finite_diff.build_operator(nx, conn, coeffs, diag_coeff)
source = jnp.array([200, 400])
b = finite_diff.build_vector(source, coords, omega=omega)
out_field = jnp.linalg.solve(A, b)
test_outputs = jnp.hstack(
    (jnp.real(out_field).reshape(-1, 1), jnp.imag(out_field).reshape(-1, 1))
)

grid_sampler = qmc.LatinHypercube(d=2, seed=6789)
grid_samples = 600 * grid_sampler.random(n=15000)
grid_samples = jnp.vstack((grid_samples, source))
# pts_x, pts_y = jnp.meshgrid(jnp.linspace(0, 600, 51), jnp.linspace(0, 600, 51))
# grid_samples = jnp.hstack((pts_x.reshape(-1, 1), pts_y.reshape(-1, 1)))

dump(
    {
        "train_inputs": grid_samples.astype(jnp.float32) / 600,
        "test_inputs": coords.astype(jnp.float32) / 600,
        "test_outputs": test_outputs.astype(jnp.float32),
    },
    "no_source_train_data.joblib",
)
