import plotly.graph_objects as go
import jax.numpy as jnp
from finite_diff import *

nx = 11
source = jnp.array([138, 202])
epsilon = 1 * jnp.ones((nx, nx))
omega = 0.2
for i in range(65, 85):
    for j in range(65, 85):
        epsilon = epsilon.at[i, j].set(2.0)
coords, conn, coeffs, diag_coeff = build_mesh(nx, epsilon, omega=omega)
A = build_operator(nx, conn, coeffs, diag_coeff)
b = build_vector(source, coords, omega=omega)
out_field = jnp.linalg.solve(A, b)
