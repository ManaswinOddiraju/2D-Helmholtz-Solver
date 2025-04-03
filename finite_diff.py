import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

DOMAIN = jnp.array([600, 600])
PML_THICKNESS = 5


def get_epsilon(pts, center):
    dist = jnp.sqrt(jnp.sum((pts - center) ** 2, axis=1))
    epsilon = jnp.where(dist <= 25, 2, 1)
    return epsilon


def s_func(coord, d):
    def calc_s(l):
        sigma_max = 1.79  # (5 * 16) / (2 * ETA * d)
        sigma = sigma_max * ((l / d) ** 3)
        return jax.lax.complex(1.0, -(sigma))

    lx = jnp.maximum(coord[:, 0] - (DOMAIN[0] - d), -(coord[:, 0] - d))
    ly = jnp.maximum(coord[:, 1] - (DOMAIN[0] - d), -(coord[:, 1] - d))
    sx = jnp.where(lx >= 0, calc_s(lx), jax.lax.complex(1.0, 0.0))
    sy = jnp.where(ly >= 0, calc_s(ly), jax.lax.complex(1.0, 0.0))
    return sx, sy


def build_mesh(nx, dielectric_loc, omega):
    pts_x = DOMAIN[0] * jnp.linspace(0, 1, nx)
    pts_y = DOMAIN[1] * jnp.linspace(0, 1, nx)
    grid_x, grid_y = jnp.meshgrid(pts_x, pts_y)
    inds = jnp.arange(nx**2).reshape(nx, nx)
    coords = jnp.hstack((grid_x.reshape(-1, 1), grid_y.reshape(-1, 1)))
    epsilon = get_epsilon(coords, dielectric_loc)
    dx = DOMAIN[0] / (nx - 1)

    sx, sy = s_func(coords, PML_THICKNESS * dx)
    a = sy / sx
    b = sx / sy

    def get_neighbors(index):
        i, j = index // nx, index % nx
        nc1 = jnp.where(
            i > 0,
            jnp.array([inds[i - 1, j], 0.5 * (b[index] + b[inds[i - 1, j]])]),
            jnp.zeros(2, dtype=jnp.complex128),
        )
        nc2 = jnp.where(
            i < nx - 1,
            jnp.array(
                [
                    inds[i + 1, j],
                    0.5 * (b[index] + b[inds[i + 1, j]]),
                ]
            ),
            jnp.zeros(2, dtype=jnp.complex128),
        )
        nc3 = jnp.where(
            j > 0,
            jnp.array(
                [
                    inds[i, j - 1],
                    0.5 * (a[index] + a[inds[i, j - 1]]),
                ]
            ),
            jnp.zeros(2, dtype=jnp.complex128),
        )
        nc4 = jnp.where(
            j < nx - 1,
            jnp.array(
                [
                    inds[i, j + 1],
                    0.5 * (a[index] + a[inds[i, j + 1]]),
                ]
            ),
            jnp.zeros(2, dtype=jnp.complex128),
        )

        neighbors = jnp.array(
            [jnp.real(nc1[0]), jnp.real(nc2[0]), jnp.real(nc3[0]), jnp.real(nc4[0])],
            dtype=jnp.int64,
        )
        coeffs = jnp.array([nc1[1], nc2[1], nc3[1], nc4[1]])
        # coeffs = jnp.array(
        #     [
        #         4 * b[index] + nc1[1] + nc2[1],
        #         4 * b[index] + nc1[1] + nc2[1],
        #         4 * a[index] + nc3[1] + nc4[1],
        #         4 * a[index] + nc3[1] + nc4[1],
        #         4 * a[index] + nc3[1] + nc4[1] + 4 * b[index] + nc1[1] + nc2[1],
        #     ]
        # )

        return neighbors, (coeffs / (dx**2))

    conn, coeffs = jax.vmap(get_neighbors, out_axes=(0))(jnp.arange(nx**2))

    diag_coeff = (sx * sy * epsilon.ravel() * (omega**2)) - ((2 / dx**2) * (a + b))

    return coords, conn, coeffs[:, :4], diag_coeff, epsilon


def build_operator(nx, conn, coeffs, diag_coeff):
    def ret_row(conn, coeffs):
        tmp_row = jnp.zeros((nx**2), dtype=jnp.complex128)
        return tmp_row.at[conn].set(coeffs)

    build_fn = jax.vmap(
        ret_row,
        in_axes=(0, 0),
    )
    mat = build_fn(conn, coeffs)
    diag_mat = jnp.diag(diag_coeff)
    return mat + diag_mat


def build_vector(source_loc, coords, omega):
    dist = jnp.sum((coords - source_loc) ** 2, axis=1) / (2 * 6**2)
    return jax.lax.complex(0.0, omega * jnp.exp(-dist))
