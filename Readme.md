# Solving the 2D Helmholtz Equation for Electric Fields
This repo contains codes both an FDFD solver as well as scripts for solving the 2d Helmholtz equation using a SIREN network.

## Problem definition
- Domain Dimensions: 600 units x 600 units
- Source: Point source (implemented as a continuous Gaussian)
- Boundary Conditions: SC-PML
- Dielectric: Circular region with $\epsilon=2$ with radius 60 units

## File Naming

train_pinn_no_source : PINN to predict $E_z$ as a function of $(x,y)$ with the point source at a fixed location.

train_pinn: PINN to predict $E_z$ as a function of $(x,y, s_x, s_y)$ where $s_x$ and $s_y$ are the source coordinates in 2D. 

pinn_utils: Contains the network modules required to train PINNs

fdfd: Contains functions to solve the 2D Helmholtz equation using finite differences.

FDFD.ipynb: Demo of the FDFD solver


## References

- Sitzmann, Vincent, Julien Martel, Alexander Bergman, David Lindell, and Gordon Wetzstein. "Implicit neural representations with periodic activation functions." Advances in neural information processing systems 33 (2020): 7462-7473.

- Shin, Wonseok, and Shanhui Fan. "Choice of the perfectly matched layer boundary condition for frequency-domain Maxwell’s equations solvers." Journal of Computational Physics 231, no. 8 (2012): 3406-3431.
