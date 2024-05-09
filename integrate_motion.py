import lagrangian as lag
import torch

from itertools import product

def differential(model, x) -> torch.Tensor:
    q_dot_dot = lag.q_dot_dot(model, x)

    d_q_dot = q_dot_dot
    d_q = x[1]

    return torch.tensor([d_q, d_q_dot], dtype=torch.float32)

def lagrangian_path(model, x):
    dt = 0.01

    path = []

    #uses range kutta 4
    #model yields acceleration - q_dot_dot.
    #for initial (q, q_dot), find path of q, q_dot
    for _ in range(800):
        path.append(x.detach())

        k1 = differential(model, x)
        k2 = differential(model, x + dt/2 * k1)
        k3 = differential(model, x + dt/2 * k2)
        k4 = differential(model, x + dt * k3)

        x = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        

    return torch.stack(path)
